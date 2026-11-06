//! Paired, deterministic evaluation matches and SPEC §9 output files.

use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{ensure, Context, Result};

use crate::eval::match_runner::{run_match, MatchConfig};
use crate::eval::openings::{openings, parse_book};
use crate::nn::infer_opts::InferOptions;

const USAGE: &str = "usage: engine match --a <PLAYER> --b <PLAYER> --games <N even>
    --sims <S> --seed <X> --out <match.json>
    [--openings <file>] [--opening-offset 0] [--max-plies 300] [--batch 8]
    [--moves-out <txt>] [--ep auto|cpu|coreml] [--threads 1]
PLAYER is an ONNX path, uniform, material, or random.
Opening books contain 50 distinct legal UCI sequences of 2–8 plies.
The ply cap includes opening moves and must accommodate each selected opening.";

struct MatchArgs {
    config: MatchConfig,
    out: PathBuf,
    moves_out: Option<PathBuf>,
    openings: Option<PathBuf>,
}

fn number<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T> {
    value
        .parse()
        .ok()
        .with_context(|| format!("bad {flag} value {value:?}"))
}

fn parse_args(args: &[String]) -> Result<MatchArgs> {
    let (mut a, mut b, mut games, mut sims, mut seed, mut out) =
        (None, None, None, None, None, None);
    let (mut moves_out, mut openings) = (None, None);
    let (mut opening_offset, mut max_plies, mut batch) = (0, 300, 8);
    let mut infer = InferOptions::default();
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        ensure!(
            matches!(
                flag.as_str(),
                "--a" | "--b" | "--games" | "--sims" | "--seed" | "--out"
                    | "--openings" | "--opening-offset" | "--max-plies" | "--batch"
                    | "--moves-out" | "--ep" | "--threads"
            ),
            "unknown argument for match: {flag}"
        );
        let value = it.next().with_context(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--a" => a = Some(value.clone()),
            "--b" => b = Some(value.clone()),
            "--games" => games = Some(number(flag, value)?),
            "--sims" => sims = Some(number(flag, value)?),
            "--seed" => seed = Some(number(flag, value)?),
            "--out" => out = Some(PathBuf::from(value)),
            "--moves-out" => moves_out = Some(PathBuf::from(value)),
            "--openings" => openings = Some(PathBuf::from(value)),
            "--opening-offset" => opening_offset = number(flag, value)?,
            "--max-plies" => max_plies = number(flag, value)?,
            "--batch" => batch = number(flag, value)?,
            "--ep" | "--threads" => {
                infer.parse_flag(flag, value)?;
            }
            _ => unreachable!(),
        }
    }
    let out = out.context("match requires --out <match.json>")?;
    ensure!(!out.as_os_str().is_empty(), "--out must not be empty");
    if let Some(path) = &moves_out {
        ensure!(!path.as_os_str().is_empty(), "--moves-out must not be empty");
        ensure!(
            resolved_output(path)? != resolved_output(&out)?,
            "--out and --moves-out must differ"
        );
    }
    Ok(MatchArgs {
        config: MatchConfig {
            a: a.context("match requires --a <PLAYER>")?,
            b: b.context("match requires --b <PLAYER>")?,
            games: games.context("match requires --games <N even>")?,
            sims: sims.context("match requires --sims <S>")?,
            seed: seed.context("match requires --seed <X>")?,
            opening_offset,
            max_plies,
            batch,
            infer,
        },
        out,
        moves_out,
        openings,
    })
}

// Normalize missing path components while resolving existing symlink ancestors.
fn resolved_output(path: &Path) -> Result<PathBuf> {
    let mut resolved = PathBuf::new();
    for part in std::path::absolute(path)?.components() {
        match part {
            Component::CurDir => {}
            Component::ParentDir => { resolved.pop(); }
            _ => {
                resolved.push(part.as_os_str());
                if let Ok(canonical) = fs::canonicalize(&resolved) {
                    resolved = canonical;
                }
            }
        }
    }
    Ok(resolved)
}

fn write_atomic(path: &Path, contents: &str, other_output: Option<&Path>) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let reserved = other_output.map(resolved_output).transpose()?;
    let mut staged = None;
    for _ in 0..100 {
        let mut tmp = path.as_os_str().to_os_string();
        tmp.push(format!(".{:016x}.tmp", rand::random::<u64>()));
        let tmp = PathBuf::from(tmp);
        if reserved.as_ref() == Some(&resolved_output(&tmp)?) {
            continue;
        }
        match OpenOptions::new().write(true).create_new(true).open(&tmp) {
            Ok(file) => { staged = Some((tmp, file)); break; }
            Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error).with_context(|| format!("staging {}", path.display())),
        }
    }
    let (tmp, mut file) = staged.context("could not create a unique output temporary file")?;
    let result = (|| {
        file.write_all(contents.as_bytes())
            .with_context(|| format!("writing {}", tmp.display()))?;
        file.sync_all().with_context(|| format!("synchronizing {}", tmp.display()))?;
        drop(file);
        fs::rename(&tmp, path).with_context(|| format!("publishing {}", path.display()))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

pub fn run(args: &[String]) -> Result<()> {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let a = parse_args(args)?;
    let book = match &a.openings {
        Some(path) => {
            let text = fs::read_to_string(path)
                .with_context(|| format!("reading opening book {}", path.display()))?;
            parse_book(&text).with_context(|| format!("opening book {}", path.display()))?
        }
        None => openings(),
    };
    let report = run_match(&a.config, &book)?;
    let json = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(path) = &a.moves_out {
        let mut text = String::new();
        for game in &report.results {
            text.push_str(&format!("{} {} {}", game.game, game.white, game.result));
            for mv in &game.moves {
                text.push(' ');
                text.push_str(mv);
            }
            text.push('\n');
        }
        write_atomic(path, &text, Some(&a.out))?;
    }
    write_atomic(&a.out, &json, a.moves_out.as_deref())?;
    eprintln!(
        "match: {} games, A {} wins, B {} wins, {} draws, A score {:.3}",
        report.games, report.a_wins, report.b_wins, report.draws, report.a_score
    );
    Ok(())
}
