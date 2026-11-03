//! `engine selfplay --model <onnx> --games <N> --sims 200 --out <bin>
//!  [--seed S] [--max-plies P] [--temp-plies T]`
//!
//! Root Dirichlet noise (alpha = 0.3, eps = 0.25) at every move. Each game is
//! flushed to disk as soon as it finishes; game `i` uses seed `S + i`.

use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use rand::rngs::StdRng;
use rand::SeedableRng;

use super::bin_writer::SampleWriter;
use crate::nn::ort_infer::load_evaluator;
use crate::nn::Evaluator;
use crate::selfplay::{game_samples, play_game, SelfPlayConfig};

pub struct SelfPlayArgs {
    pub model: PathBuf,
    pub games: u32,
    pub out: PathBuf,
    pub seed: u64,
    pub cfg: SelfPlayConfig,
}

fn parse_num<T: std::str::FromStr>(it: &mut std::slice::Iter<String>, flag: &str) -> Result<T> {
    let v = it.next().with_context(|| format!("{flag} requires a number"))?;
    v.parse().ok().with_context(|| format!("bad {flag} value {v:?}"))
}

pub fn parse_args(args: &[String]) -> Result<SelfPlayArgs> {
    let mut cfg = SelfPlayConfig::default();
    let (mut model, mut out, mut games, mut seed) = (None, None, 1u32, None);
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = Some(it.next().context("--model requires a path")?.into()),
            "--out" => out = Some(it.next().context("--out requires a path")?.into()),
            "--games" => games = parse_num(&mut it, "--games")?,
            "--sims" => cfg.sims = parse_num(&mut it, "--sims")?,
            "--seed" => seed = Some(parse_num(&mut it, "--seed")?),
            "--max-plies" => cfg.max_plies = parse_num(&mut it, "--max-plies")?,
            "--temp-plies" => cfg.temp_plies = parse_num(&mut it, "--temp-plies")?,
            other => bail!("unknown argument for selfplay: {other}"),
        }
    }
    let seed = seed.unwrap_or_else(|| {
        SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64)
    });
    Ok(SelfPlayArgs {
        model: model.context("selfplay requires --model <onnx>")?,
        out: out.context("selfplay requires --out <bin>")?,
        games,
        seed,
        cfg,
    })
}

/// Play `games` games into `out`; returns the number of samples written.
pub fn generate(eval: &mut dyn Evaluator, games: u32, seed: u64, cfg: &SelfPlayConfig, out: &Path) -> Result<u64> {
    let mut writer = SampleWriter::create(out)?;
    for i in 0..games {
        let start = Instant::now();
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
        let game = play_game(eval, cfg, &mut rng)?;
        for s in &game_samples(&game) {
            writer.write(s)?;
        }
        writer.flush()?;
        eprintln!(
            "selfplay: game {}/{games}: {} plies, {:?} ({:?}), {:.1}s, {} samples total",
            i + 1,
            game.plies.len(),
            game.result,
            game.termination,
            start.elapsed().as_secs_f32(),
            writer.count()
        );
    }
    writer.finish()
}

pub fn run(args: &[String]) -> Result<()> {
    let a = parse_args(args)?;
    let mut eval = load_evaluator(Some(&a.model))?;
    eprintln!(
        "selfplay: model {}, {} games, {} sims, seed {}",
        a.model.display(),
        a.games,
        a.cfg.sims,
        a.seed
    );
    let n = generate(eval.as_mut(), a.games, a.seed, &a.cfg, &a.out)?;
    println!("wrote {n} samples to {}", a.out.display());
    Ok(())
}
