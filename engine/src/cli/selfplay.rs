//! `engine selfplay [--model <onnx>|uniform] --games <N> --sims 200 --out <bin>
//!  [--seed S] [--max-plies P] [--temp-plies T] [--no-reuse] [--ep auto|cpu|coreml] [--threads T]
//!  [--parallel P] [--leaf-batch K] [--infer-batch N]
//!  [--resign-threshold -0.95] [--resign-consecutive 3] [--resign-playthrough 0.1]`
//!
//! Root Dirichlet noise (alpha = 0.3, eps = 0.25) at every move. Each game is
//! flushed in game-index order; game `i` uses seed `S + i`.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};

use super::chunk_writer::ChunkWriter;
use crate::nn::infer_opts::InferOptions;
use crate::nn::ort_infer::load_evaluator;
use crate::nn::Evaluator;
use crate::selfplay::batch_runner::{run_games_with_progress, BatchOptions};
use crate::selfplay::progress::{ProgressPhase, ProgressReporter};
use crate::selfplay::stats::SelfPlayStats;
use crate::selfplay::{game_samples, SelfPlayConfig};

pub struct SelfPlayArgs {
    pub model: Option<PathBuf>,
    pub infer_opts: InferOptions,
    pub games: u32,
    pub out: PathBuf,
    pub seed: u64,
    pub cfg: SelfPlayConfig,
    pub batch: BatchOptions,
}

fn parse_num<T: std::str::FromStr>(it: &mut std::slice::Iter<String>, flag: &str) -> Result<T> {
    let v = it
        .next()
        .with_context(|| format!("{flag} requires a number"))?;
    v.parse()
        .ok()
        .with_context(|| format!("bad {flag} value {v:?}"))
}

pub fn parse_args(args: &[String]) -> Result<SelfPlayArgs> {
    let mut cfg = SelfPlayConfig::default();
    let mut batch = BatchOptions::default();
    let mut infer_opts = InferOptions::default();
    let (mut model, mut out, mut games, mut seed) = (None, None, 1u32, None);
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => {
                let value = it.next().context("--model requires a path or uniform")?;
                model = (value != "uniform").then(|| PathBuf::from(value));
            }
            "--ep" | "--threads" | "--infer-batch" => {
                let value = it
                    .next()
                    .with_context(|| format!("{arg} requires a value"))?;
                infer_opts.parse_flag(arg, value)?;
            }
            "--out" => out = Some(it.next().context("--out requires a path")?.into()),
            "--games" => games = parse_num(&mut it, "--games")?,
            "--sims" => cfg.sims = parse_num(&mut it, "--sims")?,
            "--seed" => seed = Some(parse_num(&mut it, "--seed")?),
            "--max-plies" => cfg.max_plies = parse_num(&mut it, "--max-plies")?,
            "--temp-plies" => cfg.temp_plies = parse_num(&mut it, "--temp-plies")?,
            "--parallel" => batch.parallel = parse_num(&mut it, "--parallel")?,
            "--leaf-batch" => batch.leaf_batch = parse_num(&mut it, "--leaf-batch")?,
            "--resign-threshold" => cfg.resign_threshold = parse_num(&mut it, arg)?,
            "--resign-consecutive" => cfg.resign_consecutive = parse_num(&mut it, arg)?,
            "--resign-playthrough" => cfg.resign_playthrough = parse_num(&mut it, arg)?,
            "--no-reuse" => cfg.reuse = false,
            other => bail!("unknown argument for selfplay: {other}"),
        }
    }
    batch.validate(&cfg)?;
    let seed = seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
    });
    Ok(SelfPlayArgs {
        model,
        infer_opts,
        out: out.context("selfplay requires --out <bin>")?,
        games,
        seed,
        cfg,
        batch,
    })
}

/// Play `games` games into `out`; returns the number of samples written.
pub fn generate(
    eval: &mut dyn Evaluator,
    games: u32,
    seed: u64,
    cfg: &SelfPlayConfig,
    out: &Path,
) -> Result<u64> {
    generate_with_model(
        eval,
        games,
        seed,
        cfg,
        out,
        "uniform",
        BatchOptions::default(),
        None,
    )
}

fn generate_with_model(
    eval: &mut dyn Evaluator,
    games: u32,
    seed: u64,
    cfg: &SelfPlayConfig,
    out: &Path,
    model: &str,
    batch: BatchOptions,
    progress: Option<&ProgressReporter>,
) -> Result<u64> {
    batch.validate(cfg)?;
    let mut writer = ChunkWriter::create(out)?;
    let mut stats = SelfPlayStats::default();
    let start = Instant::now();
    run_games_with_progress(eval, games, seed, cfg, batch, |i, game| {
        for s in &game_samples(&game) {
            writer.write(s)?;
        }
        writer.flush()?;
        stats.record(&game);
        eprintln!(
            "selfplay: game {}/{games}: {} plies, {:?} ({:?}), {:.1}s elapsed, {} samples total",
            i + 1,
            game.plies.len(),
            game.result,
            game.termination,
            start.elapsed().as_secs_f32(),
            writer.count()
        );
        Ok(())
    }, |mut snapshot| {
        if let Some(reporter) = progress {
            if snapshot.phase == ProgressPhase::Complete { snapshot.phase = ProgressPhase::Writing; }
            reporter.update(snapshot);
        }
    })?;
    let count = writer.finish(stats.sidecar(model, seed, cfg))?;
    if let Some(reporter) = progress { reporter.set_phase(ProgressPhase::Complete); }
    Ok(count)
}

pub fn run(args: &[String]) -> Result<()> {
    let a = parse_args(args)?;
    let progress = ProgressReporter::stderr(a.games, Duration::from_secs(5));
    let mut eval = load_evaluator(a.model.as_deref(), &a.infer_opts).map_err(|error| {
        progress.set_phase(ProgressPhase::Failed);
        error
    })?;
    eprintln!(
        "selfplay: model {}, {} games, {} sims, seed {}",
        a.model
            .as_ref()
            .map_or_else(|| "uniform".to_owned(), |p| p.display().to_string()),
        a.games,
        a.cfg.sims,
        a.seed
    );
    let model = a
        .model
        .as_ref()
        .map_or_else(|| "uniform".to_owned(), |p| p.display().to_string());
    let n = generate_with_model(
        eval.as_mut(),
        a.games,
        a.seed,
        &a.cfg,
        &a.out,
        &model,
        a.batch,
        Some(&progress),
    ).map_err(|error| {
        progress.set_phase(ProgressPhase::Failed);
        error
    })?;
    println!("wrote {n} samples to {}", a.out.display());
    Ok(())
}
