//! Fixed-corpus search throughput benchmark (SPEC §8).
//!
//! Timing includes fresh-tree searches and inference, but excludes model loading,
//! FEN parsing and JSON output. `ep` names the selected provider; ONNX Runtime
//! can still fall back to CPU when CoreML cannot run an operation.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use serde::Serialize;

use crate::board::from_fen;
use crate::mcts::arena::MAX_NODES;
use crate::mcts::{search, SearchLimits};
use crate::nn::infer_opts::{Ep, InferOptions};
use crate::nn::ort_infer::load_evaluator;
use crate::nn::{Evaluator, NetOutput};

const USAGE: &str = "usage: engine bench [--model <onnx>|uniform] [--sims 800]
    [--batch 8] [--positions 6] [--ep auto|cpu|coreml] [--threads 1] [--infer-batch N]
Search the first 1..6 fixed positions, with a fresh tree for each.
Print one JSON line with completed simulations, evaluations and throughput.";

/// Stable benchmark order. The four middlegames cover closed/open centres,
/// opposite-side castling and tactical material imbalance, with both sides to move.
pub const BENCH_FENS: [(&str, &str); 6] = [
    (
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ),
    // Ruy Lopez, after 10...Nbd7.
    (
        "ruy-lopez",
        "r1bq1rk1/2pnbppp/p2p1n2/1p2p3/3PP3/1BP2N1P/PP3PP1/RNBQR1K1 w - - 1 11",
    ),
    // Queen's Gambit Declined, after 10.Nxd5.
    (
        "queens-gambit",
        "rnb2rk1/p1p1qpp1/1p2p2p/3N4/3P4/4PN2/PP3PPP/R2QKB1R b KQ - 0 10",
    ),
    // Sicilian Najdorf, after 10...Nbd7.
    (
        "sicilian",
        "r2q1rk1/1p1nbppp/p2pbn2/4p3/4P3/1NN1BP2/PPPQ2PP/2KR1B1R w - - 5 11",
    ),
    // French Winawer poisoned pawn, after 10.Ne2.
    (
        "french",
        "rnb1k1r1/ppq1np1Q/4p3/3pP3/3p4/P1P5/2P1NPPP/R1B1KB1R b KQq - 1 10",
    ),
];

struct BenchArgs {
    model: Option<PathBuf>,
    sims: u32,
    batch: usize,
    positions: usize,
    infer: InferOptions,
}

fn number<T: std::str::FromStr>(flag: &str, value: &str) -> Result<T> {
    value
        .parse()
        .ok()
        .with_context(|| format!("bad {flag} value {value:?}"))
}

fn parse_args(args: &[String]) -> Result<BenchArgs> {
    let mut a = BenchArgs {
        model: None,
        sims: 800,
        batch: 8,
        positions: BENCH_FENS.len(),
        infer: InferOptions::default(),
    };
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        ensure!(
            matches!(
                flag.as_str(),
                "--model" | "--sims" | "--batch" | "--positions" | "--ep" | "--threads" | "--infer-batch"
            ),
            "unknown argument for bench: {flag}"
        );
        let value = it
            .next()
            .with_context(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--model" => a.model = (value != "uniform").then(|| PathBuf::from(value)),
            "--sims" => a.sims = number(flag, value)?,
            "--batch" => a.batch = number(flag, value)?,
            "--positions" => a.positions = number(flag, value)?,
            "--ep" | "--threads" | "--infer-batch" => {
                a.infer.parse_flag(flag, value)?;
            }
            _ => unreachable!(),
        }
    }
    ensure!(a.sims > 0, "--sims must be positive");
    ensure!(a.batch > 0, "--batch must be positive");
    ensure!(
        (1..=BENCH_FENS.len()).contains(&a.positions),
        "--positions must be in 1..=6"
    );
    a.infer.validate()?;
    Ok(a)
}

/// Count actual evaluator inputs, rather than assuming every simulation evaluates.
struct CountingEvaluator<'a> {
    inner: &'a mut dyn Evaluator,
    evals: u64,
    calls: u64,
}

impl Evaluator for CountingEvaluator<'_> {
    fn evaluate(&mut self, planes: &[f32], batch: usize) -> Result<Vec<NetOutput>> {
        let outputs = self.inner.evaluate(planes, batch)?;
        self.evals += batch as u64;
        self.calls += 1;
        Ok(outputs)
    }
}

#[derive(Serialize)]
struct BenchResult {
    sims: u64,
    evals: u64,
    seconds: f64,
    sims_per_sec: f64,
    evals_per_sec: f64,
    avg_batch: f64,
    ep: &'static str,
}

fn provider(a: &BenchArgs) -> &'static str {
    if a.model.is_none() {
        return "uniform";
    }
    match a.infer.ep {
        Ep::Cpu => "cpu",
        Ep::CoreMl => "coreml",
        Ep::Auto if cfg!(target_os = "macos") => "coreml",
        Ep::Auto => "cpu",
    }
}

pub fn run(args: &[String]) -> Result<()> {
    if args.len() == 1 && matches!(args[0].as_str(), "--help" | "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let a = parse_args(args)?;
    let mut inner = load_evaluator(a.model.as_deref(), &a.infer)?;
    let boards = BENCH_FENS[..a.positions]
        .iter()
        .map(|(name, fen)| from_fen(fen).with_context(|| format!("bench position {name}")))
        .collect::<Result<Vec<_>>>()?;
    let mut eval = CountingEvaluator {
        inner: inner.as_mut(),
        evals: 0,
        calls: 0,
    };
    let limits = SearchLimits {
        nodes: Some(a.sims),
        movetime: None,
        batch: a.batch,
    };
    let mut sims = 0u64;
    let start = Instant::now();
    for board in &boards {
        let result = search(board, &mut eval, limits, MAX_NODES)?;
        ensure!(
            result.simulations == a.sims,
            "bench search did not complete its simulation budget"
        );
        sims += u64::from(result.simulations);
    }
    let seconds = start.elapsed().as_secs_f64();
    if seconds <= 0.0 || eval.calls == 0 {
        bail!("bench completed without measurable search work");
    }
    let result = BenchResult {
        sims,
        evals: eval.evals,
        seconds,
        sims_per_sec: sims as f64 / seconds,
        evals_per_sec: eval.evals as f64 / seconds,
        avg_batch: eval.evals as f64 / eval.calls as f64,
        ep: provider(&a),
    };
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
