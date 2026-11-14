//! `engine` binary entry point. No arguments (or `uci`) starts the UCI loop.

use std::io::{stdin, stdout};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use engine::nn::infer_opts::InferOptions;
use engine::nn::ort_infer::load_evaluator;
use engine::uci::uci_loop;

const USAGE: &str = "usage: engine [command] [args]

commands:
  uci [--model <path.onnx>|uniform] [--ep auto|cpu|coreml] [--threads T] [--infer-batch N]
                                     UCI loop (default; uniform net without --model)
  dump-test-vectors [--out <path>]   write contract test vectors
  bench [--model <onnx>|uniform] [--sims 800] [--batch 8] [--positions 6]
        [--ep auto|cpu|coreml] [--threads T] [--infer-batch N]
                                     fixed-position search throughput (JSON)
  infer-bench --model <onnx> [--batches 32,31,32,20] [--cycles 5] [--repeat 2]
        [--ep auto|cpu|coreml] [--threads T] [--memory-pattern on|off] [--pad-to N]
        [--verify-fixed N]
                                     direct model-call timing (JSONL)
  match --a <PLAYER> --b <PLAYER> --games <N even> --sims <S> --seed <X> --out <json>
        [--openings <file>] [--opening-offset 0] [--max-plies 300] [--batch 8]
        [--moves-out <txt>] [--ep auto|cpu|coreml] [--threads T] [--infer-batch N]
                                     paired evaluation (onnx, uniform, material, random)
  convert-pgn --input <pgn>|- (--out <bin> | --out-dir <dir>)
              [--chunk-samples 262144] [--max-games N] [--max-samples M] [--skip-games K]
                                     PGN games -> supervised training samples
  selfplay [--model <onnx>|uniform] --games <N> --sims 200 --out <bin>
           [--seed S] [--max-plies 512] [--temp-plies 30] [--ep auto|cpu|coreml] [--threads T]
           [--no-reuse] [--infer-batch N]
                                     self-play with root Dirichlet noise";

fn run_uci(args: &[String]) -> Result<()> {
    let mut model: Option<PathBuf> = None;
    let mut opts = InferOptions::default();
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => {
                let value = it.next().context("--model requires a path or uniform")?;
                model = (value != "uniform").then(|| PathBuf::from(value));
            }
            "--ep" | "--threads" | "--infer-batch" => {
                let value = it.next().with_context(|| format!("{arg} requires a value"))?;
                opts.parse_flag(arg, value)?;
            }
            other => bail!("unknown argument for uci: {other}\n\n{USAGE}"),
        }
    }
    let evaluator = load_evaluator(model.as_deref(), &opts)?;
    match &model {
        Some(p) => eprintln!("chessbot: loaded model {}", p.display()),
        None => eprintln!("chessbot: using uniform dummy network"),
    }
    uci_loop::run(evaluator, stdin().lock(), stdout().lock())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => run_uci(&[]),
        Some("uci") => run_uci(&args[1..]),
        Some("dump-test-vectors") => engine::cli::dump_vectors::run(&args[1..]),
        Some("bench") => engine::cli::bench::run(&args[1..]),
        Some("infer-bench") => engine::cli::infer_bench::run(&args[1..]),
        Some("match") => engine::cli::match_cmd::run(&args[1..]),
        Some("convert-pgn") => engine::cli::convert_pgn::run(&args[1..]),
        Some("selfplay") => engine::cli::selfplay::run(&args[1..]),
        Some("-h" | "--help") => {
            println!("{USAGE}");
            Ok(())
        }
        Some(other) => bail!("unknown command: {other}\n\n{USAGE}"),
    }
}
