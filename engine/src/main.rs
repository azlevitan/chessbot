//! `engine` binary entry point. No arguments (or `uci`) starts the UCI loop.

use std::io::{stdin, stdout};
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use engine::nn::ort_infer::load_evaluator;
use engine::uci::uci_loop;

const USAGE: &str = "usage: engine [command] [args]

commands:
  uci [--model <path.onnx>]          UCI loop (default; uniform dummy net without --model)
  dump-test-vectors [--out <path>]   write contract test vectors
  convert-pgn --input <pgn> --out <bin> [--max-games N]
                                     PGN games -> supervised training samples
  selfplay --model <onnx> --games <N> --sims 200 --out <bin>
           [--seed S] [--max-plies 512] [--temp-plies 30]
                                     self-play with root Dirichlet noise";

fn run_uci(args: &[String]) -> Result<()> {
    let mut model: Option<PathBuf> = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--model" => model = Some(it.next().context("--model requires a path")?.into()),
            other => bail!("unknown argument for uci: {other}\n\n{USAGE}"),
        }
    }
    let evaluator = load_evaluator(model.as_deref())?;
    match &model {
        Some(p) => eprintln!("chessbot: loaded model {}", p.display()),
        None => eprintln!("chessbot: no --model given, using uniform dummy network"),
    }
    uci_loop::run(evaluator, stdin().lock(), stdout().lock())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        None => run_uci(&[]),
        Some("uci") => run_uci(&args[1..]),
        Some("dump-test-vectors") => engine::cli::dump_vectors::run(&args[1..]),
        Some("convert-pgn") => engine::cli::convert_pgn::run(&args[1..]),
        Some("selfplay") => engine::cli::selfplay::run(&args[1..]),
        Some("-h" | "--help") => {
            println!("{USAGE}");
            Ok(())
        }
        Some(other) => bail!("unknown command: {other}\n\n{USAGE}"),
    }
}
