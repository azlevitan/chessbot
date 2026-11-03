//! Benchmark CLI contract: real counts, partial batches, corpus and validation.

use std::process::{Command, Output};

use anyhow::Result;
use cozy_chess::GameStatus;
use engine::board::from_fen;
use engine::cli::bench::BENCH_FENS;
use engine::mcts::arena::MAX_NODES;
use engine::mcts::{search, SearchLimits};
use engine::nn::ort_infer::UniformEvaluator;
use engine::nn::{Evaluator, NetOutput};
use serde_json::Value;

fn command(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_engine"))
        .arg("bench")
        .args(args)
        .output()
        .unwrap()
}

fn bench(args: &[&str]) -> Value {
    let out = command(args);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    assert_eq!(text.lines().count(), 1, "{text}");
    let value: Value = serde_json::from_str(&text).unwrap();
    let mut keys = value
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "avg_batch",
            "ep",
            "evals",
            "evals_per_sec",
            "seconds",
            "sims",
            "sims_per_sec"
        ]
    );
    for field in ["seconds", "sims_per_sec", "evals_per_sec", "avg_batch"] {
        let number = value[field].as_f64().unwrap();
        assert!(number.is_finite() && number > 0.0, "{field}: {number}");
    }
    for (rate, count) in [("sims_per_sec", "sims"), ("evals_per_sec", "evals")] {
        let reconstructed = value[rate].as_f64().unwrap() * value["seconds"].as_f64().unwrap();
        assert!((reconstructed - value[count].as_f64().unwrap()).abs() < 1e-8);
    }
    assert_eq!(value["ep"], "uniform");
    value
}

#[test]
fn all_six_fixed_positions_are_distinct_valid_and_nonterminal() {
    assert_eq!(BENCH_FENS.len(), 6);
    assert_eq!(BENCH_FENS[0].0, "startpos");
    assert_eq!(BENCH_FENS[1].0, "kiwipete");
    let mut hashes = std::collections::HashSet::new();
    let mut black_to_move = 0;
    for (name, fen) in BENCH_FENS {
        let board = from_fen(fen).unwrap();
        assert_eq!(board.status(), GameStatus::Ongoing, "{name}");
        assert!(hashes.insert(board.hash()), "duplicate {name}");
        black_to_move += usize::from(board.side_to_move() == cozy_chess::Color::Black);
    }
    assert_eq!(black_to_move, 2);
}

#[test]
fn default_run_and_explicit_uniform_use_the_same_search_configuration() {
    let default = bench(&[]);
    let explicit = bench(&[
        "--model",
        "uniform",
        "--sims",
        "800",
        "--batch",
        "8",
        "--positions",
        "6",
        "--ep",
        "cpu",
        "--threads",
        "2",
    ]);
    assert_eq!(default["sims"], 4800);
    assert!(default["avg_batch"].as_f64().unwrap() >= 6.0);
    for field in ["sims", "evals", "avg_batch", "ep"] {
        assert_eq!(default[field], explicit[field], "{field}");
    }
}

#[derive(Default)]
struct RecordingEvaluator {
    batches: Vec<usize>,
}

impl Evaluator for RecordingEvaluator {
    fn evaluate(&mut self, planes: &[f32], batch: usize) -> Result<Vec<NetOutput>> {
        self.batches.push(batch);
        UniformEvaluator.evaluate(planes, batch)
    }
}

#[test]
fn json_counts_match_direct_search_including_small_budgets_and_partial_batches() {
    for (sims, batch, positions) in [(1, 8, 6), (2, 32, 1), (19, 8, 3), (53, 1, 6), (800, 8, 6)] {
        let mut eval = RecordingEvaluator::default();
        let mut completed = 0u64;
        for (_, fen) in &BENCH_FENS[..positions] {
            let limits = SearchLimits {
                nodes: Some(sims),
                batch,
                ..SearchLimits::default()
            };
            let result = search(&from_fen(fen).unwrap(), &mut eval, limits, MAX_NODES).unwrap();
            assert!(result.nodes_used <= MAX_NODES);
            completed += u64::from(result.simulations);
        }
        let measured = bench(&[
            "--sims",
            &sims.to_string(),
            "--batch",
            &batch.to_string(),
            "--positions",
            &positions.to_string(),
        ]);
        let evals: usize = eval.batches.iter().sum();
        assert_eq!(measured["sims"], completed);
        assert_eq!(completed, u64::from(sims) * positions as u64);
        assert_eq!(measured["evals"], evals);
        let average = evals as f64 / eval.batches.len() as f64;
        assert!((measured["avg_batch"].as_f64().unwrap() - average).abs() < 1e-12);
        assert!(eval.batches.iter().all(|&size| size > 0 && size <= batch));
    }
}

#[test]
fn invalid_arguments_fail_without_emitting_benchmark_json() {
    for args in [
        vec!["--sims", "0"],
        vec!["--sims", "-1"],
        vec!["--sims", "4294967296"],
        vec!["--batch", "0"],
        vec!["--batch", "-1"],
        vec!["--batch", "bad"],
        vec!["--positions", "0"],
        vec!["--positions", "7"],
        vec!["--positions", "-1"],
        vec!["--threads", "0"],
        vec!["--ep", "gpu"],
        vec!["--unknown"],
        vec![
            "--model",
            "/definitely-missing-chessbot-model.onnx",
            "--ep",
            "cpu",
        ],
    ] {
        let out = command(&args);
        assert!(!out.status.success(), "accepted {args:?}");
        assert!(out.stdout.is_empty(), "failure emitted JSON: {args:?}");
        assert!(!out.stderr.is_empty());
    }
    for flag in [
        "--model",
        "--sims",
        "--batch",
        "--positions",
        "--ep",
        "--threads",
    ] {
        let out = command(&[flag]);
        assert!(!out.status.success(), "accepted missing {flag}");
        assert!(out.stdout.is_empty());
        assert!(String::from_utf8_lossy(&out.stderr).contains(flag));
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn explicit_coreml_is_rejected_on_linux_even_for_uniform() {
    let out = command(&["--ep", "coreml"]);
    assert!(!out.status.success());
    assert!(out.stdout.is_empty());
    assert!(String::from_utf8_lossy(&out.stderr).contains("only supported on macOS"));
}

#[test]
fn command_and_top_level_help_advertise_benchmark_options() {
    let help = command(&["--help"]);
    assert!(help.status.success());
    let help = String::from_utf8(help.stdout).unwrap();
    assert!(help.contains("--positions 6") && help.contains("--batch 8"));
    let main_help = Command::new(env!("CARGO_BIN_EXE_engine"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(main_help.status.success());
    assert!(String::from_utf8(main_help.stdout)
        .unwrap()
        .contains("bench [--model"));
}
