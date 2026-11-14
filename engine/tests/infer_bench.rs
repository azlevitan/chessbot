//! Actual Rust/ORT diagnostic order, accounting, bounds and untouched search bench.

use std::collections::HashSet;
use std::process::{Command, Output};

use serde_json::Value;

const MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/match_uniform.onnx");
const FORMAT: &str = "chessbot-infer-bench-v1";

fn command(extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_engine"))
        .args(["infer-bench", "--model", MODEL, "--ep", "cpu", "--threads", "1"])
        .args(extra)
        .output()
        .unwrap()
}

fn records(extra: &[&str]) -> Vec<Value> {
    let output = command(extra);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let text = String::from_utf8(output.stdout).unwrap();
    let rows: Vec<Value> = text.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
    assert!(rows.iter().all(|row| row["format"] == FORMAT));
    assert_eq!(rows.first().unwrap()["kind"], "session");
    assert_eq!(rows.last().unwrap()["kind"], "summary");
    assert!(rows[1..rows.len() - 1].iter().all(|row| row["kind"] == "call"));
    rows
}

fn finite_seconds(row: &Value, key: &str) -> f64 {
    let value = row[key].as_f64().unwrap();
    assert!(value.is_finite() && value >= 0.0, "invalid {key}: {row}");
    value
}

#[test]
fn one_session_preserves_requested_sequence_repeats_and_first_use_across_cycles() {
    let rows = records(&["--batches", "32,31,32,20", "--cycles", "2", "--repeat", "2"]);
    assert_eq!(rows.len(), 18);
    let session = &rows[0];
    assert_eq!(session["model"], std::fs::canonicalize(MODEL).unwrap().to_str().unwrap());
    assert_eq!(session["requested_ep"], "cpu");
    assert_eq!(session["selected_ep"], "cpu");
    assert_eq!(session["threads"], 1);
    assert_eq!(session["memory_pattern"], true);
    assert!(session["pad_to"].is_null());
    assert_eq!(session["batches"], serde_json::json!([32, 31, 32, 20]));
    assert_eq!((session["cycles"].as_u64(), session["repeat"].as_u64()), (Some(2), Some(2)));
    assert_eq!(session["os"], std::env::consts::OS);
    assert_eq!(session["arch"], std::env::consts::ARCH);
    assert!(session["ort_build"].as_str().is_some_and(|value| !value.is_empty()));
    finite_seconds(session, "load_seconds");

    let mut seen = HashSet::new();
    let mut previous = None;
    let mut seconds = 0.0;
    let sequence = [32_u64, 32, 31, 31, 32, 32, 20, 20];
    for (index, row) in rows[1..17].iter().enumerate() {
        let requested = sequence[index % sequence.len()];
        assert_eq!(row["index"], index + 1);
        assert_eq!(row["cycle"], index / sequence.len() + 1);
        assert_eq!(row["repeat_index"], index % 2 + 1);
        assert_eq!(row["requested_batch"], requested);
        assert_eq!(row["physical_batch"], requested);
        assert_eq!(row["shape_changed"], previous.is_some_and(|value| value != requested));
        assert_eq!(row["first_for_shape"], seen.insert(requested));
        previous = Some(requested);
        seconds += finite_seconds(row, "seconds");
    }
    let summary = rows.last().unwrap();
    assert_eq!(summary["calls"], 16);
    assert_eq!(summary["requested_positions"], 460);
    assert_eq!(summary["physical_positions"], 460);
    assert!((finite_seconds(summary, "inference_seconds") - seconds).abs() <= 1e-9);
}

#[test]
fn diagnostic_padding_keeps_requested_counts_separate_from_physical_work() {
    let rows = records(&["--batches", "31,20", "--cycles", "1", "--repeat", "2",
                         "--pad-to", "32", "--memory-pattern", "off"]);
    assert_eq!(rows.len(), 6);
    assert_eq!(rows[0]["memory_pattern"], false);
    assert_eq!(rows[0]["pad_to"], 32);
    for (index, row) in rows[1..5].iter().enumerate() {
        assert_eq!(row["requested_batch"], if index < 2 { 31 } else { 20 });
        assert_eq!(row["physical_batch"], 32);
        assert_eq!(row["first_for_shape"], index == 0);
        assert_eq!(row["shape_changed"], false);
        finite_seconds(row, "seconds");
    }
    let summary = rows.last().unwrap();
    assert_eq!(summary["calls"], 4);
    assert_eq!(summary["requested_positions"], 102);
    assert_eq!(summary["physical_positions"], 128);
}

#[test]
fn fixed_shape_control_and_explicit_memory_pattern_on_are_reported() {
    let rows = records(&["--batches", "32", "--cycles", "2", "--repeat", "1",
                         "--memory-pattern", "on"]);
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0]["memory_pattern"], true);
    assert_eq!(rows[1]["first_for_shape"], true);
    assert_eq!(rows[2]["first_for_shape"], false);
    assert_eq!(rows[1]["shape_changed"], false);
    assert_eq!(rows[2]["shape_changed"], false);
    assert_eq!(rows[3]["calls"], 2);
    assert_eq!(rows[3]["requested_positions"], 64);
}

#[test]
fn invalid_arguments_fail_before_session_output_or_inference() {
    for flags in [
        vec!["--batches", "0"], vec!["--batches", "65"],
        vec!["--batches", ""], vec!["--batches", "32,,20"],
        vec!["--batches", "32", "--pad-to", "31"], vec!["--pad-to", "65"],
        vec!["--memory-pattern", "maybe"], vec!["--repeat", "0"],
        vec!["--cycles", "0"], vec!["--threads", "0"],
        vec!["--batches", "32", "--cycles", "10001", "--repeat", "1"],
        vec!["--batches", "32", "--cycles", "18446744073709551615", "--repeat", "2"],
        vec!["--unknown", "1"], vec!["--batches"], vec!["--ep", "gpu"],
    ] {
        let output = command(&flags);
        assert!(!output.status.success(), "accepted invalid {flags:?}");
        assert!(output.stdout.is_empty(), "started work for invalid {flags:?}");
        assert!(!output.stderr.is_empty());
    }
    let missing_model = Command::new(env!("CARGO_BIN_EXE_engine"))
        .args(["infer-bench", "--cycles", "1", "--repeat", "1"])
        .output().unwrap();
    assert!(!missing_model.status.success());
    assert!(missing_model.stdout.is_empty());
}

#[test]
fn existing_search_benchmark_keeps_its_single_line_seven_field_json_contract() {
    let output = Command::new(env!("CARGO_BIN_EXE_engine"))
        .args(["bench", "--model", "uniform", "--sims", "8", "--positions", "1"])
        .output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 1);
    let value: Value = serde_json::from_str(stdout.trim()).unwrap();
    let mut keys: Vec<_> = value.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, ["avg_batch", "ep", "evals", "evals_per_sec", "seconds", "sims", "sims_per_sec"]);
}
