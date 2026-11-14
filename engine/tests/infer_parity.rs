//! Public fixed-batch parity diagnostics keep failures machine-readable.

use std::process::Command;

use serde_json::Value;

const MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/match_uniform.onnx");

#[test]
fn fixed_parity_reports_every_requested_shape_and_chunk_boundary() {
    let output = Command::new(env!("CARGO_BIN_EXE_engine"))
        .args(["infer-bench", "--model", MODEL, "--ep", "cpu", "--verify-fixed", "32"])
        .output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let records: Vec<Value> = String::from_utf8(output.stdout).unwrap().lines()
        .map(|line| serde_json::from_str(line).unwrap()).collect();
    assert_eq!(records.len(), 9);
    assert!(records.iter().all(|row| row["format"] == "chessbot-infer-parity-v1"));
    assert_eq!(records[0]["kind"], "session");
    assert_eq!(records[0]["policy_probability_mask"], "none");
    for (row, batch) in records[1..8].iter().zip([1_u64, 7, 20, 31, 32, 33, 65]) {
        assert_eq!(row["kind"], "parity");
        assert_eq!(row["requested_batch"], batch);
        assert_eq!(row["physical_calls"], batch.div_ceil(32));
        assert_eq!(row["raw_close"], true);
        assert_eq!(row["probabilities_close"], true);
        assert_eq!(row["pass"], true);
    }
    assert_eq!(records[8]["kind"], "summary");
    assert_eq!(records[8]["batches_checked"], 7);
    assert_eq!(records[8]["pass"], true);
}

#[test]
fn invalid_fixed_widths_and_conflicting_padding_fail_before_inference() {
    for flags in [vec!["--verify-fixed", "0"], vec!["--verify-fixed", "65"],
                  vec!["--verify-fixed", "32", "--pad-to", "32"]] {
        let output = Command::new(env!("CARGO_BIN_EXE_engine"))
            .args(["infer-bench", "--model", MODEL, "--ep", "cpu"])
            .args(flags).output().unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
    }
}
