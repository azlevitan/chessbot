//! Phase 6.1: portable inference options and uniform CLI self-play.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

use engine::encoder::sample::SAMPLE_BYTES;
use engine::nn::infer_opts::{Ep, InferOptions};
use engine::nn::ort_infer::load_evaluator;

#[test]
fn defaults_load_a_uniform_evaluator() {
    let opts = InferOptions::default();
    assert_eq!(opts.ep, Ep::Auto);
    assert_eq!(opts.threads, 1);
    let mut eval = load_evaluator(None, &opts).unwrap();
    let out = eval.evaluate(&vec![0.0; 18 * 64], 1).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].policy_logits, vec![0.0; 4168]);
    assert_eq!(out[0].wdl_logits, [0.0; 3]);
}

#[test]
fn shared_flags_parse_and_reject_bad_values() {
    let mut opts = InferOptions::default();
    for (value, expected) in [("cpu", Ep::Cpu), ("coreml", Ep::CoreMl), ("auto", Ep::Auto)] {
        assert!(opts.parse_flag("--ep", value).unwrap());
        assert_eq!(opts.ep, expected);
    }
    assert!(opts.parse_flag("--threads", "2").unwrap());
    assert_eq!(opts.threads, 2);
    let before = opts.clone();
    assert!(!opts.parse_flag("--model", "net.onnx").unwrap());
    assert_eq!(opts, before);
    assert!(opts.parse_flag("--ep", "gpu").is_err());
    for value in ["0", "-1", "abc", "1.5", "", "999999999999999999999999999999"] {
        assert!(opts.parse_flag("--threads", value).is_err(), "accepted {value:?}");
    }
}

#[test]
fn loader_rejects_zero_threads_from_direct_callers() {
    let opts = InferOptions { threads: 0, ..InferOptions::default() };
    assert!(load_evaluator(None, &opts).is_err());
}

#[cfg(not(target_os = "macos"))]
#[test]
fn explicit_coreml_is_rejected_before_loading_a_model() {
    let opts = InferOptions { ep: Ep::CoreMl, threads: 1 };
    let err = load_evaluator(None, &opts).err().expect("CoreML must fail on this OS");
    assert!(err.to_string().contains("only supported on macOS"));
    for command in ["uci", "selfplay"] {
        let mut args = vec![command, "--ep", "coreml"];
        if command == "selfplay" {
            args.extend(["--out", "unused.bin"]);
        }
        let output = engine(&args);
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("only supported on macOS"));
    }
}

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("chessbot-infer-{}-{stamp}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn engine(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_engine"))
        .args(args)
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

#[test]
fn selfplay_without_a_model_and_explicit_uniform_write_identical_samples() {
    let dir = TempDir::new();
    let mut results = Vec::new();
    for (file, extra) in [
        ("default.bin", vec![]),
        ("uniform.bin", vec!["--model", "uniform", "--ep", "cpu", "--threads", "2"]),
    ] {
        let path = dir.0.join(file);
        let mut args = vec![
            "selfplay", "--games", "1", "--sims", "8", "--max-plies", "10",
            "--seed", "1", "--out", path.to_str().unwrap(),
        ];
        args.extend(extra);
        let output = engine(&args);
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let bytes = fs::read(path).unwrap();
        assert_eq!(bytes.len(), 10 * SAMPLE_BYTES);
        results.push(bytes);
    }
    assert_eq!(results[0], results[1]);
}

#[test]
fn uci_accepts_inference_flags_and_uniform_model() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_engine"))
        .args(["uci", "--model", "uniform", "--ep", "cpu", "--threads", "2"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"uci\nisready\nquit\n").unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.lines().any(|line| line == "uciok"));
    assert!(text.lines().any(|line| line == "readyok"));
}

#[test]
fn both_commands_reject_invalid_and_missing_inference_values() {
    for command in ["uci", "selfplay"] {
        for flags in [vec!["--ep", "gpu"], vec!["--threads", "0"], vec!["--threads", "bad"],
                      vec!["--ep"], vec!["--threads"]] {
            let mut args = vec![command];
            args.extend(flags);
            let output = engine(&args);
            assert!(!output.status.success(), "accepted {args:?}");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(stderr.contains("--ep") || stderr.contains("--threads"), "{stderr}");
        }
    }
}
