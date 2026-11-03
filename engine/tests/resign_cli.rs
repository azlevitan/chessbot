//! Resignation flags, validation before output creation, and disabled-mode chunks.

use std::path::PathBuf;
use std::process::Command;

use engine::cli::selfplay::{generate, parse_args};
use engine::encoder::sample::SAMPLE_BYTES;
use engine::nn::ort_infer::UniformEvaluator;
use engine::selfplay::SelfPlayConfig;
use serde_json::{json, Value};

struct TestDir(PathBuf);
impl TestDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("chessbot-resign-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for TestDir {
    fn drop(&mut self) { std::fs::remove_dir_all(&self.0).unwrap(); }
}

#[test]
fn cli_uses_the_spec_defaults_and_accepts_disabled_thresholds_and_probability_endpoints() {
    let base = vec!["--out".to_string(), "unused.bin".to_string()];
    let cfg = parse_args(&base).unwrap().cfg;
    assert_eq!((cfg.resign_threshold, cfg.resign_consecutive, cfg.resign_playthrough), (-0.95, 3, 0.1));
    for probability in ["0", "1"] {
        let mut args = base.clone();
        args.extend(["--resign-threshold", "-1.5", "--resign-consecutive", "5",
            "--resign-playthrough", probability].map(str::to_string));
        let cfg = parse_args(&args).unwrap().cfg;
        assert_eq!(cfg.resign_threshold, -1.5);
        assert_eq!(cfg.resign_consecutive, 5);
        assert_eq!(cfg.resign_playthrough, probability.parse::<f32>().unwrap());
    }
    for flag in ["--resign-threshold", "--resign-consecutive", "--resign-playthrough"] {
        let mut args = base.clone();
        args.push(flag.to_string());
        assert!(parse_args(&args).is_err());
    }
}

#[test]
fn invalid_flags_preserve_existing_binary_and_sidecar_without_temporary_files() {
    let dir = TestDir::new("invalid");
    let out = dir.0.join("keep.bin");
    std::fs::write(&out, b"keep binary").unwrap();
    std::fs::write(out.with_extension("json"), b"keep sidecar").unwrap();
    for (flag, value) in [
        ("--resign-threshold", "NaN"), ("--resign-threshold", "inf"),
        ("--resign-threshold", "-inf"), ("--resign-threshold", "bad"),
        ("--resign-consecutive", "0"), ("--resign-consecutive", "-1"),
        ("--resign-consecutive", "bad"), ("--resign-playthrough", "NaN"),
        ("--resign-playthrough", "inf"), ("--resign-playthrough", "-0.1"),
        ("--resign-playthrough", "1.1"), ("--resign-playthrough", "bad"),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_engine"))
            .args(["selfplay", "--out"]).arg(&out).args([flag, value]).output().unwrap();
        assert!(!output.status.success(), "accepted {flag} {value}");
        assert_eq!(std::fs::read(&out).unwrap(), b"keep binary");
        assert_eq!(std::fs::read(out.with_extension("json")).unwrap(), b"keep sidecar");
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 2);
    }
    for cfg in [
        SelfPlayConfig { resign_threshold: f32::NAN, ..SelfPlayConfig::default() },
        SelfPlayConfig { resign_consecutive: 0, ..SelfPlayConfig::default() },
        SelfPlayConfig { resign_playthrough: f32::INFINITY, ..SelfPlayConfig::default() },
    ] {
        assert!(generate(&mut UniformEvaluator, 0, 0, &cfg, &out).is_err());
        assert_eq!(std::fs::read(&out).unwrap(), b"keep binary");
        assert_eq!(std::fs::read(out.with_extension("json")).unwrap(), b"keep sidecar");
    }
}

#[test]
fn disabled_uniform_selfplay_publishes_configured_audit_metadata_and_valid_chunks() {
    let dir = TestDir::new("disabled");
    let mut reference = None;
    for parallel in [1, 4] {
        let out = dir.0.join(format!("p{parallel}.bin"));
        let output = Command::new(env!("CARGO_BIN_EXE_engine"))
            .args(["selfplay", "--model", "uniform", "--ep", "cpu", "--games", "3",
                "--sims", "8", "--max-plies", "20", "--seed", "7", "--leaf-batch", "4",
                "--resign-threshold", "-1.0", "--resign-consecutive", "5",
                "--resign-playthrough", "1", "--parallel", &parallel.to_string(), "--out"])
            .arg(&out).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let bytes = std::fs::read(&out).unwrap();
        let meta: Value = serde_json::from_slice(&std::fs::read(out.with_extension("json")).unwrap()).unwrap();
        assert_eq!(meta["format"], "chessbot-chunk-v1");
        assert_eq!(meta["samples"].as_u64().unwrap() as usize * SAMPLE_BYTES, bytes.len());
        assert_eq!(meta["games"], 3);
        assert_eq!(meta["terminations"]["resign"], 0);
        assert_eq!(meta["resign"], json!({"threshold": -1.0, "consecutive": 5,
            "playthrough_games": 3, "false_positives": 0}));
        for key in ["results", "terminations"] {
            assert_eq!(meta[key].as_object().unwrap().values().map(|v| v.as_u64().unwrap()).sum::<u64>(), 3);
        }
        let actual = (bytes, meta);
        if let Some(expected) = &reference { assert_eq!(&actual, expected); }
        else { reference = Some(actual); }
    }
    assert!(std::fs::read_dir(&dir.0).unwrap().all(|p| p.unwrap().path().extension().unwrap() != "tmp"));
}
