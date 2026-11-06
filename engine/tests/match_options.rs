//! Match CLI validation happens before existing outputs are changed.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

struct TestDir(PathBuf);
impl TestDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("chessbot-match-options-{name}-{}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("keep.json"), b"keep json").unwrap();
        fs::write(path.join("keep.txt"), b"keep log").unwrap();
        Self(path)
    }

    fn args(&self) -> Vec<String> {
        let mut args: Vec<_> = ["--a", "uniform", "--b", "uniform", "--games", "2", "--sims", "2",
            "--seed", "0", "--max-plies", "8", "--ep", "cpu", "--out"].map(str::to_string).into();
        args.push(self.0.join("keep.json").to_str().unwrap().to_string());
        args.push("--moves-out".to_string());
        args.push(self.0.join("keep.txt").to_str().unwrap().to_string());
        args
    }

    fn reject(&self, args: &[String]) {
        let before = fs::read_dir(&self.0).unwrap().count();
        let output = command(args);
        assert!(!output.status.success(), "accepted {args:?}");
        assert!(!output.stderr.is_empty(), "missing error for {args:?}");
        assert_eq!(fs::read(self.0.join("keep.json")).unwrap(), b"keep json");
        assert_eq!(fs::read(self.0.join("keep.txt")).unwrap(), b"keep log");
        assert_eq!(fs::read_dir(&self.0).unwrap().count(), before);
    }
}
impl Drop for TestDir {
    fn drop(&mut self) { fs::remove_dir_all(&self.0).unwrap(); }
}

fn command(args: &[String]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_engine")).arg("match").args(args).output().unwrap()
}

fn set(args: &mut Vec<String>, flag: &str, value: &str) {
    if let Some(index) = args.iter().position(|a| a == flag) { args[index + 1] = value.to_string(); }
    else { args.extend([flag.to_string(), value.to_string()]); }
}

#[test]
fn invalid_limits_options_and_models_preserve_existing_outputs() {
    let dir = TestDir::new("invalid");
    for (flag, value) in [
        ("--games", "0"), ("--games", "1"), ("--games", "3"), ("--games", "-2"),
        ("--games", "4294967296"), ("--sims", "0"), ("--sims", "1"),
        ("--sims", "bad"), ("--batch", "0"), ("--batch", "-1"),
        ("--max-plies", "0"), ("--max-plies", "7"), ("--max-plies", "-1"),
        ("--opening-offset", "-1"), ("--seed", "18446744073709551615"),
        ("--seed", "-1"), ("--threads", "0"), ("--ep", "gpu"),
        ("--a", ""), ("--b", "/definitely-missing-chessbot-match-model.onnx"),
        ("--a", "/definitely-missing-chessbot-match-model.onnx"),
        ("--openings", "/definitely-missing-chessbot-openings.txt"),
    ] {
        let mut args = dir.args();
        set(&mut args, flag, value);
        dir.reject(&args);
    }
    let mut same_path = dir.args();
    set(&mut same_path, "--moves-out", dir.0.join("keep.json").to_str().unwrap());
    dir.reject(&same_path);
    let mut unknown = dir.args();
    unknown.push("--unknown".to_string());
    dir.reject(&unknown);
}

#[test]
fn required_flags_and_missing_values_are_rejected_before_output_writes() {
    let dir = TestDir::new("missing");
    for required in ["--a", "--b", "--games", "--sims", "--seed", "--out"] {
        let mut args = dir.args();
        let index = args.iter().position(|a| a == required).unwrap();
        args.drain(index..index + 2);
        dir.reject(&args);
    }
    for flag in ["--a", "--b", "--games", "--sims", "--seed", "--out", "--max-plies",
        "--batch", "--opening-offset", "--moves-out", "--openings", "--ep", "--threads"] {
        let mut args = dir.args();
        args.push(flag.to_string());
        dir.reject(&args);
    }
}

#[test]
fn malformed_custom_books_are_rejected_without_replacing_outputs() {
    let dir = TestDir::new("books");
    let source = include_str!("../data/openings.txt");
    let mut texts = vec![source.lines().take(49).collect::<Vec<_>>().join("\n"), "e2e4 e7e5\n".repeat(50)];
    for replacement in ["e2e5 e7e5", "e2e4", "", "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1"] {
        let mut lines: Vec<_> = source.lines().collect();
        lines[0] = replacement;
        texts.push(lines.join("\n"));
    }
    let path = dir.0.join("invalid.txt");
    for text in texts {
        fs::write(&path, text).unwrap();
        let mut args = dir.args();
        set(&mut args, "--openings", path.to_str().unwrap());
        dir.reject(&args);
    }
}

#[test]
fn output_paths_that_overlap_a_temporary_name_are_published_safely_or_rejected() {
    let dir = TestDir::new("temp-collision");
    let first = dir.0.join("collision.json");
    let second = dir.0.join("collision.json.tmp");
    for reverse in [false, true] {
        fs::write(&first, b"first sentinel").unwrap();
        fs::write(&second, b"second sentinel").unwrap();
        let (json, moves) = if reverse { (&second, &first) } else { (&first, &second) };
        let mut args = dir.args();
        set(&mut args, "--out", json.to_str().unwrap());
        set(&mut args, "--moves-out", moves.to_str().unwrap());
        let output = command(&args);
        if output.status.success() {
            let report: serde_json::Value = serde_json::from_slice(&fs::read(json).unwrap()).unwrap();
            assert_eq!(report["format"], "chessbot-match-v1");
            assert_eq!(report["games"], 2);
            let log = fs::read_to_string(moves).unwrap();
            assert_eq!(log.lines().count(), 2);
            for line in log.lines() { assert_eq!(line.split_whitespace().count(), 11); }
        } else {
            assert_eq!(fs::read(&first).unwrap(), b"first sentinel");
            assert_eq!(fs::read(&second).unwrap(), b"second sentinel");
        }
        assert_eq!(fs::read(dir.0.join("keep.json")).unwrap(), b"keep json");
        assert_eq!(fs::read(dir.0.join("keep.txt")).unwrap(), b"keep log");
    }
}

#[test]
fn equivalent_output_paths_are_rejected_before_publishing() {
    let dir = TestDir::new("equivalent-paths");
    fs::create_dir_all(dir.0.join("child")).unwrap();
    for suffix in ["./keep.json", "child/../keep.json"] {
        let mut args = dir.args();
        set(&mut args, "--moves-out", dir.0.join(suffix).to_str().unwrap());
        dir.reject(&args);
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn explicit_coreml_is_rejected_on_linux() {
    let dir = TestDir::new("coreml");
    let mut args = dir.args();
    set(&mut args, "--ep", "coreml");
    dir.reject(&args);
}

#[test]
fn match_help_advertises_all_supported_flags() {
    let args = vec!["--help".to_string()];
    let output = command(&args);
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    for flag in ["--a", "--b", "--games", "--sims", "--seed", "--out", "--max-plies",
        "--batch", "--opening-offset", "--moves-out", "--openings", "--ep", "--threads"] {
        assert!(text.contains(flag), "help omits {flag}");
    }
    let help = Command::new(env!("CARGO_BIN_EXE_engine")).arg("--help").output().unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8(help.stdout).unwrap().contains("match --a"));
}
