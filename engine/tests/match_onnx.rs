//! Real ONNX sessions coexist and can be reconstructed across paired games.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;

const MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/match_uniform.onnx");

struct TestDir(PathBuf);
impl TestDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("chessbot-match-onnx-{name}-{}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn run(&self, a: &str, b: &str) -> (Vec<u8>, String, Value) {
        let json = self.0.join("match.json");
        let moves = self.0.join("moves.txt");
        let output = Command::new(env!("CARGO_BIN_EXE_engine"))
            .args(["match", "--a", a, "--b", b, "--games", "4", "--sims", "4",
                "--max-plies", "10", "--seed", "13", "--batch", "8", "--ep", "cpu",
                "--threads", "1", "--out"])
            .arg(&json)
            .arg("--moves-out")
            .arg(&moves)
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let bytes = fs::read(json).unwrap();
        let report: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(report["a"], a);
        assert_eq!(report["b"], b);
        assert_eq!(report["games"], 4);
        assert_eq!(report["a_score"], 0.5);
        let log = fs::read_to_string(moves).unwrap();
        let lines: Vec<_> = log.lines().map(|line| line.split_whitespace().collect::<Vec<_>>()).collect();
        assert_eq!(lines.len(), 4);
        for pair in lines.chunks_exact(2) {
            assert_eq!(&pair[0][2..], &pair[1][2..]);
            assert_eq!(pair[0].len(), 13); // game, white, result, then ten actual plies.
        }
        assert!(report["results"].as_array().unwrap().iter().all(|game| game["plies"] == 10));
        (bytes, log, report)
    }
}
impl Drop for TestDir {
    fn drop(&mut self) { fs::remove_dir_all(&self.0).unwrap(); }
}

#[test]
fn two_onnx_players_share_runtime_and_reload_deterministically_between_games() {
    let dir = TestDir::new("two-models");
    let first = dir.run(MODEL, MODEL);
    let second = dir.run(MODEL, MODEL);
    assert_eq!((first.0, first.1), (second.0, second.1));
}

#[test]
fn an_onnx_player_matches_uniform_logits_from_the_builtin_player() {
    let dir = TestDir::new("mixed");
    let (_, mixed_moves, mixed) = dir.run(MODEL, "uniform");
    let (_, uniform_moves, uniform) = dir.run("uniform", "uniform");
    assert_eq!(mixed["results"], uniform["results"]);
    assert_eq!(mixed_moves, uniform_moves);
}
