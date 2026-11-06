//! Phase 8.2: paired match results, reproducible output and replayable UCI logs.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use cozy_chess::{Board, GameStatus};
use engine::board::legal_moves;
use engine::uci::notation::{move_from_uci, move_to_uci};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde_json::Value;

const BOOK: &str = include_str!("../data/openings.txt");

struct TestDir(PathBuf);
impl TestDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("chessbot-match-{name}-{}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn run(&self, args: &[&str]) -> (Vec<u8>, String, Value) {
        let json = self.0.join("match.json");
        let moves = self.0.join("moves.txt");
        let output = Command::new(env!("CARGO_BIN_EXE_engine"))
            .arg("match")
            .args(args)
            .args(["--ep", "cpu", "--out"])
            .arg(&json)
            .arg("--moves-out")
            .arg(&moves)
            .output()
            .unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let bytes = fs::read(json).unwrap();
        let value = serde_json::from_slice(&bytes).unwrap();
        (bytes, fs::read_to_string(moves).unwrap(), value)
    }
}
impl Drop for TestDir {
    fn drop(&mut self) { fs::remove_dir_all(&self.0).unwrap(); }
}

fn replay_logs(log: &str, report: &Value, book: &str) -> Vec<Vec<String>> {
    let rows = report["results"].as_array().unwrap();
    assert_eq!(log.lines().count(), rows.len());
    log.lines().zip(rows).map(|(line, row)| {
        let words: Vec<_> = line.split_whitespace().collect();
        assert_eq!(words[0].parse::<u64>().unwrap(), row["game"].as_u64().unwrap());
        assert_eq!(words[1], row["white"].as_str().unwrap());
        assert_eq!(words[2], row["result"].as_str().unwrap());
        let moves = &words[3..];
        assert_eq!(moves.len() as u64, row["plies"].as_u64().unwrap());
        let opening = book.lines().nth(row["opening"].as_u64().unwrap() as usize).unwrap();
        let prefix: Vec<_> = opening.split_whitespace().collect();
        assert!(moves.starts_with(&prefix), "opening missing from {line}");
        let mut board = Board::default();
        for text in moves {
            assert_eq!(board.status(), GameStatus::Ongoing);
            let mv = move_from_uci(&board, text).unwrap();
            assert_eq!(move_to_uci(&board, mv), *text);
            board.play(mv);
        }
        moves.iter().map(|mv| mv.to_string()).collect()
    }).collect()
}

#[test]
fn material_beats_random_with_reproducible_results_and_complete_legal_move_logs() {
    let dir = TestDir::new("material");
    let args = ["--a", "material", "--b", "random", "--games", "20", "--sims", "32",
        "--seed", "0", "--max-plies", "200"];
    let first = dir.run(&args);
    let report = &first.2;
    let expected_keys = ["a", "a_score", "a_wins", "b", "b_wins", "draws", "format", "games",
        "max_plies", "opening_offset", "results", "seed", "sims"];
    assert_eq!(report.as_object().unwrap().keys().map(String::as_str).collect::<Vec<_>>(), expected_keys);
    assert_eq!(report["format"], "chessbot-match-v1");
    assert_eq!(report["a"], "material");
    assert_eq!(report["b"], "random");
    assert_eq!(report["games"], 20);
    assert_eq!(report["sims"], 32);
    assert_eq!(report["seed"], 0);
    assert_eq!(report["max_plies"], 200);
    assert_eq!(report["opening_offset"], 0);
    let mut totals = [0u64; 3];
    let rows = report["results"].as_array().unwrap();
    assert_eq!(rows.len(), 20);
    for (game, row) in rows.iter().enumerate() {
        assert_eq!(row.as_object().unwrap().len(), 6);
        assert_eq!(row["game"], game);
        assert_eq!(row["white"], if game % 2 == 0 { "a" } else { "b" });
        assert_eq!(row["opening"], game / 2);
        assert!(row["plies"].as_u64().unwrap() <= 200);
        assert!(["checkmate", "rule_draw", "repetition", "insufficient_material", "ply_cap"]
            .contains(&row["termination"].as_str().unwrap()));
        let result = row["result"].as_str().unwrap();
        if result == "1/2-1/2" { totals[2] += 1; }
        else {
            assert!(["1-0", "0-1"].contains(&result));
            let a_won = (result == "1-0") == (game % 2 == 0);
            totals[usize::from(!a_won)] += 1;
        }
    }
    for (key, count) in ["a_wins", "b_wins", "draws"].into_iter().zip(totals) {
        assert_eq!(report[key], count);
    }
    assert_eq!(totals.iter().sum::<u64>(), 20);
    let score = (totals[0] as f64 + 0.5 * totals[2] as f64) / 20.0;
    assert_eq!(report["a_score"].as_f64().unwrap(), score);
    assert!(score >= 0.75, "material score was {score}");
    replay_logs(&first.1, report, BOOK);
    let second = dir.run(&args);
    assert_eq!((first.0, first.1), (second.0, second.1));
}

#[test]
fn uniform_players_have_colour_symmetric_pairs_at_both_batch_sizes() {
    let dir = TestDir::new("symmetric");
    for batch in ["1", "8"] {
        let (_, log, report) = dir.run(&["--a", "uniform", "--b", "uniform", "--games", "4",
            "--sims", "32", "--seed", "27", "--batch", batch]);
        assert_eq!(report["a_score"], 0.5);
        assert_eq!(report["a_wins"], report["b_wins"]);
        let moves = replay_logs(&log, &report, BOOK);
        for pair in moves.chunks_exact(2) { assert_eq!(pair[0], pair[1]); }
    }
}

#[test]
fn opening_offset_wraps_and_opening_moves_count_toward_the_ply_cap() {
    let dir = TestDir::new("offset");
    let (_, log, report) = dir.run(&["--a", "random", "--b", "random", "--games", "6",
        "--sims", "2", "--seed", "0", "--opening-offset", "49", "--max-plies", "8"]);
    assert_eq!(report["opening_offset"], 49);
    for (row, opening) in report["results"].as_array().unwrap().iter().zip([49, 49, 0, 0, 1, 1]) {
        assert_eq!(row["opening"], opening);
        assert_eq!(row["plies"], 8);
        assert_eq!(row["termination"], "ply_cap");
    }
    replay_logs(&log, &report, BOOK);
}

#[test]
fn custom_opening_book_is_used_and_castling_is_logged_in_standard_uci() {
    let dir = TestDir::new("custom");
    let mut lines: Vec<_> = BOOK.lines().collect();
    lines.rotate_left(1);
    let book = format!("{}\n", lines.join("\n"));
    let path = dir.0.join("custom.txt");
    fs::write(&path, &book).unwrap();
    let (_, log, report) = dir.run(&["--a", "uniform", "--b", "uniform", "--games", "2",
        "--sims", "2", "--seed", "0", "--max-plies", "8", "--openings", path.to_str().unwrap()]);
    let moves = replay_logs(&log, &report, &book);
    assert!(moves.iter().all(|line| line[6] == "e1g1"));
    assert!(report["results"].as_array().unwrap().iter().all(|row| row["opening"] == 0));
}

#[test]
fn random_players_are_reseeded_with_seed_plus_game_without_opening_rng_draws() {
    let dir = TestDir::new("seeded");
    let (_, log, report) = dir.run(&["--a", "random", "--b", "random", "--games", "4",
        "--sims", "2", "--seed", "99", "--max-plies", "20"]);
    let logs = replay_logs(&log, &report, BOOK);
    for (game, moves) in logs.iter().enumerate() {
        let mut rngs = [StdRng::seed_from_u64(99 + game as u64), StdRng::seed_from_u64(99 + game as u64)];
        let opening_len = BOOK.lines().nth(game / 2).unwrap().split_whitespace().count();
        let mut board = Board::default();
        for (ply, text) in moves.iter().enumerate() {
            if ply >= opening_len {
                let mv = *legal_moves(&board).choose(&mut rngs[ply % 2]).unwrap();
                assert_eq!(move_to_uci(&board, mv), *text, "game {game}, ply {ply}");
            }
            board.play(move_from_uci(&board, text).unwrap());
        }
    }
}

#[test]
fn terminal_openings_end_before_search_and_take_precedence_over_the_cap() {
    let dir = TestDir::new("terminal-openings");
    for (opening, cap, reason, result) in [
        ("f2f3 e7e5 g2g4 d8h4", "4", "checkmate", "0-1"),
        ("g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8", "8", "repetition", "1/2-1/2"),
    ] {
        let mut lines: Vec<_> = BOOK.lines().collect();
        lines[0] = opening;
        let book = lines.join("\n");
        let path = dir.0.join("terminal.txt");
        fs::write(&path, &book).unwrap();
        let (_, log, report) = dir.run(&[
            "--a", "material", "--b", "uniform", "--games", "2", "--sims", "2",
            "--seed", "0", "--max-plies", cap, "--openings", path.to_str().unwrap(),
        ]);
        assert_eq!(report["a_score"], 0.5);
        for row in report["results"].as_array().unwrap() {
            assert_eq!(row["termination"], reason);
            assert_eq!(row["result"], result);
            assert_eq!(row["plies"], cap.parse::<u32>().unwrap());
        }
        replay_logs(&log, &report, &book);
    }
}
