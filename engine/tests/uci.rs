//! UCI notation round-trips and the command loop over in-memory I/O.

use std::io::Cursor;

use engine::board::{from_fen, legal_moves};
use engine::cli::dump_vectors::TEST_FENS;
use engine::nn::ort_infer::UniformEvaluator;
use engine::uci::notation::{move_from_uci, move_to_uci};
use engine::uci::uci_loop::{parse_go, parse_position, run};

#[test]
fn notation_roundtrips_every_legal_move() {
    for (_, fen) in TEST_FENS {
        let board = from_fen(fen).unwrap();
        for mv in legal_moves(&board) {
            let text = move_to_uci(&board, mv);
            assert_eq!(move_from_uci(&board, &text).unwrap(), mv, "{text} in {fen}");
        }
    }
}

#[test]
fn castling_and_promotion_use_standard_uci() {
    let board = from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
    assert_eq!(move_to_uci(&board, "e1h1".parse().unwrap()), "e1g1");
    assert_eq!(move_to_uci(&board, "e1a1".parse().unwrap()), "e1c1");
    assert!(move_from_uci(&board, "e1g1").is_ok());
    assert!(move_from_uci(&board, "e1h1").is_err(), "cozy form must not be accepted");
    let promo = from_fen("n1n5/1P6/8/8/8/8/8/K6k w - - 0 1").unwrap();
    assert_eq!(move_from_uci(&promo, "b7a8n").unwrap().to_string(), "b7a8n");
    assert!(move_from_uci(&promo, "b7b8").is_err(), "promotion piece is required");
}

#[test]
fn position_command_applies_moves() {
    let b = parse_position(&"startpos moves e2e4 e7e5 g1f3".split(' ').collect::<Vec<_>>()).unwrap();
    let want = from_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2").unwrap();
    assert!(b.same_position(&want));
    let castled = parse_position(
        &"fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1 moves e1g1 e8c8".split(' ').collect::<Vec<_>>(),
    )
    .unwrap();
    let want = from_fen("2kr3r/8/8/8/8/8/8/R4RK1 w - - 2 2").unwrap();
    assert!(castled.same_position(&want));
    assert!(parse_position(&["startpos", "moves", "e2e5"]).is_err());
    assert!(parse_position(&["nonsense"]).is_err());
}

#[test]
fn go_limits_parse() {
    let l = parse_go(&["nodes", "400"]).unwrap();
    assert_eq!(l.nodes, Some(400));
    let l = parse_go(&["wtime", "1000", "movetime", "250"]).unwrap();
    assert_eq!(l.movetime.unwrap().as_millis(), 250);
    assert!(parse_go(&["nodes", "abc"]).is_err());
}

#[test]
fn loop_handshake_and_search() {
    let script = "uci\nisready\nposition startpos moves e2e4\ngo nodes 50\nbogus\nposition startpos moves e2e5\nisready\nquit\ngo nodes 5\n";
    let mut out = Vec::new();
    run(Box::new(UniformEvaluator), Cursor::new(script), &mut out).unwrap();
    let text = String::from_utf8(out).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert!(lines.contains(&"uciok"));
    assert_eq!(lines.iter().filter(|l| **l == "readyok").count(), 2);
    let best: Vec<&&str> = lines.iter().filter(|l| l.starts_with("bestmove ")).collect();
    assert_eq!(best.len(), 1, "commands after quit must not run: {text}");
    let after_e4 = parse_position(&["startpos", "moves", "e2e4"]).unwrap();
    let mv = best[0].trim_start_matches("bestmove ");
    assert!(move_from_uci(&after_e4, mv).is_ok(), "bestmove {mv} not legal");
    assert!(lines.iter().any(|l| l.starts_with("info nodes 50 ")));
    assert!(lines.iter().any(|l| l.starts_with("info string unknown command: bogus")));
    assert!(lines.iter().any(|l| l.starts_with("info string error:")));
}
