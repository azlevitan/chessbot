//! SAN resolution, PGN tokenizing, and PGN -> TrainingSample conversion.

use std::io::Cursor;

use engine::board::{from_fen, legal_moves};
use engine::cli::bin_writer::{GameResult, SampleWriter};
use engine::cli::convert_pgn::{convert, convert_game};
use engine::encoder::move_mapper::move_to_index;
use engine::encoder::sample::{TrainingSample, SAMPLE_BYTES};
use engine::pgn::{san_to_move, PgnReader};
use engine::uci::notation::move_to_uci;

fn san_uci(fen: &str, san: &str) -> String {
    let board = from_fen(fen).unwrap();
    move_to_uci(&board, san_to_move(&board, san).unwrap())
}

#[test]
fn san_resolves_every_move_kind() {
    let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    assert_eq!(san_uci(start, "e4"), "e2e4");
    assert_eq!(san_uci(start, "Nf3"), "g1f3");
    // Castling both sides, both notations, White and Black.
    let castle = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
    assert_eq!(san_uci(castle, "O-O"), "e1g1");
    assert_eq!(san_uci(castle, "0-0-0"), "e1c1");
    let castle_b = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1";
    assert_eq!(san_uci(castle_b, "O-O-O+"), "e8c8");
    // File and rank disambiguation.
    let knights = "4k3/8/8/8/8/8/8/1N1NK3 w - - 0 1";
    assert_eq!(san_uci(knights, "Nbc3"), "b1c3");
    assert_eq!(san_uci(knights, "Ndc3"), "d1c3");
    let rooks = "7k/8/8/8/4R3/8/8/4R1K1 w - - 0 1";
    assert_eq!(san_uci(rooks, "R1e2"), "e1e2");
    assert_eq!(san_uci(rooks, "R4e2"), "e4e2");
    // Captures, en passant, check/annotation suffixes.
    let ep = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3";
    assert_eq!(san_uci(ep, "exf6"), "e5f6");
    assert_eq!(san_uci(ep, "exf6!?"), "e5f6");
    // Promotions, with and without '=', including underpromotion with capture.
    let promo = "n1n5/1P6/8/8/8/8/8/K6k w - - 0 1";
    assert_eq!(san_uci(promo, "b8=Q"), "b7b8q");
    assert_eq!(san_uci(promo, "b8N+"), "b7b8n");
    assert_eq!(san_uci(promo, "bxa8=R"), "b7a8r");
    assert_eq!(san_uci(promo, "bxc8=B#"), "b7c8b");
}

#[test]
fn san_rejects_illegal_and_ambiguous() {
    let board = from_fen("4k3/8/8/8/8/8/8/1N1NK3 w - - 0 1").unwrap();
    assert!(san_to_move(&board, "Nc3").is_err(), "ambiguous knight move accepted");
    assert!(san_to_move(&board, "Qd4").is_err(), "move by a missing piece accepted");
    assert!(san_to_move(&board, "e9").is_err());
    let promo = from_fen("n1n5/1P6/8/8/8/8/8/K6k w - - 0 1").unwrap();
    assert!(san_to_move(&promo, "b8").is_err(), "promotion without piece accepted");
}

const PGN: &str = r#"[Event "Annotated"]
[White "A"]
[Result "1/2-1/2"]

1. e4 {a comment
spanning lines} e5 2. Nf3 (2. f4 exf4 (2... d5) 3. Nf3) 2... Nc6 $1 3. Bb5 a6 ; rest of line
% escaped line ignored
4. Ba4 Nf6 5. O-O 1/2-1/2

[Event "No termination token"]
[Result "0-1"]

1. f3 e5 2. g4 Qh4#
[Event "Unfinished"]
[Result "*"]

1. d4 *
"#;

#[test]
fn reader_skips_comments_variations_and_nags() {
    let mut reader = PgnReader::new(Cursor::new(PGN));
    let g1 = reader.next_game().unwrap().unwrap();
    assert_eq!(g1.tag("White"), Some("A"));
    assert_eq!(g1.sans, ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O"]);
    assert_eq!(g1.result(), Some(GameResult::Draw));
    let g2 = reader.next_game().unwrap().unwrap();
    assert_eq!(g2.sans, ["f3", "e5", "g4", "Qh4#"]);
    assert_eq!(g2.result(), Some(GameResult::BlackWins), "falls back to Result tag");
    let g3 = reader.next_game().unwrap().unwrap();
    assert_eq!(g3.result(), None);
    assert!(reader.next_game().unwrap().is_none());
}

#[test]
fn converts_game_with_labels_and_legal_mask() {
    let mut reader = PgnReader::new(Cursor::new(PGN));
    reader.next_game().unwrap();
    let game = reader.next_game().unwrap().unwrap(); // fool's mate, 0-1
    let samples = convert_game(&game).unwrap();
    assert_eq!(samples.len(), 4);
    let mut board = game.start_board().unwrap();
    for (ply, (s, san)) in samples.iter().zip(&game.sans).enumerate() {
        // White (even plies) lost, Black won.
        let expected_wdl = if ply % 2 == 0 { 0 } else { 2 };
        let (wdl, q) = (s.outcome_wdl, s.q_value);
        assert_eq!(wdl, expected_wdl, "ply {ply}");
        assert_eq!(q, if ply % 2 == 0 { -1.0 } else { 1.0 });
        let played = san_to_move(&board, san).unwrap();
        let entries = s.policy_entries();
        assert_eq!(entries.len(), legal_moves(&board).len().min(64));
        let hot: Vec<_> = entries.iter().filter(|e| e.1 != 0.0).collect();
        assert_eq!(hot.len(), 1);
        assert_eq!(*hot[0], (move_to_index(&board, played) as u16, 1.0));
        board.play_unchecked(played);
    }
}

#[test]
fn convert_streams_games_and_skips_bad_ones() {
    let bad = "[Result \"1-0\"]\n\n1. e4 e5 2. Ke3 1-0\n\n";
    let input = format!("{bad}{PGN}");
    let dir = std::env::temp_dir().join(format!("chessbot-pgn-{}", std::process::id()));
    let out = dir.join("out.bin");
    let mut writer = SampleWriter::create(&out).unwrap();
    let stats = convert(Cursor::new(input), &mut writer, None).unwrap();
    assert_eq!(writer.finish().unwrap(), 13);
    assert_eq!((stats.games_read, stats.converted, stats.skipped, stats.samples), (4, 2, 2, 13));
    let bytes = std::fs::read(&out).unwrap();
    assert_eq!(bytes.len(), 13 * SAMPLE_BYTES);
    let first = TrainingSample::from_bytes(&bytes[..SAMPLE_BYTES]);
    assert_eq!(first.outcome_wdl, 1, "first converted game is a draw");
    // --max-games counts games read, including skipped ones.
    let mut writer = SampleWriter::create(&out).unwrap();
    let stats = convert(Cursor::new(PGN), &mut writer, Some(1)).unwrap();
    assert_eq!((stats.games_read, stats.samples), (1, 9));
    std::fs::remove_dir_all(dir).unwrap();
}
