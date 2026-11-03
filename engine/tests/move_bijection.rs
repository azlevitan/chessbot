//! GATE 1.2: the 4,168-index move bijection is unique and roundtrips (SPEC.md §1).

use std::collections::HashSet;

use cozy_chess::{Board, Color, GameStatus, Move};
use engine::board::{from_fen, legal_moves};
use engine::cli::dump_vectors::TEST_FENS;
use engine::encoder::move_mapper::{index_to_move, move_to_index, NUM_MOVES};

/// Deterministic xorshift64 so the test needs no `rand` dependency.
struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Check every legal move of `board`; returns the number checked.
fn check_position(board: &Board) -> usize {
    let moves = legal_moves(board);
    let mut seen = HashSet::new();
    for &mv in &moves {
        let idx = move_to_index(board, mv);
        assert!(idx < NUM_MOVES, "index {idx} out of range for {mv} in {board}");
        assert!(seen.insert(idx), "duplicate index {idx} for {mv} in {board}");
        assert_eq!(index_to_move(board, idx), Some(mv), "roundtrip failed for {mv} in {board}");
    }
    // Decoding the whole action space yields exactly the legal moves, nothing else.
    let decoded = (0..NUM_MOVES).filter_map(|i| index_to_move(board, i)).count();
    assert_eq!(decoded, moves.len(), "decoded move count mismatch in {board}");
    moves.len()
}

#[test]
fn random_playouts_roundtrip() {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let (mut positions, mut black_positions, mut moves_checked) = (0, 0, 0);
    while positions < 1000 {
        let mut board = Board::default();
        for _ply in 0..200 {
            if board.status() != GameStatus::Ongoing || positions >= 1000 {
                break;
            }
            moves_checked += check_position(&board);
            positions += 1;
            black_positions += (board.side_to_move() == Color::Black) as usize;
            let moves = legal_moves(&board);
            board.play_unchecked(moves[(rng.next() % moves.len() as u64) as usize]);
        }
    }
    assert_eq!(positions, 1000);
    assert!(black_positions > 400, "too few black-to-move positions: {black_positions}");
    assert!(moves_checked > 10_000, "suspiciously few moves checked: {moves_checked}");
}

#[test]
fn test_vector_positions_roundtrip() {
    for (_, fen) in TEST_FENS {
        check_position(&from_fen(fen).unwrap());
    }
}

fn idx(fen: &str, uci: &str) -> usize {
    let board = from_fen(fen).unwrap();
    let mv: Move = uci.parse().unwrap();
    assert!(board.is_legal(mv), "{uci} not legal in {fen}");
    move_to_index(&board, mv)
}

#[test]
fn hand_checked_indices() {
    let start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    assert_eq!(idx(start, "e2e4"), 12 * 64 + 28);
    // Black's e7e5 is canonically e2e4.
    let after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
    assert_eq!(idx(after_e4, "e7e5"), 796);
    // Castling: cozy king-takes-rook form -> king destination (e1g1 / e1c1).
    let castle_w = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
    assert_eq!(idx(castle_w, "e1h1"), 4 * 64 + 6);
    assert_eq!(idx(castle_w, "e1a1"), 4 * 64 + 2);
    let castle_b = "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1";
    assert_eq!(idx(castle_b, "e8h8"), 262);
    assert_eq!(idx(castle_b, "e8a8"), 258);
    // Promotions.
    let promo_w = "n1n5/1P6/8/8/8/8/8/K6k w - - 0 1";
    assert_eq!(idx(promo_w, "b7a8n"), 4096 + 9);
    assert_eq!(idx(promo_w, "b7b8r"), 4096 + 9 + 3 + 2);
    assert_eq!(idx(promo_w, "b7c8b"), 4096 + 9 + 6 + 1);
    assert_eq!(idx(promo_w, "b7c8q"), 49 * 64 + 58);
    // Black underpromotion mirrors to the same canonical index.
    let promo_b = "k6K/8/8/8/8/8/1p6/N1N5 b - - 0 1";
    assert_eq!(idx(promo_b, "b2a1n"), 4096 + 9);
    assert_eq!(idx(promo_b, "b2c1q"), 49 * 64 + 58);
}

#[test]
fn decode_rejects_out_of_range_and_illegal() {
    let board = Board::default();
    assert_eq!(index_to_move(&board, NUM_MOVES), None);
    assert_eq!(index_to_move(&board, 12 * 64 + 36), None); // e2e5
    assert_eq!(index_to_move(&board, 4096), None); // a7 underpromo, no pawn there
    // Castling index with no right / blocked path decodes to None.
    assert_eq!(index_to_move(&board, 262), None);
}
