//! GATE 1.1: move generation correctness via perft.

use engine::board::{from_fen, perft};

#[test]
fn perft4_startpos() {
    let board = cozy_chess::Board::default();
    assert_eq!(perft(&board, 4), 197_281);
}

#[test]
fn perft3_kiwipete() {
    // Exercises castling, en passant, promotions and pins.
    let board =
        from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
    assert_eq!(perft(&board, 3), 97_862);
}
