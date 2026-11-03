//! Thin wrapper over `cozy_chess::Board`: FEN parsing, legal move lists, perft.

use anyhow::{anyhow, Result};
use cozy_chess::{Board, Move};

/// Parse a standard (non-Shredder) FEN string.
pub fn from_fen(fen: &str) -> Result<Board> {
    Board::from_fen(fen, false).map_err(|e| anyhow!("invalid FEN {fen:?}: {e}"))
}

/// All legal moves in cozy-chess's native form (castling = king captures rook).
pub fn legal_moves(board: &Board) -> Vec<Move> {
    let mut moves = Vec::with_capacity(64);
    board.generate_moves(|piece_moves| {
        moves.extend(piece_moves);
        false
    });
    moves
}

/// Count leaf nodes of the legal move tree to `depth` plies (bulk-counted at depth 1).
pub fn perft(board: &Board, depth: u32) -> u64 {
    match depth {
        0 => 1,
        1 => {
            let mut nodes = 0;
            board.generate_moves(|piece_moves| {
                nodes += piece_moves.len() as u64;
                false
            });
            nodes
        }
        _ => {
            let mut nodes = 0;
            board.generate_moves(|piece_moves| {
                for mv in piece_moves {
                    let mut child = board.clone();
                    child.play_unchecked(mv);
                    nodes += perft(&child, depth - 1);
                }
                false
            });
            nodes
        }
    }
}
