//! cozy-chess `Move` <-> UCI long algebraic notation (castling as e1g1 / e1c1).

use anyhow::{anyhow, Result};
use cozy_chess::{Board, Move, Piece};

use crate::board::legal_moves;
use crate::encoder::move_mapper::standard_to;

/// UCI string for a legal move on `board` (e.g. `e2e4`, `e1g1`, `b7b8q`).
pub fn move_to_uci(board: &Board, mv: Move) -> String {
    let promo = match mv.promotion {
        Some(Piece::Queen) => "q",
        Some(Piece::Rook) => "r",
        Some(Piece::Bishop) => "b",
        Some(Piece::Knight) => "n",
        _ => "",
    };
    format!("{}{}{promo}", mv.from, standard_to(board, mv))
}

/// Parse a UCI move string, accepting only moves legal on `board`.
pub fn move_from_uci(board: &Board, text: &str) -> Result<Move> {
    legal_moves(board)
        .into_iter()
        .find(|&mv| move_to_uci(board, mv) == text)
        .ok_or_else(|| anyhow!("illegal or malformed move {text:?}"))
}
