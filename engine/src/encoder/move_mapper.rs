//! Canonical square flipping and the 4,168-entry move bijection (SPEC.md §1).
//!
//! Castling is indexed by the king's destination (e1g1 / e1c1), matching UCI and
//! python-chess. cozy-chess internally encodes castling as king-captures-rook;
//! that conversion happens here and nowhere else.

use cozy_chess::{Board, Color, File, Move, Piece, Square};

/// Size of the policy action space.
pub const NUM_MOVES: usize = 4168;
/// First index of the underpromotion block.
pub const UNDERPROMO_BASE: usize = 4096;

const UNDERPROMOS: [Piece; 3] = [Piece::Knight, Piece::Bishop, Piece::Rook];

/// Canonical square index (0..64) from the side to move's perspective.
/// Self-inverse: applying it twice returns the original square.
#[inline]
pub fn canon_sq(sq: Square, stm: Color) -> usize {
    let idx = sq as usize;
    if stm == Color::Black { idx ^ 56 } else { idx }
}

#[inline]
fn uncanon_sq(idx: usize, stm: Color) -> Square {
    Square::index(if stm == Color::Black { idx ^ 56 } else { idx })
}

/// True if `mv` is a castling move in cozy-chess form (king moves onto its own rook).
#[inline]
fn is_cozy_castle(board: &Board, mv: Move) -> bool {
    board.piece_on(mv.from) == Some(Piece::King)
        && board.color_on(mv.to) == Some(board.side_to_move())
}

/// Destination square in standard notation: for castling, the king's landing
/// square (g1/c1/g8/c8) instead of cozy-chess's rook square.
pub fn standard_to(board: &Board, mv: Move) -> Square {
    if is_cozy_castle(board, mv) {
        let file = if mv.to.file() > mv.from.file() { File::G } else { File::C };
        Square::new(file, mv.from.rank())
    } else {
        mv.to
    }
}

/// Map a legal move on `board` to its canonical policy index in `0..NUM_MOVES`.
pub fn move_to_index(board: &Board, mv: Move) -> usize {
    let stm = board.side_to_move();
    let to = standard_to(board, mv);
    let from_c = canon_sq(mv.from, stm);
    let to_c = canon_sq(to, stm);
    match mv.promotion {
        None | Some(Piece::Queen) => from_c * 64 + to_c,
        Some(promo) => {
            debug_assert!((48..56).contains(&from_c), "underpromotion from non-7th rank");
            let promo_idx = UNDERPROMOS
                .iter()
                .position(|&p| p == promo)
                .expect("promotion piece must be N, B, R or Q");
            let from_file = from_c % 8;
            let delta = (to_c % 8) as i32 - from_file as i32 + 1;
            debug_assert!((0..3).contains(&delta), "underpromotion delta out of range");
            UNDERPROMO_BASE + from_file * 9 + delta as usize * 3 + promo_idx
        }
    }
}

/// Inverse of [`move_to_index`]: returns the cozy-chess move for `idx` if it is
/// legal on `board`, else `None`.
pub fn index_to_move(board: &Board, idx: usize) -> Option<Move> {
    let stm = board.side_to_move();
    let (from_c, to_c, underpromo) = if idx < UNDERPROMO_BASE {
        (idx / 64, idx % 64, None)
    } else if idx < NUM_MOVES {
        let rel = idx - UNDERPROMO_BASE;
        let from_file = rel / 9;
        let delta = (rel % 9) / 3;
        let to_file = (from_file + delta).checked_sub(1).filter(|&f| f < 8)?;
        (48 + from_file, 56 + to_file, Some(UNDERPROMOS[rel % 3]))
    } else {
        return None;
    };

    let from = uncanon_sq(from_c, stm);
    let mut to = uncanon_sq(to_c, stm);
    if board.color_on(from) != Some(stm) {
        return None;
    }
    // Landing on our own piece is never a valid index: cozy's king-takes-rook
    // castling form is reachable only via the king-destination index below.
    if board.color_on(to) == Some(stm) {
        return None;
    }
    let piece = board.piece_on(from)?;

    let mut promotion = underpromo;
    if piece == Piece::Pawn && underpromo.is_none() && to_c >= 56 {
        promotion = Some(Piece::Queen);
    }
    if piece == Piece::King
        && from.rank() == to.rank()
        && (from.file() as i32 - to.file() as i32).abs() == 2
    {
        let rights = board.castle_rights(stm);
        let rook_file = if to.file() > from.file() { rights.short } else { rights.long };
        to = Square::new(rook_file?, from.rank());
    }

    let mv = Move { from, to, promotion };
    board.is_legal(mv).then_some(mv)
}
