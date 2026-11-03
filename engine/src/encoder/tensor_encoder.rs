//! Board -> canonical `18x8x8` f32 planes and -> `TrainingSample` (SPEC.md §2, §3).

use cozy_chess::{Board, Color, Move, Piece};

use super::move_mapper::move_to_index;
use super::sample::{SparsePolicyEntry, TrainingSample, POLICY_SLOTS};

pub const NUM_PLANES: usize = 18;
pub const TENSOR_LEN: usize = NUM_PLANES * 64;

const PIECES: [Piece; 6] = [
    Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King,
];

/// Vertical flip (`sq ^ 56`) of a whole bitboard == byte swap.
#[inline]
fn orient(bb: u64, stm: Color) -> u64 {
    if stm == Color::Black { bb.swap_bytes() } else { bb }
}

/// Planes 0..11 as canonical bitboards: us P..K, then them P..K.
pub fn encode_bitboards(board: &Board) -> [u64; 12] {
    let us = board.side_to_move();
    let mut out = [0u64; 12];
    for (i, &piece) in PIECES.iter().enumerate() {
        out[i] = orient(board.colored_pieces(us, piece).0, us);
        out[6 + i] = orient(board.colored_pieces(!us, piece).0, us);
    }
    out
}

/// bit0 us K, bit1 us Q, bit2 them K, bit3 them Q (castling *rights*).
pub fn castling_bits(board: &Board) -> u8 {
    let us = board.side_to_move();
    let (ours, theirs) = (board.castle_rights(us), board.castle_rights(!us));
    (ours.short.is_some() as u8)
        | (ours.long.is_some() as u8) << 1
        | (theirs.short.is_some() as u8) << 2
        | (theirs.long.is_some() as u8) << 3
}

/// Canonical en-passant target square (always on canonical rank 6: 40..47).
pub fn ep_square(board: &Board) -> Option<u8> {
    board.en_passant().map(|file| 40 + file as u8)
}

/// Write the 18x64 planes from already-canonical components into `out`.
fn write_planes(bitboards: &[u64; 12], castling: u8, ep: Option<u8>, hmc: u8, out: &mut [f32]) {
    assert_eq!(out.len(), TENSOR_LEN);
    out.fill(0.0);
    for (plane, &bb) in bitboards.iter().enumerate() {
        let mut rest = bb;
        while rest != 0 {
            let sq = rest.trailing_zeros() as usize;
            out[plane * 64 + sq] = 1.0;
            rest &= rest - 1;
        }
    }
    for bit in 0..4 {
        if castling & (1 << bit) != 0 {
            out[(12 + bit) * 64..(13 + bit) * 64].fill(1.0);
        }
    }
    if let Some(sq) = ep {
        out[16 * 64 + sq as usize] = 1.0;
    }
    out[17 * 64..].fill(hmc.min(100) as f32 / 100.0);
}

/// Encode `board` into a flat `18*64` f32 tensor (plane-major) into `out`.
pub fn encode_planes_into(board: &Board, out: &mut [f32]) {
    write_planes(
        &encode_bitboards(board),
        castling_bits(board),
        ep_square(board),
        board.halfmove_clock(),
        out,
    );
}

/// Encode `board` into a flat `18*64` f32 tensor (plane-major).
pub fn encode_planes(board: &Board) -> Vec<f32> {
    let mut out = vec![0.0; TENSOR_LEN];
    encode_planes_into(board, &mut out);
    out
}

/// Reconstruct the f32 planes from a stored sample (mirrors the Python reader).
pub fn sample_to_planes(sample: &TrainingSample) -> Vec<f32> {
    let bitboards = sample.bitboards;
    let flags = sample.castling_and_ep;
    let ep = (flags & 0x10 != 0).then_some(sample.ep_square);
    let mut out = vec![0.0; TENSOR_LEN];
    write_planes(&bitboards, flags & 0x0F, ep, sample.halfmove_clock, &mut out);
    out
}

/// Convert `(move, prob)` pairs into sorted sparse entries, applying the
/// SPEC.md §3 64-slot rule (keep top-64 by prob, ties -> lower index, renormalize).
pub fn sparse_policy(board: &Board, policy: &[(Move, f32)]) -> Vec<(u16, f32)> {
    let mut entries: Vec<(u16, f32)> = policy
        .iter()
        .map(|&(mv, p)| (move_to_index(board, mv) as u16, p))
        .collect();
    if entries.len() > POLICY_SLOTS {
        entries.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        entries.truncate(POLICY_SLOTS);
        let total: f32 = entries.iter().map(|e| e.1).sum();
        if total > 0.0 {
            entries.iter_mut().for_each(|e| e.1 /= total);
        }
    }
    entries.sort_by_key(|e| e.0);
    entries
}

/// Build the on-disk sample. `policy` should list every legal move of `board`
/// (prob 0.0 for unvisited / unplayed moves) so the entries form the legal mask.
pub fn encode_sample(
    board: &Board,
    policy: &[(Move, f32)],
    outcome_wdl: u8,
    q_value: f32,
) -> TrainingSample {
    debug_assert!(outcome_wdl <= 2, "outcome_wdl must be 0, 1 or 2");
    let entries = sparse_policy(board, policy);
    let mut slots = [SparsePolicyEntry::UNUSED; POLICY_SLOTS];
    for (slot, &(move_idx, prob)) in slots.iter_mut().zip(&entries) {
        *slot = SparsePolicyEntry { move_idx, prob };
    }
    let ep = ep_square(board);
    TrainingSample {
        bitboards: encode_bitboards(board),
        castling_and_ep: castling_bits(board) | (ep.is_some() as u8) << 4,
        ep_square: ep.unwrap_or(0),
        halfmove_clock: board.halfmove_clock().min(100),
        outcome_wdl,
        q_value,
        policy: slots,
        num_legal_moves: entries.len() as u32,
    }
}
