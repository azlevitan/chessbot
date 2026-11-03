//! The shared on-disk `TrainingSample` record (SPEC.md §3). 492 bytes, little-endian.

use bytemuck::{Pod, Zeroable};

/// Number of sparse policy slots per sample.
pub const POLICY_SLOTS: usize = 64;
/// Exact on-disk size of one `TrainingSample`.
pub const SAMPLE_BYTES: usize = 492;
/// `move_idx` sentinel for an unused policy slot.
pub const UNUSED_MOVE: u16 = u16::MAX;

#[cfg(not(target_endian = "little"))]
compile_error!("TrainingSample is written with native byte order; a little-endian target is required");

#[repr(C, packed)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SparsePolicyEntry {
    /// Policy index in `0..4168`, or `UNUSED_MOVE`.
    pub move_idx: u16,
    /// Target probability (MCTS visit share, or 1.0/0.0 for supervised data).
    pub prob: f32,
}

impl SparsePolicyEntry {
    pub const UNUSED: Self = Self { move_idx: UNUSED_MOVE, prob: 0.0 };
}

#[repr(C, packed)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TrainingSample {
    /// Planes 0..11 (us P,N,B,R,Q,K then them P..K), canonical orientation.
    pub bitboards: [u64; 12],
    /// bit0 us K, bit1 us Q, bit2 them K, bit3 them Q, bit4 has_ep.
    pub castling_and_ep: u8,
    /// Canonical EP target square (40..47) when has_ep, else 0.
    pub ep_square: u8,
    /// Halfmove clock clamped to 0..=100.
    pub halfmove_clock: u8,
    /// 0 = Loss, 1 = Draw, 2 = Win from `us` perspective.
    pub outcome_wdl: u8,
    /// MCTS root evaluation in [-1, 1].
    pub q_value: f32,
    /// Active entries sorted by ascending `move_idx`, then `UNUSED` padding.
    pub policy: [SparsePolicyEntry; POLICY_SLOTS],
    /// Number of active entries in `policy`.
    pub num_legal_moves: u32,
}

const _: () = assert!(core::mem::size_of::<SparsePolicyEntry>() == 6);
const _: () = assert!(core::mem::size_of::<TrainingSample>() == SAMPLE_BYTES);
const _: () = assert!(core::mem::align_of::<TrainingSample>() == 1);

impl TrainingSample {
    /// Raw little-endian bytes exactly as written to disk.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    /// Read one sample from exactly `SAMPLE_BYTES` bytes.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        bytemuck::pod_read_unaligned(bytes)
    }

    /// Active policy entries as `(move_idx, prob)` pairs.
    pub fn policy_entries(&self) -> Vec<(u16, f32)> {
        let policy = self.policy; // copy out of the packed struct
        let n = self.num_legal_moves as usize;
        policy[..n].iter().map(|e| (e.move_idx, e.prob)).collect()
    }
}
