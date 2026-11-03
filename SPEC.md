# SPEC.md — Technical Contracts & Data Schemas

## 1. Canonical Perspective & Move Indexing (`4,168` Action Space)
All boards and moves are canonicalized relative to the **side to move** (`us` vs `them`).
- If `side_to_move == Color::Black`, mirror every square vertically (`sq_idx ^ 56`) before encoding planes or move indices.
- **Deterministic Move Bijection (`0..4168`):**
  Given canonical `from_sq` (`0..63`) and `to_sq` (`0..63`):
  1. For all non-promotions AND Queen promotions (`Promotion::None` or `Promotion::Queen`):
     `index = from_sq * 64 + to_sq` (indices `0..4095`).
  2. For underpromotions (`Knight = 0`, `Bishop = 1`, `Rook = 2`), where a pawn moves from rank 6 (`from_sq` in `48..55`):
     - `from_file = from_sq % 8` (`0..7`)
     - `delta = (to_sq % 8) as i32 - (from_file as i32) + 1` (`0` for left capture, `1` for straight, `2` for right capture)
     - `index = 4096 + (from_file * 9) + (delta as usize * 3) + promo_idx` (indices `4096..4167`).
- **Castling** is indexed by the king's destination square (`e1g1` / `e1c1` canonical), matching UCI and python-chess — NOT cozy-chess's internal king-captures-rook form. The conversion lives only in `move_mapper.rs`.
- **Decoding needs the board:** a Queen promotion shares `from*64+to` with no other legal move, but its index is indistinguishable from a non-promotion without context. `index_to_move(&Board, idx)` attaches `Promotion::Queen` when the piece on `from` is a pawn landing on the last rank.
- **Illegal Move Masking:** Both in MCTS and loss computation, logits for illegal moves MUST be masked to `-1e9` prior to Softmax.

## 2. Input Tensor Schema (`18 x 8 x 8` Float32 Planes)
Constructed in Rust from a `18 * 64` flat array:
- Planes `0..=5`: `us` pieces (`Pawn, Knight, Bishop, Rook, Queen, King`). `1.0` if present on square, else `0.0`.
- Planes `6..=11`: `them` pieces (`Pawn, Knight, Bishop, Rook, Queen, King`), already vertically flipped if Black is to move.
- Plane `12`: `us` kingside castling right (all `1.0` if the *right* is held — not whether castling is legal this move — else `0.0`).
- Plane `13`: `us` queenside castling right.
- Plane `14`: `them` kingside castling right.
- Plane `15`: `them` queenside castling right.
- Plane `16`: En passant target square (`1.0` on the canonical target square if available, else `0.0`).
- Plane `17`: Halfmove (50-move rule) clock normalized as `halfmove_clock as f32 / 100.0`.

## 3. Shared Binary Training Sample (`#[repr(C, packed)]`)
Written by Rust (`engine`) and read via `np.memmap` in Python (`training`). Total size: **492 bytes per sample** (little-endian).

~~~rust
#[repr(C, packed)]
pub struct SparsePolicyEntry {
    pub move_idx: u16, // 0..4167 (u16::MAX = unused padding slot)
    pub prob: f32,     // MCTS visit probability (or 1.0 for supervised played move)
}

#[repr(C, packed)]
pub struct TrainingSample {
    pub bitboards: [u64; 12],             // 96 bytes: Planes 0..11 packed as 64-bit integers
    pub castling_and_ep: u8,              // 1 byte: bit0 = us K, bit1 = us Q, bit2 = them K, bit3 = them Q, bit4 = has_ep
    pub ep_square: u8,                    // 1 byte: canonical ep target square (40..47 when has_ep, else 0)
    pub halfmove_clock: u8,               // 1 byte: clamped to 0..100
    pub outcome_wdl: u8,                  // 1 byte: 0 = Loss, 1 = Draw, 2 = Win (from `us` perspective)
    pub q_value: f32,                     // 4 bytes: MCTS root evaluation in [-1.0, 1.0]
    pub policy: [SparsePolicyEntry; 64],  // 384 bytes: legal moves & probabilities
    pub num_legal_moves: u32,             // 4 bytes: Number of active entries in `policy`
}
~~~

**Policy slot rules:**
- Every legal move is written (prob `0.0` for unvisited / unplayed moves), so the active entries are the exact legal-move mask for the loss.
- Unused slots are `{ move_idx: u16::MAX, prob: 0.0 }`.
- If a position has more than 64 legal moves, keep the top 64 by prob (ties → lower `move_idx`) and renormalize the kept probs to sum to 1. `num_legal_moves` is then 64 (entries written), not the true legal count.
- Active entries are written sorted by ascending `move_idx`.

*Note for Python reader:* Unpack the 12 `uint64` bitboards into `(12, 8, 8)` `float32` planes using bitwise shifts `((bb >> sq) & 1)` inside `__getitem__` so disk storage stays compact.

## 4. Neural Network & ONNX Contract
- **Architecture:** `SE-ResNet-6x64`
  - Stem: `Conv2d(18, 64, kernel_size=3, padding=1, bias=False)` -> `BatchNorm2d(64)` -> `ReLU`
  - Trunk: 6 Residual Blocks, each with two `3x3` conv layers (64 channels) + Squeeze-and-Excitation (reduction ratio 4).
  - Policy Head: `Conv2d(64, 32, kernel_size=1)` -> `BN` -> `ReLU` -> `Flatten` -> `Linear(32*64, 4168)` outputting `policy_logits`.
  - Value (WDL) Head: `Conv2d(64, 8, kernel_size=1)` -> `BN` -> `ReLU` -> `Flatten` -> `Linear(8*64, 64)` -> `ReLU` -> `Linear(64, 3)` outputting `wdl_logits` (`[Loss, Draw, Win]`).
- **ONNX Export Contract:**
  - Opset: `17`
  - Input name: `"input_planes"`, shape `[batch_size, 18, 8, 8]`, type `float32`
  - Output 0 name: `"policy_logits"`, shape `[batch_size, 4168]`, type `float32`
  - Output 1 name: `"wdl_logits"`, shape `[batch_size, 3]`, type `float32`
  - Dynamic axis `0` (`batch_size`) on all inputs and outputs.
- **Rust `ort` (`=2.0.0-rc.15`) Session Initialization:**
  - Explicitly initialize the environment first via `let env = ort::init().build()?;` and pass `&env` into `Session::builder(&env)?` with the CoreML execution provider enabled.

## 5. MCTS Search Formula (PUCT)
- Scalar value from `wdl_logits`: after Softmax `[p_loss, p_draw, p_win]`, compute `v = p_win - p_loss` in `[-1.0, 1.0]`.
- At node `s`, pick legal child action `a` maximizing:
  `Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))`
  where `c_puct = 1.5`, and `Q(s, a)` is negated when backing up between alternating sides to move.