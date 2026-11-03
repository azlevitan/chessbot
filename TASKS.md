---
# TASKS.md — Phased Implementation Checklist

## Phase 1: Rust Board Wrapper, Move Bijection & Binary Encoder
- [x] Initialize Cargo project in `engine/` with `cozy-chess = "0.3.4"`, `bytemuck`, and `anyhow`.
- [x] Implement `engine/src/encoder/move_mapper.rs`: canonical square flipping and the 4,168 move bijection from `SPEC.md`.
- [x] Implement `engine/src/encoder/tensor_encoder.rs`: converts `cozy_chess::Board` into both the `18x8x8` `f32` tensor and the 492-byte `TrainingSample` struct.
- [x] Add a CLI subcommand `engine dump-test-vectors --out data_store/test_vectors.bin` that encodes 5 known FENs (startpos, Kiwipete, en-passant, underpromotion, black-to-move check).
- [x] **GATE 1:** `cd engine && cargo test --release` passes unit tests verifying:
  1. Perft(4) from startpos equals `197,281` nodes.
  2. Every legal move across 1,000 random playout positions maps to a unique index in `0..4168` and roundtrips back to the identical `cozy_chess::Move`.

## Phase 2: Python Memmap Reader & Cross-Language Contract Verification
- [x] Set up `training/` with `pyproject.toml` (`torch`, `numpy`, `onnx`, `onnxruntime`, `pytest`, `chess`).
- [x] Implement `training/data/replay_buffer.py`: `np.memmap` Dataset reading `TrainingSample` (`492` bytes) and unpacking `18x8x8` planes + `4168` dense policy target + WDL label.
- [x] Write `training/tests/test_contract.py` that runs `engine dump-test-vectors`, reads the resulting `.bin` file in Python, and asserts exact plane values, castling flags, and move indices against `python-chess`.
- [x] **GATE 2:** `pytest training/tests/test_contract.py` exits with `0`.

## Phase 3: SE-ResNet-6x64 Architecture, MPS Training & ONNX Export
- [x] Implement `training/models/net.py` (`SE-ResNet-6x64` matching `SPEC.md`).
- [x] Implement `training/losses.py`: Cross-Entropy on masked `policy_logits` + Cross-Entropy on `wdl_logits`.
- [x] Implement `training/export_onnx.py` to export a `.pt` checkpoint to `.onnx` and verify inference output parity with `onnxruntime`.
- [x] Write `training/tests/test_overfit.py`: trains on a single synthetic batch of 16 positions for 80 steps on `mps` (or `cpu` fallback) and asserts total loss drops below `0.05` and exported ONNX outputs match PyTorch within `1e-4`.
- [x] **GATE 3:** `pytest training/tests/test_overfit.py` exits with `0`.

## Phase 4: Rust ONNX Inference, Arena MCTS & UCI Loop
- [x] Add `ort = { version = "2.0.0-rc.15", features = ["coreml"] }` to `engine/Cargo.toml`.
- [x] Implement `engine/src/nn/ort_infer.rs`: loads an `.onnx` model (with dummy uniform fallback if no model path is passed) and evaluates batches of `18x8x8` planes.
- [x] Implement `engine/src/mcts/`: fixed-capacity arena MCTS (`65,536` max nodes) with PUCT (`c_puct = 1.5`), legal move masking, and perspective negation on backup.
- [x] Implement `engine/src/uci/uci_loop.rs`: supports `uci`, `isready`, `ucinewgame`, `position startpos/fen ... moves ...`, and `go nodes <N> / movetime <ms>`.
- [x] Write `pipeline/test_uci_mate.py`: spawns the compiled `engine` binary via subprocess, feeds 3 Mate-in-1 FENs with `go nodes 400`, and asserts the engine outputs the checkmating move every time.
- [x] **GATE 4:** `python3 pipeline/test_uci_mate.py` exits with `0`.

## Phase 5: Supervised Bootstrapping & Self-Play RL Loop
- [x] Add `engine convert-pgn --input <pgn> --out <bin>` in Rust to rapidly convert PGN games into `.bin` training chunks.
- [x] Add `engine selfplay --model <onnx> --games <N> --sims 200 --out <bin>` with root Dirichlet noise (`alpha=0.3, eps=0.25`).
- [x] **GATE 5:** `cd engine && cargo test --release` and `pytest training/tests/test_pgn_contract.py` exit with `0`.
