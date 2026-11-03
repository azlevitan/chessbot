# CLAUDE.md — Project Constitution & M1 Hardware Guardrails

## Project Overview
A hybrid Rust + PyTorch deep learning chess engine (AlphaZero/Leela style) engineered specifically for an Apple M1 MacBook Pro with 8GB unified RAM.
- `engine/`: Rust workspace using `cozy-chess = "0.3.4"` for board state/move generation and `ort` (`2.0.0-rc.15`) for neural network inference.
- `training/`: Python 3.11+ & PyTorch (`mps` backend) for training a Dual-Head Squeeze-and-Excitation ResNet and exporting to ONNX.

## Strict Architectural Laws (NEVER Violate)
1. **Single Source of Truth for Encoding:** NEVER write board-to-tensor or move-to-index encoding logic in Python. The Rust `engine` binary encodes all positions and writes flat `#[repr(C)]` binary files (`.bin`). Python only reads those exact bytes via `numpy.memmap`.
2. **8GB Unified RAM Guardrails:**
   - In Python `DataLoader`, ALWAYS set `num_workers=0` and `pin_memory=False` (macOS multiprocessing duplicates process memory and causes OOM on 8GB M1).
   - Never load full `.bin` datasets into RAM; always slice lazily inside `__getitem__` from a `np.memmap` buffer.
   - Cap the Rust MCTS tree arena at 65,536 nodes per search (`~4MB` RAM).
   - Always call `torch.mps.empty_cache()` at the end of every training epoch.
3. **Perspective Normalization:** Every board state MUST be encoded from the perspective of the **side to move**. If Black is to move, vertically flip all bitboards (`square ^ 56`) and swap White/Black piece planes before encoding.
4. **Modularity:** Keep every Rust and Python source file under 250 lines. Read `SPEC.md` before writing or modifying any data structure.

## Commands for Verification
- Build & test Rust engine: `cd engine && cargo test --release`
- Run Cross-Language Contract Gate: `pytest training/tests/test_contract.py`
- Run Single-Batch Overfit Gate: `pytest training/tests/test_overfit.py`
- Run Engine UCI Smoke Test: `python3 pipeline/test_uci_mate.py`

## Workflow Rules
- Complete ONLY the current task requested from `TASKS.md`.
- Do NOT mark a task complete until its verification command exits with code `0`.