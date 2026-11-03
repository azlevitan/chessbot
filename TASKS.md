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

> Phase 5 is closed (GATE 5 passed). The rest of the supervised bootstrap (training script, generation 0) and the self-play RL loop continue in Phases 6–10 below.

---
## Roadmap for Phases 6–10
**Assumptions:**
- Target: every promoted generation beats the previous best (SPRT, SPEC §10) and the `random`/`material` baselines, and Elo is tracked as an outcome[cite: 1]. The ladder gives relative Elo per generation; an optional Stockfish calibration gives absolute Elo[cite: 1]. Not a lichess bot[cite: 1].
- Supervised bootstrap: one Lichess Elite monthly PGN (downloaded by the owner), capped at about 4.2M samples (about 2 GB of `.bin`)[cite: 1].
- One generation (selfplay + train + match) may take an **overnight ~8 h** on the M1[cite: 1].
- Inference: CoreML EP on macOS with automatic CPU fallback[cite: 1]. CPU-only everywhere else[cite: 1].
- Whole-pipeline peak RAM ≤ 3–4 GB[cite: 1]. Stages run as **separate sequential subprocesses**, so peak RAM is the maximum of any single stage, never the sum[cite: 1].
- Measured baselines (M1, commit `49c9462`): batch-1 CoreML ≈ 380 evals/s (a 200-sim selfplay game ≈ 80 s); tree-only search with the uniform net ≈ 1M sims/s[cite: 1].

**Rules for every task below:**
- Read `SPEC.md` (§6–§11 are the new contracts) before starting[cite: 1].
- Keep every touched `.rs`/`.py` file < 250 lines[cite: 1]. Split along the named files if needed[cite: 1].
- Python never encodes boards or moves[cite: 1]. It only reads §3 bytes through `np.memmap` (`training/data/replay_buffer.py` decoders)[cite: 1].
- Every `DataLoader` goes through `make_loader` (`num_workers=0`, `pin_memory=False`)[cite: 1].
- Every MCTS arena is capped at `MAX_NODES = 65,536`[cite: 1].
- Gate commands must pass on a Linux CPU box (no MPS, no CoreML) with fixed seeds in under ~5 minutes[cite: 1].
- `pipeline/*.py` scripts are stdlib-only[cite: 1]. They call training code through `subprocess` with `sys.executable training/<script>.py`, so each stage's RAM is freed when its process exits[cite: 1].
- Python tests for `pipeline/` live in `pipeline/tests/` and are run as `pytest pipeline/tests/<file>`[cite: 1].

## Phase 6: Chunked Data, Training Script & Supervised Generation 0
- [ ] **6.1 Portable inference options + CoreML fixed-bucket padding + uniform selfplay.**
  - Files: `engine/src/nn/ort_infer.rs`, `engine/src/nn/infer_opts.rs` (new), `engine/src/nn/mod.rs`, `engine/src/cli/selfplay.rs`, `engine/src/main.rs`, `engine/tests/infer_opts.rs` (new)[cite: 1].
  - Add `pub enum Ep { Auto, Cpu, CoreMl }` and `pub struct InferOptions { pub ep: Ep, pub threads: usize }` (default `Auto`, `1`), with `InferOptions::parse_flag(flag, value) -> Result<bool>` for `--ep` / `--threads`[cite: 1].
  - Change the loader to `load_evaluator(model: Option<&Path>, opts: &InferOptions)`[cite: 1]. Register CoreML only when `cfg!(target_os = "macos")` and `ep != Cpu`[cite: 1]. `ep == CoreMl` on another OS is an error[cite: 1]. Set `with_intra_threads(opts.threads)`[cite: 1].
  - **CoreML Fixed-Bucket Padding:** In `OrtEvaluator::evaluate`, when `actual_batch = batch.len()` is not in `{1, 8, 16, 32, 64}` and CoreML is active, zero-pad the input tensor batch up to the next bucket size in `{1, 8, 16, 32, 64}` (or run in chunks of 64 if `> 64`), run inference, and slice output logits back to `0..actual_batch`. This prevents Apple Neural Engine graph recompilation on partial tail batches.
  - `engine selfplay`: `--model` becomes optional (omitted or `uniform` → `UniformEvaluator`)[cite: 1]. Add `--ep` and `--threads` to `selfplay` and `uci`[cite: 1].
  - Test (Rust cannot export ONNX, so no real model is loaded here; ONNX + `--ep cpu` is exercised by 6.5's test)[cite: 1]:
    1. `load_evaluator(None, &InferOptions::default())` works[cite: 1].
    2. Flag parsing accepts `--ep cpu` and `--threads 2` and rejects `--ep gpu`[cite: 1].
    3. `next_bucket(n)` maps `1->1`, `2..=8->8`, `9..=16->16`, `17..=32->32`, `33..=64->64`.
    4. `engine selfplay --games 1 --sims 8 --max-plies 10 --seed 1 --out <tmp>` with **no** `--model` exits 0 and writes `10 * 492` bytes[cite: 1].
  - `GATE 5` must still pass[cite: 1].
- [ ] **6.2 Chunk writer with atomic write + sidecar (SPEC §6).**
  - Files: `engine/Cargo.toml` (add `serde = { version = "1", features = ["derive"] }`, `serde_json = "1"`), `engine/src/cli/chunk_writer.rs` (new; wraps `SampleWriter`), `engine/src/cli/mod.rs`, `engine/src/cli/selfplay.rs`, `engine/src/cli/convert_pgn.rs`, `engine/src/selfplay/stats.rs` (new: result/termination counters), `engine/tests/chunks.rs` (new)[cite: 1].
  - Interface: `ChunkWriter::create(bin_path) -> Result<Self>` (writes `.bin.tmp`), `.write(&TrainingSample)`, `.finish(sidecar: serde_json::Value) -> Result<u64>`[cite: 1]. `finish` does the rename dance from §6 and fills `"format"` and `"samples"`[cite: 1].
  - `selfplay --out X.bin` now writes `X.bin` and `X.json` (selfplay sidecar from §6; `resign` block with `threshold: -1.0`, zeros until 7.5)[cite: 1].
  - `convert-pgn` keeps `--out <bin>` (one chunk plus a sidecar, so GATE 5 is unchanged) and adds[cite: 1]:
    - `--out-dir <dir> --chunk-samples N` (default `262144`), which writes `chunk_000000.bin…`[cite: 1]
    - `--max-samples M` (stop after the game that crosses `M`)[cite: 1]
    - `--skip-games K` (skip the first `K` games read; sidecar `first_game`)[cite: 1]
    - `--input -` (stdin, so `zstdcat file.pgn.zst | engine convert-pgn --input - …` never writes the decompressed PGN to disk)[cite: 1]
  - Test (`engine/tests/chunks.rs`)[cite: 1]:
    1. A 2-game uniform selfplay (`--sims 8 --max-plies 20 --seed 3`) leaves no `*.tmp`[cite: 1]. The sidecar has `samples * 492 == bin size`, `results` summing to `games`, and `terminations` summing to `games`[cite: 1].
    2. Converting `training/tests/fixtures/sample.pgn` with `--out-dir --chunk-samples 25` gives chunks of 25, 25 and 10 samples (60 total) with `games` summing to 3[cite: 1].
    3. `--skip-games 1` gives 3 − 1 games' samples[cite: 1].
  - RAM: ≤ 1 chunk buffered by `BufWriter` (KB scale)[cite: 1].
- [ ] **6.3 Multi-chunk memmap dataset + weighted source sampler.**
  - Files: `training/data/chunk_dataset.py` (new), `training/data/sampler.py` (new), `training/tests/test_chunk_dataset.py` (new)[cite: 1].
  - Interface:
    - `read_sidecar(bin_path) -> dict`[cite: 1]
    - `list_complete_chunks(directory) -> list[Path]` (sorted; SPEC §6 completeness rule)[cite: 1]
    - `class ChunkDataset(Dataset)` with `__init__(paths: list[Path])`, `__len__`, `__getitem__(i)` (same dict as `ReplayBuffer.__getitem__`, including `"q_value"` as a float32 tensor in `[-1.0, 1.0]`), and a vectorized `__getitems__(indices: list[int])` that returns the collated batch dict[cite: 1]. It opens one `np.memmap` per chunk via `open_samples` and maps the global index with `np.searchsorted` on cumulative lengths[cite: 1]. Nothing is copied into RAM except the requested records[cite: 1].
    - `class WeightedSourceSampler(Sampler[int])` with `__init__(source_ranges: list[tuple[int, int]], weights: list[float], num_samples: int, seed: int)`: per draw, it picks a source by weight (zero-weight or empty sources excluded), then a uniform index in `[start, end)`[cite: 1]. It uses `numpy.random.default_rng(seed)`[cite: 1].
  - Reuse `unpack_planes` / `unpack_policy` from `replay_buffer.py`[cite: 1]. Do not duplicate the decoders[cite: 1].
  - Test, using `engine_bin` from `conftest.py` to make one uniform selfplay chunk (`--games 2 --sims 8 --max-plies 20 --seed 5`) and one convert-pgn fixture chunk[cite: 1]:
    1. `len == sum` of the sidecars' `samples`[cite: 1].
    2. Item `i` equals `ReplayBuffer(file)[local_i]` for every `i`[cite: 1].
    3. Every selfplay sample's policy sums to 1 ± 1e-5[cite: 1].
    4. `outcome_wdl` matches the sidecar game result from each sample's side to move[cite: 1]. Use single-game chunks (`--games 1`, two calls) so the result is unambiguous; Black-to-move samples come from odd plies[cite: 1].
    5. `__getitems__` equals stacking `__getitem__`[cite: 1].
    6. A `.bin` without `.json` and a stray `.tmp` are ignored[cite: 1].
    7. With weights `[0.25, 0.75]` over 4,000 draws, the source frequencies are within ±0.03[cite: 1].
    8. The same seed gives the same index sequence[cite: 1].
- [ ] **6.4 `training/train.py`: streaming trainer with WDL/Q target blending & validation.**
  - Files: `training/losses.py` (update for `q_ratio`), `training/train.py` (new, CLI, ≤ 150 lines), `training/train_loop.py` (new: `train_steps`, `evaluate`, ≤ 200 lines), `training/schedule.py` (new: warmup + cosine to 10% LR), `training/pyproject.toml` (add the new `py-modules`), `training/tests/test_train.py` (new)[cite: 1].
  - CLI:
    ~~~
    python training/train.py --source PATH[,PATH...][:WEIGHT] ... --out model.pt --steps N
      [--init model.pt] [--batch-size 512] [--lr 1e-3] [--weight-decay 1e-4] [--warmup-steps 100]
      [--q-ratio 0.0] [--steps-per-epoch 500] [--val DIR_OR_BIN] [--val-samples 4096]
      [--device auto|cpu|mps] [--seed 0] [--log-json train.json]
    ~~~
    Each `--source` is one sampling source made of comma-separated chunk dirs (complete chunks only) and/or `.bin` files[cite: 1]. `:WEIGHT` is the text after the last `:` if it parses as a float; the default is 1.0[cite: 1]. Sources are concatenated into one `ChunkDataset`, and their index ranges feed `WeightedSourceSampler`[cite: 1].
  - **WDL + Search Q Blending (`--q-ratio`, default `0.0` for supervised bootstrap, `0.3` in RL config):**
    In `losses.wdl_loss(wdl_logits, outcome_wdl, q_value=None, q_ratio=0.0)`: when `q_ratio > 0` and `q_value` is provided (clamped to `[-1.0, 1.0]`), construct soft WDL probabilities from `q` as `soft_q = [clamp(-q, 0, 1), 1.0 - abs(q), clamp(q, 0, 1)]` (`[Loss, Draw, Win]`), blend with `one_hot(outcome_wdl, 3)` via `target_probs = (1.0 - q_ratio) * one_hot + q_ratio * soft_q`, and compute soft cross-entropy `-(target_probs * log_softmax(wdl_logits, dim=-1)).sum(dim=-1).mean()`. When `q_ratio == 0.0`, behavior is identical to standard hard-label cross-entropy.
  - AdamW[cite: 1]. Loss = `losses.total_loss` (passing `q_value` and `q_ratio`)[cite: 1].
  - The data path is `ChunkDataset` + `WeightedSourceSampler(num_samples = steps * batch)` + `make_loader` (**`num_workers=0`, `pin_memory=False`**)[cite: 1].
  - At the end of every `steps_per_epoch` block: run validation (if given), call **`device.empty_cache(device)`** (i.e. `torch.mps.empty_cache()`), and append a JSON-lines log record[cite: 1].
  - Writes the `.pt` via `save_checkpoint` atomically (`.tmp` + rename), then writes `train.json`: `{"steps","samples_seen","seconds","final_loss","policy_loss","wdl_loss","val":{"policy_top1","policy_loss","wdl_loss"}|null,"sources":[…]}`[cite: 1].
  - Seeds: `torch.manual_seed(seed)` and the sampler seed[cite: 1].
  - Test (CPU)[cite: 1]:
    1. On the 60-sample fixture chunk with `--steps 60 --batch-size 16 --lr 3e-3 --q-ratio 0.3 --device cpu --seed 0`, the mean loss of the last 10 steps is ≤ 0.7× that of the first 10[cite: 1].
    2. Unit test on `wdl_loss`: verify `q_ratio=0.0` matches `F.cross_entropy` exactly, and `q_ratio=1.0` with `q=0.0` puts 100% target mass on index `1` (`Draw`).
    3. The `.pt` loads with `load_checkpoint`[cite: 1].
    4. Two identical runs give identical `final_loss`[cite: 1].
    5. `--init` resumes from a given checkpoint[cite: 1].
    6. `val.policy_top1` is in `[0, 1]`[cite: 1].
  - RAM: model 36 MB + AdamW state 72 MB + batch 512 dense policy 8.5 MB + activations[cite: 1]. Expected peak RSS ≤ 2 GB on MPS at batch 512[cite: 1].
- [ ] **6.5 Run layout + supervised bootstrap script (generation 0).**
  - Files: `pipeline/pyproject.toml` (new: `[tool.pytest.ini_options] pythonpath=["."] testpaths=["tests"]`), `pipeline/run_layout.py` (new: path helpers for SPEC §10 plus `read_run_json`/`write_run_json` with atomic tmp+rename), `pipeline/bootstrap.py` (new), `pipeline/tests/conftest.py` (new: `engine_bin` fixture, repo paths), `pipeline/tests/test_bootstrap.py` (new)[cite: 1].
  - CLI:
    ~~~
    python3 pipeline/bootstrap.py --pgn FILE.pgn|FILE.pgn.zst --run DIR --config CFG.json
      [--val-games 2000] [--max-samples 4194304] [--steps 30000] [--batch-size 512] [--device auto]
    ~~~
    For a `.zst` file, each convert step pipes `zstd -dc FILE` into `engine convert-pgn --input -` with `subprocess.Popen`, so the decompressed PGN never touches disk[cite: 1].
  - Steps (each skipped when its output already exists, so it can resume)[cite: 1]:
    1. `convert-pgn --max-games val_games` → `supervised_val/`[cite: 1]
    2. `convert-pgn --skip-games val_games --max-samples …` → `supervised/`[cite: 1]
    3. `train.py --source supervised --val supervised_val --q-ratio 0.0` → `gens/gen_0000/model.pt` + `train.json`[cite: 1]
    4. `export_onnx.py` → `model.onnx`[cite: 1]
    5. `decision.json` with `{"promoted":true,"reason":"bootstrap"}`[cite: 1]
    6. `run.json` with `best_gen=0`, `current_gen=1`, `stage="selfplay"`, `resign_threshold=-0.95`, and a copy of the config[cite: 1]
  - Test: run on the fixture PGN with `--val-games 1 --steps 20 --batch-size 16 --device cpu`[cite: 1]. Assert:
    1. Every file listed above exists[cite: 1].
    2. `run.json` matches[cite: 1].
    3. `engine uci --model gens/gen_0000/model.onnx --ep cpu` answers `go nodes 32` with a legal `bestmove`[cite: 1].
    4. A second invocation is a no-op: the `model.pt` mtime is unchanged[cite: 1].
- [ ] **GATE 6:** `cd engine && cargo test --release && cd .. && pytest training/tests/test_chunk_dataset.py training/tests/test_train.py && pytest pipeline/tests/test_bootstrap.py` exits with `0`[cite: 1].

## Phase 7: Search Throughput, Repetition & FPU (Batched MCTS, Tree Reuse, Parallel Self-Play)
- [ ] **7.1 Refactor search into a resumable `SearchTree` + In-Tree Repetition + FPU.**
  - Files: `engine/src/mcts/tree.rs` (new), `engine/src/mcts/search.rs` (becomes a thin driver), `engine/src/mcts/mod.rs`, `engine/tests/mcts_golden.rs` (new)[cite: 1].
  - API:
    - `SearchTree::new(root: Board, history_hashes: &[u64], capacity: usize) -> Self` (or `with_history`)[cite: 1]
    - `select_leaf(&mut self) -> Leaf` (`Leaf::Terminal(v)` or `Leaf::NeedsEval { node, board }`)[cite: 1]
    - `expand_and_backup(&mut self, node, priors: &[(Move, f32)], value: f32)`[cite: 1]
    - `backup_terminal(&mut self, path, v)`[cite: 1]
    - `apply_root_noise(&RootNoise, &mut dyn RngCore)`[cite: 1]
    - `result(&self) -> SearchResult`[cite: 1]
  - **In-Tree Repetition Detection (moved forward from 10.3 so Self-Play & Match are repetition-aware):**
    - Track Zobrist hashes from game history (`history_hashes`) plus the active trajectory down the tree. Any non-root child node whose `board.hash()` appears in `history_hashes` or earlier in the current search path is immediately marked as a terminal draw (`Leaf::Terminal(0.0)`, `ChildStats.terminal = Some(0.0)`)[cite: 1].
  - **First Play Urgency (FPU Reduction):**
    - During PUCT child selection at node `s`: if a child `a` has `visits == 0`, instead of `Q(s, a) = 0.0`, compute `fpu_q = if is_root { 0.0 } else { (parent_q_from_s_perspective - 0.25).clamp(-1.0, 1.0) }` where `parent_q_from_s_perspective = -parent_edge_q` (since `parent_edge` was scored from `s`'s parent's side to move).
  - Test:
    1. Determinism test (`engine/tests/mcts_golden.rs`): record root child `(uci, visits)` lists for startpos, Kiwipete, and the scholar's-mate FEN with `UniformEvaluator`, `nodes 300`, and noise seed 1[cite: 1]. Two runs must match identically[cite: 1].
    2. Repetition test: in `7k/8/8/8/8/8/8/K5Q1 w - - 0 1` after moves `g1g2 h8h7 g2g1 h7h8` (passed as `history_hashes`), searching `nodes 200` marks root child `g1g2` with `terminal == Some(0.0)` and avoids picking `g1g2` when winning alternatives exist[cite: 1].
    3. `engine/tests/mcts.rs` and `GATE 4` must stay green[cite: 1].
- [ ] **7.2 Virtual-loss batched leaf evaluation within one tree (SPEC §8).**
  - Files: `engine/src/mcts/tree.rs`, `engine/src/mcts/batch.rs` (new: gather/collision/undo logic), `engine/src/mcts/search.rs` (`SearchLimits.batch: usize`, default 1), `engine/src/nn/mod.rs` (add `evaluate_boards(eval, &[Board]) -> Result<Vec<(Vec<(Move,f32)>, f32)>>`, one `evaluate` call), `engine/tests/mcts_batch.rs` (new)[cite: 1].
  - **Virtual Loss Sign Rule:** When a trajectory selects a leaf for batch evaluation, apply virtual loss along every edge on the path (`visits += 1`, `value_sum -= 1.0` from the choosing node's perspective so `Q` temporarily drops and subsequent in-batch selections diversify). When `expand_and_backup` (or collision/terminal undo) runs, remove the virtual loss (`visits -= 1`, `value_sum += 1.0`) before applying the real backup (`visits += 1`, `value_sum += v`).
  - Test:
    1. `batch = 1` still equals the 7.1 golden lists[cite: 1].
    2. For `batch ∈ {4, 8, 16}` with `nodes 400`: root visits == simulations; every expanded node satisfies `visits == 1 + Σ child visits` (or `== Σ` for the root); no leftover virtual loss remains (`|value_sum| <= visits` on every edge); `arena.len() <= 65,536`; and the 3 GATE-4 mate-in-1 FENs are still solved[cite: 1].
    3. A counting evaluator shows `avg batch ≥ 6` at `batch 8` from startpos[cite: 1].
- [ ] **7.3 Tree reuse (subtree compaction).**
  - Files: `engine/src/mcts/reuse.rs` (new: `SearchTree::advance(&mut self, mv: Move, new_board: Board, history_hashes: &[u64]) -> bool` copies the child's subtree BFS into a fresh arena, ≤ 65,536 nodes), `engine/src/mcts/tree.rs`, `engine/src/selfplay/game.rs` (reuse + §7 noise-on-reused-root + `--sims` = new playouts), `engine/src/cli/selfplay.rs` (`--no-reuse`), `engine/tests/reuse.rs` (new)[cite: 1].
  - Test:
    1. After `advance(best, ...)`, the new root's visits equal the old child's visits[cite: 1].
    2. Arena length equals the subtree size, and every child index is valid[cite: 1].
    3. A move outside the tree returns `false` and resets[cite: 1].
    4. Selfplay with reuse (seed 9, `--sims 16`, 1 game, `--max-plies 30`) is deterministic across two runs, and every sample's policy sums to 1 ± 1e-5[cite: 1].
    5. Root visits after a reused search are `≥ sims`[cite: 1].
  - RAM: 2 arenas briefly during compaction (≈ 3.6 MB)[cite: 1].
- [ ] **7.4 Lock-step multi-game self-play (`--parallel P`) + Temperature Schedule.**
  - Files: `engine/src/selfplay/driver.rs` (new: `GameDriver` = one game's board, history, tree, RNG and plies; `step_request()` / `step_apply()`), `engine/src/selfplay/batch_runner.rs` (new: runs `P` drivers, one `evaluate` call per step over all pending leaves, writes finished games in index order), `engine/src/selfplay/game.rs` (`play_game` = `P=1` runner), `engine/src/cli/selfplay.rs` (`--parallel P` default 16, `--leaf-batch K` default 1, `--temp-plies T` default 30), `engine/tests/selfplay_parallel.rs` (new)[cite: 1].
  - **Move Selection Temperature Schedule:**
    - Regardless of temperature, the **training policy target** written to `TrainingSample.policy` is always the normalized root visit counts `N(r, a) / Σ_b N(r, b)`.
    - For the **move actually played on the board**: for ply `< temp_plies` (default `30`), sample proportionally to visit counts (`T = 1.0`). For ply `>= temp_plies`, play greedily (`T -> 0`: pick the child with maximum `visits`, breaking ties by highest prior `P` then lowest canonical move index).
  - Test:
    1. With `UniformEvaluator`, 4 games, `--sims 16 --max-plies 40 --temp-plies 10 --seed 11`, `--parallel 4` and `--parallel 1` produce **byte-identical** `.bin` and identical sidecar `results`[cite: 1].
    2. With `--temp-plies 0`, the played move at every ply equals the argmax of the saved `TrainingSample.policy`.
  - RAM: `P = 32` → 32 trees × ≤ 3.6 MB + ORT batch I/O (32 × 4168 × 4 B ≈ 0.5 MB) ≈ 120 MB[cite: 1]. The selfplay process stays ≤ 600 MB including the ORT/CoreML runtime[cite: 1].
- [ ] **7.5 Resignation with playthrough audit (SPEC §7).**
  - Files: `engine/src/selfplay/resign.rs` (new: `ResignTracker { threshold, consecutive, playthrough }`; `update(stm: Color, root_q: f32) -> Option<Color>` returns the resigner; `would_resign` recorded in playthrough games), `engine/src/selfplay/driver.rs`, `engine/src/selfplay/stats.rs`, `engine/src/cli/selfplay.rs` (`--resign-threshold -0.95 --resign-consecutive 3 --resign-playthrough 0.1`), `engine/tests/resign.rs` (new)[cite: 1].
  - Test (pure logic)[cite: 1]:
    1. With threshold −0.9 and 3 consecutive, the sequence White q = [−0.95, −0.95, −0.5, −0.95, −0.95, −0.95] resigns only on the 6th White move[cite: 1].
    2. A playthrough tracker never resigns, but records `would_resign = White` at that point[cite: 1].
    3. The false-positive counter increments iff the game result is not a White loss[cite: 1].
    4. Integration: uniform selfplay with `--resign-threshold -1.0` has `terminations.resign == 0` and a sidecar `resign` block with the configured values[cite: 1].
- [ ] **7.6 `engine bench` + regression floor.**
  - Files: `engine/src/cli/bench.rs` (new), `engine/src/cli/mod.rs`, `engine/src/main.rs`, `pipeline/bench_gate.py` (new, stdlib)[cite: 1].
  - CLI: `engine bench [--model <onnx>|uniform] [--sims 800] [--batch 8] [--positions 6] [--ep ..] [--threads ..]` searches 6 fixed FENs (startpos, Kiwipete, 4 middlegames listed in the file) and prints the SPEC §8 JSON line[cite: 1].
  - `bench_gate.py` builds the release binary, runs `bench --model uniform --sims 800 --batch 1` and `--batch 8`, and asserts[cite: 1]:
    1. `sims_per_sec ≥ 50,000` at batch 1 (the M1 measures about 1M/s; this floor catches accidental O(n²) regressions on slow CI CPUs)[cite: 1].
    2. `avg_batch ≥ 6.0` at batch 8[cite: 1].
    3. `sims == 6 * 800`[cite: 1].
  - In the task's PR description, record the M1 numbers for a real 6x64 ONNX at batch 1/8/32 under `--ep coreml` and `--ep cpu`[cite: 1]. The faster one becomes the default `ep` in `pipeline/configs/m1_overnight.json` (9.1)[cite: 1].
- [ ] **GATE 7:** `cd engine && cargo test --release && cd .. && python3 pipeline/bench_gate.py` exits with `0`[cite: 1].

## Phase 8: Evaluation, Baselines & Elo Math
- [ ] **8.1 Players, material evaluator & opening book.**
  - Files: `engine/src/eval/mod.rs` (new), `engine/src/eval/material.rs` (new: `MaterialEvaluator: Evaluator` per SPEC §9, reading planes 0..11 of the already-encoded tensor), `engine/src/eval/players.rs` (new: `enum Player { Mcts { eval, tree, batch, sims }, Random { rng } }` with `choose(&mut self, board, history) -> Result<Move>` using greedy argmax visit selection (`T = 0`) and `observe(mv)` for tree reuse; `Player::from_spec(spec, sims, batch, opts, seed)`), `engine/src/eval/openings.rs` (new: `include_str!("../../data/openings.txt")`, `fn openings() -> Vec<Vec<Move>>` validated against legality), `engine/data/openings.txt` (new: 50 balanced lines, 2–8 plies, UCI), `engine/src/lib.rs`, `engine/tests/eval_players.rs` (new)[cite: 1].
  - Test:
    1. Material `v == 0` at startpos and `v > 0.8` for White to move a queen up (`4k3/8/8/8/8/8/8/3QK3 w - - 0 1`)[cite: 1].
    2. Material MCTS (`sims 64`) captures a hanging queen in `4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1` (`e4d5`)[cite: 1].
    3. All 50 openings parse and are legal, with no duplicates[cite: 1].
    4. `Random` with the same seed picks the same moves[cite: 1].
- [ ] **8.2 `engine match` (SPEC §9).**
  - Files: `engine/src/eval/match_runner.rs` (new: plays one game pair at a time sequentially; per-player in-tree batching via `--batch`, passing game history hashes so MCTS avoids 3-fold repetitions when winning), `engine/src/eval/adjudicate.rs` (new: §7 termination + ply-cap material adjudication), `engine/src/cli/match_cmd.rs` (new: flags, `match.json` via serde, optional `--moves-out`), `engine/src/cli/mod.rs`, `engine/src/main.rs`, `engine/tests/match_cmd.rs` (new)[cite: 1].
  - Test:
    1. `material` vs `random`, 20 games, `--sims 32 --seed 0 --max-plies 200`: `a_score ≥ 0.75`; `a_wins + b_wins + draws == 20`; games alternate `white: a/b`; openings are `offset + k`[cite: 1].
    2. Re-running with the same args gives a byte-identical `match.json`[cite: 1].
    3. `uniform` vs `uniform`, 4 games: `a_score == 0.5` (colour-symmetric pairs)[cite: 1].
    4. An odd `--games` is rejected[cite: 1].
  - RAM: 2 trees + 2 evaluators ≈ 200 MB with two 6x64 ONNX sessions[cite: 1].
- [ ] **8.3 Elo / SPRT math.**
  - Files: `pipeline/elo.py` (new, stdlib), `pipeline/tests/test_elo.py` (new)[cite: 1].
  - Interface:
    - `score(w, d, l) -> float`[cite: 1]
    - `elo(w, d, l) -> float`[cite: 1]
    - `elo_ci95(w, d, l) -> tuple[float, float]`[cite: 1]
    - `llr(w, d, l, elo0, elo1) -> float`[cite: 1]
    - `sprt_bounds(alpha, beta) -> tuple[float, float]`[cite: 1]
    - `decide(w, d, l, games_played, max_games, elo0, elo1, alpha, beta) -> tuple[str | None, bool]`, returning `(reason, promoted)` with `reason=None` meaning "continue"[cite: 1]
    - all exactly per SPEC §9/§10[cite: 1]
  - Test:
    1. `elo(60, 0, 40) ≈ 70.4 ± 0.1`[cite: 1].
    2. `elo(50, 0, 50) == 0`[cite: 1].
    3. `elo(10, 0, 0)` is finite (clamping)[cite: 1].
    4. `sprt_bounds(0.05, 0.05) ≈ (−2.944, 2.944)`[cite: 1].
    5. `llr` is negative at score 0.5 and positive at 0.65 with 200 games[cite: 1].
    6. `decide` returns `sprt_accept` for (90, 20, 10), `sprt_reject` for (20, 20, 80), and `max_games` with `promoted == (llr > 0)` at the cap[cite: 1].
- [ ] **8.4 Block-wise evaluation driver.**
  - Files: `pipeline/evaluate.py` (new, stdlib), `pipeline/tests/test_evaluate.py` (new)[cite: 1].
  - CLI:
    ~~~
    python3 pipeline/evaluate.py --engine BIN --a SPEC --b SPEC --out-dir DIR --decision PATH
      --sims S --block-games 40 --max-games 400 --seed-base N
      [--elo0 0 --elo1 40 --alpha 0.05 --beta 0.05 --batch 8 --max-plies 300 --ep auto --threads 1]
    ~~~
  - For block `b = 0, 1, …`, skip it if `DIR/block_BB.json` already exists[cite: 1]. Otherwise run `engine match … --opening-offset 20*b --seed seed_base+b --out DIR/block_BB.json.tmp` and rename[cite: 1]. After each block, call `elo.decide` on the cumulative totals and stop when it decides[cite: 1]. Write `decision.json` (SPEC §10 schema) atomically[cite: 1].
  - Test:
    1. `material` vs `random` with `--block-games 10 --max-games 40 --sims 16`: promoted, reason `sprt_accept`, `games ≤ 40`[cite: 1].
    2. Re-run: no new blocks and block mtimes unchanged[cite: 1].
    3. `uniform` vs `uniform` with `--max-games 20`: not promoted[cite: 1].
    4. Deleting `decision.json` and re-running reproduces it from the existing blocks without running the engine (check that the call count via a fake `--engine` wrapper script is 0)[cite: 1].
- [ ] **GATE 8:** `cd engine && cargo test --release && cd .. && pytest pipeline/tests/test_elo.py pipeline/tests/test_evaluate.py` exits with `0`[cite: 1].

## Phase 9: Closed, Resumable RL Loop
- [ ] **9.1 Config + replay window.**
  - Files: `pipeline/config.py` (new: `load_config(path) -> dict`, deep-merged over SPEC §10 defaults including `train.q_ratio: 0.3` and `selfplay.temp_plies: 30`, and validated: `games_per_chunk ≤ 1000`, `block_games` even, …), `pipeline/configs/m1_overnight.json` (new: the SPEC §10 example with `q_ratio: 0.3`, `ep` from the 7.6 bench), `pipeline/configs/ci_tiny.json` (new: `selfplay.games 4, games_per_chunk 2, sims 16, parallel 4, max_plies 40, temp_plies 10, ep "cpu"`; `train.steps 20, batch_size 16, q_ratio 0.3, device cpu, steps_per_epoch 10`; `match.block_games 4, max_games 8, sims 16`; `anchors.games 2`; `window.max_samples 10000`), `pipeline/window.py` (new: `select_window(run_dir, gen, cfg) -> list[tuple[list[Path], float]]`, returning `(chunk paths, weight)` per source per SPEC §10; `train_args(...) -> list[str]` building one `--source a.bin,b.bin,...:W` flag per source), `pipeline/tests/test_window.py` (new)[cite: 1].
  - Test: build a fake run tree with sidecars and zero-filled `.bin` files of the correct size (no engine needed)[cite: 1]. Assert:
    1. Gen 6 with `W = 4` uses gens 3–6 only, newest-first[cite: 1].
    2. It stops adding chunks once `max_samples` is reached[cite: 1].
    3. Incomplete chunks are excluded[cite: 1].
    4. The supervised weight is 0.5 at gen 1 and 0.0 at gen ≥ 5[cite: 1].
    5. A zero-weight source is omitted[cite: 1].
- [ ] **9.2 Idempotent stages.**
  - Files: `pipeline/stages.py` (new; one function per stage `selfplay | train | export | match | decide | anchors`, each `(run_dir, gen, cfg, state) -> None`, each skips completed work per the SPEC §10 resume rule and uses `subprocess.run(check=True)`), `pipeline/tests/test_stages.py` (new)[cite: 1].
  - Details:
    - The selfplay stage calls `engine selfplay` once per missing chunk, with `--seed 1_000_000*gen + 1000*idx + cfg.seed`, `--temp-plies cfg.selfplay.temp_plies`, and `--model <best onnx>`[cite: 1]. After the last chunk it updates `resign_threshold` per SPEC §10[cite: 1].
    - The train stage passes `--q-ratio cfg.train.q_ratio` (default `0.3`) and `--init <best model.pt>` to `training/train.py`.
    - The anchors stage runs `engine match --a <new best> --b random|material --games anchors.games`[cite: 1].
  - Test (session-scoped fixture: a `ci_tiny` bootstrap from 6.5): run each stage for gen 1 in order and assert its outputs[cite: 1]. Then:
    1. Delete one selfplay chunk's `.json`; re-running the selfplay stage regenerates only that chunk (other mtimes unchanged)[cite: 1].
    2. A stray `model.pt.tmp` is removed and `train` re-runs[cite: 1].
- [ ] **9.3 `pipeline/loop.py` orchestrator with lock + crash-resume.**
  - Files: `pipeline/loop.py` (new), `pipeline/tests/test_loop.py` (new)[cite: 1].
  - CLI: `python3 pipeline/loop.py --run DIR --engine BIN [--generations N] [--until-gen G]`[cite: 1]. The config comes from `run.json` (written by bootstrap)[cite: 1]. The loop advances `stage` through SPEC §10 and saves `run.json` after every stage[cite: 1]. It takes `loop.lock` (refuses if the PID inside is alive and removes a stale one)[cite: 1]. On SIGINT, it terminates the running child process, leaves `run.json` at the last completed stage, removes the lock, and exits 130[cite: 1]. Every stage's work is already resumable[cite: 1].
  - Test hook: env `CHESSBOT_CRASH_AT=<gen>:<stage>` makes the loop `os._exit(17)` right *after* that stage's work and *before* saving `run.json`[cite: 1]. This simulates the worst-case crash[cite: 1].
  - Test (ci_tiny)[cite: 1]:
    1. Run with `CHESSBOT_CRASH_AT=1:train` and expect exit code 17[cite: 1].
    2. Rerun without it, `--until-gen 2`: exit 0, `current_gen == 3`, `stage == "selfplay"`, gens 1–2 each have `decision.json`, and the selfplay chunk count is exactly 2 per gen[cite: 1].
    3. No `*.tmp` files remain anywhere[cite: 1].
    4. `best_gen` equals the last gen with `"promoted": true`[cite: 1].
    5. A second concurrent start (lock held by a live dummy PID) exits non-zero[cite: 1].
- [ ] **9.4 Disk-budget pruning.**
  - Files: `pipeline/prune.py` (new: `prune(run_dir, state, cfg) -> list[Path]` removed; also a CLI `--dry-run`), `pipeline/stages.py` (add the `prune` stage), `pipeline/tests/test_prune.py` (new)[cite: 1].
  - Test (fake run tree, sized files)[cite: 1]:
    1. Selfplay dirs older than the window are removed[cite: 1].
    2. A rejected gen loses `model.pt`/`model.onnx` but keeps its JSONs[cite: 1].
    3. With the budget set below the current size, the oldest promoted `.pt` goes first, and the best and current gens are never touched[cite: 1].
    4. If the budget is still unreachable, it raises `DiskBudgetError`[cite: 1].
- [ ] **9.5 Ladder + report.**
  - Files: `pipeline/report.py` (new: `build_ladder(run_dir) -> list[dict]` and `write_report(run_dir)` writing `ladder.json` and `report.tsv` with columns `gen promoted reason games score elo elo_lo elo_hi ladder_elo vs_random vs_material train_loss val_top1 selfplay_samples selfplay_secs train_secs match_secs`; CLI `--run DIR` prints the TSV), `pipeline/loop.py` (call `write_report` after `prune`), `pipeline/tests/test_report.py` (new)[cite: 1].
  - Test: a fake run with gen 0 (bootstrap), gen 1 promoted at +50, gen 2 rejected, gen 3 promoted at +30 gives ladder Elo `[0, 50, 80]` for gens `[0, 1, 3]`[cite: 1]. Every TSV row has the same column count[cite: 1]. Missing anchor files give `NA`[cite: 1].
- [ ] **GATE 9:** `pytest pipeline/tests/test_window.py pipeline/tests/test_stages.py pipeline/tests/test_loop.py pipeline/tests/test_prune.py pipeline/tests/test_report.py` exits with `0`[cite: 1].

## Phase 10: Play Strength & UCI Polish
- [ ] **10.1 Time management.**
  - Files: `engine/src/uci/time_mgmt.rs` (new: `fn budget(stm: Color, clock: &Clock, overhead_ms: u64) -> Option<Duration>` per SPEC §11), `engine/src/uci/uci_loop.rs` (`parse_go` accepts `wtime btime winc binc movestogo infinite`), `engine/tests/time_mgmt.rs` (new)[cite: 1].
  - Test:
    1. `wtime 60000 winc 1000`, White → `60000/30 + 750 − 50 = 2700 ms`[cite: 1].
    2. `movestogo 10 wtime 10000` → `950 ms`[cite: 1].
    3. `wtime 100` → floor `10 ms`; the cap `rem/4` applies[cite: 1].
    4. Black uses `btime/binc`[cite: 1].
    5. `movetime` overrides[cite: 1].
- [ ] **10.2 Async search: `stop`, `isready` during search, periodic `info`.**
  - Files: `engine/src/uci/search_thread.rs` (new: worker owns the tree + evaluator; `Arc<AtomicBool>` stop flag; channel back to the main thread; always selects `bestmove` via greedy max-visit `T = 0`), `engine/src/uci/uci_loop.rs` (stdin reader stays on the main thread), `engine/src/mcts/search.rs` (`SearchLimits.stop: Option<Arc<AtomicBool>>` and an `on_info` callback every 500 ms), `engine/tests/uci_async.rs` (new; spawns `env!("CARGO_BIN_EXE_engine")`)[cite: 1].
  - Test:
    1. `go infinite` → `isready` → `readyok` within 200 ms[cite: 1].
    2. `stop` after 300 ms → `bestmove` within 100 ms (allow 500 ms on CI)[cite: 1].
    3. At least 1 `info nodes` line before `bestmove`[cite: 1].
    4. `go nodes 400` still returns by itself[cite: 1].
    5. GATE 4 still passes[cite: 1].
- [ ] **10.3 `setoption` + reuse-aware, repetition-aware UCI search.**
  - Files: `engine/src/uci/options.rs` (new: SPEC §11 options; `ModelPath` reloads the evaluator, `Threads`, `BatchSize`, `MoveOverhead`), `engine/src/uci/uci_loop.rs`, `engine/src/uci/search_thread.rs` (passes `position ... moves ...` history hashes into `SearchTree` and reuses subtree via `advance` when `position` extends the previous one by 1–2 plies), `engine/tests/uci_options.rs` (new)[cite: 1].
  - Test:
    1. `uci` lists exactly the 4 options[cite: 1].
    2. `setoption name BatchSize value 16` takes effect (the bench-style counter shows avg batch > 8)[cite: 1].
    3. A bad `ModelPath` gives `info string error` and keeps the old net[cite: 1].
    4. In `position fen 7k/8/8/8/8/8/8/K5Q1 w - - 0 1 moves g1g2 h8h7 g2g1 h7h8`, searching via UCI with that history (`go nodes 200`) marks the root child `g1g2` as a terminal draw (`ChildStats.terminal == Some(0.0)`) and avoids playing `bestmove g1g2`[cite: 1].
    5. After `position startpos moves e2e4` following a `go` on startpos, the engine prints `info string reused N nodes` with `N > 0`[cite: 1].
- [ ] **10.4 UCI client + clocked protocol test.**
  - Files: `pipeline/uci_client.py` (new: the `Engine` class extracted from `test_uci_mate.py`, plus `go_clock()` timing helpers), `pipeline/test_uci_mate.py` (import from `uci_client`; behavior unchanged), `pipeline/test_uci_protocol.py` (new, stdlib)[cite: 1].
  - The protocol test plays a referee-run game between two engine processes (uniform net, `BatchSize 8`) at `wtime/btime 3000 winc/binc 50`, up to 120 plies or game end[cite: 1]. It asserts:
    1. No side's measured think time exceeds its remaining clock[cite: 1].
    2. No `info string error` appears[cite: 1].
    3. Each `bestmove` is a well-formed UCI move that the opponent engine accepts in `position … moves`[cite: 1].
    4. `go infinite` + `stop` returns within 500 ms[cite: 1].
  - If `cutechess-cli` is on `PATH`, it also runs `cutechess-cli -engine cmd=engine/target/release/engine -engine cmd=… -each proto=uci tc=2+0.05 -rounds 1 -games 2` and asserts no "illegal move"/"disconnects"/"time forfeit"[cite: 1]. Otherwise it prints `SKIP cutechess-cli not found` (still exit 0)[cite: 1].
- [ ] **10.5 Absolute Elo calibration vs Stockfish (optional at runtime).**
  - Files: `pipeline/calibrate.py` (new), `pipeline/tests/test_calibrate.py` (new)[cite: 1].
  - CLI: `python3 pipeline/calibrate.py --run DIR --gen G --stockfish PATH --levels 1320,1600,1900 --games 40 --tc 10+0.1 [--ep auto]` plays our engine against Stockfish with `UCI_LimitStrength true` and `UCI_Elo L` via `uci_client`, alternating colours over the 8.1 opening book (passed as `position startpos moves …`)[cite: 1]. It writes `gens/gen_GGGG/calibration.json`: `{"levels":[{"elo":1320,"w":..,"d":..,"l":..}],"estimate":1480.0,"ci95":[1410,1550]}` using a 1-D maximum-likelihood fit of our Elo given the logistic expected score vs each level[cite: 1].
  - Test: the MLE alone, no engines[cite: 1]. Synthetic results drawn from a true Elo of 1500 (40 games per level at 1320/1600/1900, exact expected scores) estimate within ±5[cite: 1]. All wins clamps to `> max level` without crashing[cite: 1].
- [ ] **GATE 10:** `cd engine && cargo test --release && cd .. && python3 pipeline/test_uci_mate.py && python3 pipeline/test_uci_protocol.py && pytest pipeline/tests/test_calibrate.py` exits with `0`[cite: 1].

## Milestones (real M1 runs; owner-driven, not cloud-gated)
Each is checked off when its acceptance command shows the stated numbers[cite: 1]. Use `~/.venvs/chessbot/bin/python`[cite: 1].
- [ ] **M1 – Generation 0 (after GATE 8):**
  - Command: `python pipeline/bootstrap.py --pgn lichess_elite_YYYY-MM.pgn.zst --run data_store/runs/main --config pipeline/configs/m1_overnight.json` (unzip the Lichess Elite `.zip` first if needed; plain `.pgn` also works), then `engine match` anchors vs `random` and `material`[cite: 1].
  - Accept when:
    1. `gens/gen_0000/train.json` shows `val.policy_top1 ≥ 0.35` and `val.wdl_loss ≤ 1.0`[cite: 1].
    2. The anchor scores are `vs_random ≥ 0.95` and `vs_material ≥ 0.70` (20 games, 200 sims)[cite: 1].
    3. `/usr/bin/time -l` shows peak RSS ≤ 2.5 GB[cite: 1].
- [ ] **M2 – First overnight generation (after GATE 9):**
  - Command: `python pipeline/loop.py --run data_store/runs/main --engine engine/target/release/engine --generations 1`[cite: 1].
  - Accept when:
    1. `report.tsv` shows `selfplay_secs + train_secs + match_secs ≤ 8 h`[cite: 1].
    2. Peak RSS of every stage is ≤ 3.5 GB[cite: 1].
    3. Interrupting with Ctrl-C mid-selfplay and restarting finishes the generation without regenerating complete chunks[cite: 1].
- [ ] **M3 – Self-improvement:** at least 3 promoted generations after gen 0[cite: 1]. `ladder.json` Elo is strictly increasing over promoted gens[cite: 1]. A direct 200-game `evaluate.py` match of the latest best vs gen 0 has `elo_ci95[0] > 0`[cite: 1].
- [ ] **M4 – Calibrated strength (after GATE 10):** `calibrate.py` on the latest best writes `calibration.json`, and the engine completes a 20-game `cutechess-cli` match vs `stockfish UCI_Elo 1320` with no illegal moves or time forfeits[cite: 1]. Report the Elo estimate in `report.tsv`'s notes[cite: 1].
