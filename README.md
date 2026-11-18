# chessbot

chessbot is an AlphaZero-style chess engine and training pipeline designed for an 8 GB Apple M1. Rust runs neural-network-guided Monte Carlo tree search (MCTS) and generates training data. PyTorch trains a policy and win/draw/loss network, then exports it to ONNX for the Rust engine.

- **Search.** PUCT with first-play urgency, a 65,536-node arena per tree, virtual-loss batching, and subtree reuse in self-play and evaluation matches. Parallel self-play combines positions from multiple games in one inference call.
- **Rust–Python contract.** Rust canonicalizes positions to the side to move and writes 492-byte training records. Python memory-maps the records and decodes them into 18 input planes and targets over 4,168 move slots. Contract tests compare the data with an independent `python-chess` oracle.
- **Training and evaluation.** A dual-head SE-ResNet-6x64 learns from PGN games and self-play. The generation loop resumes at saved stages, reuses completed chunks, validates ONNX exports, and evaluates candidates in color-swapped opening pairs before promotion. It maintains a replay window and writes `ladder.json` and `report.tsv`.

## Run a smoke test

Use a C toolchain and a recent stable Rust toolchain (`ort` requires Rust 1.92 or newer). The Python tools require Python 3.11 or newer. Commands below run from the repository root.

```bash
python3 -m venv ../chessbot-venv
source ../chessbot-venv/bin/activate
```

On Linux, install the CPU build of PyTorch first.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install the training package and its dependencies.

```bash
python -m pip install -e training
```

On x86-64 Linux, the pinned `ort` build requests CoreML, for which the Linux prebuilt distribution has no matching binary. Link against the CPU library from the ONNX Runtime wheel before building. Keep these variables set for subsequent builds and runs.

```bash
python -m pip install 'onnxruntime==1.31.1'
export ORT_LIB_PATH="$VIRTUAL_ENV/ort-lib"
mkdir -p "$ORT_LIB_PATH"
ort_so=$(python -c 'from importlib.util import find_spec; from pathlib import Path; print(next((Path(find_spec("onnxruntime").origin).parent / "capi").glob("libonnxruntime.so.*")))')
ln -sfn "$ort_so" "$ORT_LIB_PATH/libonnxruntime.so"
ln -sfn "$ort_so" "$ORT_LIB_PATH/libonnxruntime.so.1"
export ORT_PREFER_DYNAMIC_LINK=1
export LD_LIBRARY_PATH="$ORT_LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Build the engine and ask it for a move. The uniform evaluator runs this demo without trained weights.

```bash
cargo build --locked --release --manifest-path engine/Cargo.toml
printf 'uci\nisready\nposition startpos\ngo nodes 64\nquit\n' | \
  engine/target/release/engine uci --model uniform
```

Expect `uciok`, `readyok`, and a `bestmove` line. To play with a trained checkpoint, pass its exported `.onnx` file to `uci --model`.

## Run a small training cycle

This CPU example uses the bundled PGN fixture and a temporary run directory. It creates a supervised Generation 0, runs one small self-play generation, and prints the report.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
run_dir=$(mktemp -d)
python pipeline/bootstrap.py \
  --pgn training/tests/fixtures/sample.pgn --run "$run_dir" \
  --config pipeline/configs/ci_tiny.json --val-games 1 --steps 1 \
  --batch-size 2 --device cpu
python pipeline/loop.py \
  --run "$run_dir" --engine engine/target/release/engine --until-gen 1
cat "$run_dir/report.tsv"
```

Use a new run directory when changing bootstrap training flags, and a fresh output directory for standalone `convert-pgn --out-dir` runs. For a larger bootstrap, supply a standard-chess `.pgn` or `.pgn.zst` and keep the input and config with the run metadata. Compressed input requires the `zstd` CLI. [Export recovery](training/EXPORT_RECOVERY.md)

## Current status

- **Generation 0.** The recorded M1 run used 30,000 supervised updates and achieved 41.43% policy top-1 accuracy on held-out human positions and WDL loss of 0.8882. A score of 1.0 was reported in each 20-game match against the random and material baselines at 200 simulations per move. The raw model and match files are held locally. [Results and provenance](training/GENERATION_0_RESULTS.json)
- **Self-play loop.** The training, promotion, pruning, and reporting stages are implemented and exercised by small CPU integration tests. Full-scale overnight timing, peak RSS per stage, and sustained improvement across promoted generations are open milestones. `selfplay_secs` and `match_secs` are currently unavailable in the report. [Roadmap and milestones](TASKS.md#milestones-real-m1-runs-owner-driven-not-cloud-gated)
- **Playing interface.** UCI supports synchronous `go nodes` and `go movetime`. Game-clock management, interruptible search, live search info, runtime options, and Stockfish calibration are Phase 10 work. [Technical contracts](SPEC.md#11-planned-uci-extensions-phase-10)

The ladder is relative to Generation 0. It accumulates estimated Elo differences from promotion matches, and each displayed interval reflects uncertainty for that match conditional on the incumbent's point estimate. The promotion rule uses a GSPRT approximation over game outcomes. The 50-opening book wraps in longer runs, so repeated opening pairs can contribute repeated evidence.

## Verification

```bash
cargo test --locked --release --manifest-path engine/Cargo.toml
python -m pytest -c training/pyproject.toml training/tests
python -m pytest -c pipeline/pyproject.toml pipeline/tests
python pipeline/test_uci_mate.py
python pipeline/bench_gate.py
```

One known fixture failure at `07e6dd2` is `test_single_game_sidecars_match_both_perspectives`. Both seeded games draw at the ply cap, so the fixture exercises draw labels only. The benchmark gate uses the uniform evaluator and measures search throughput and batch filling.

The walkthrough explains the algorithms. The specification records the data contracts. The roadmap tracks milestones and historical checks. [Walkthrough](docs/NOTEBOOKLM_OVERVIEW.md) · [Specification](SPEC.md) · [Roadmap](TASKS.md)
