# Resume Generation 0 after an export parity failure

The training checkpoint and `train.json` are saved before ONNX export starts.
An export error does not require another training run. Use the updated exporter
and bootstrap's `--export-only` option from the repository root, with the
chessbot Python environment active:

```bash
python -m pip install -e training
set -o pipefail
pgn_path="$(python -c 'import json; from pathlib import Path; print(json.loads(Path("data_store/runs/main/supervised/.complete.json").read_text())["input"]["pgn"])')"
python -u pipeline/bootstrap.py \
  --pgn "$pgn_path" \
  --run data_store/runs/main \
  --config data_store/bootstrap.json \
  --export-only \
  2>&1 | tee /tmp/chessbot-export-retry.log
```

This reads the original PGN path from the conversion manifest. Keep the original
config file. Export-only recovery verifies the completed chunk manifests and
checkpoint/summary, then exports and validates the saved model. It never calls
conversion or training, including when input PGN timestamps have changed. Missing
or incomplete required artifacts stop recovery without deleting the checkpoint.
It can run without a built Rust binary because it only reads existing Rust data.
If a prior validated ONNX model is already published, a failed retry preserves
that model and its published bootstrap state.

Success ends with `Generation 0 ready: .../gens/gen_0000`. The finalized
`model.onnx`, bootstrap `decision.json`, and `run.json` are then available. A failed
parity check still exits nonzero and prevents bootstrap publication; do not rename
an unverified `.onnx.tmp` file yourself.

## What the new validation checks

The old random-input check rejected a trained M1 model at a maximum absolute
logit difference of `1.83e-4`, on policy logits whose magnitude reached about 264.
The owner then measured 64 real held-out positions: raw errors passed `1e-4` at
batches 1, 7 and 32, probability differences stayed below `4.5e-6`, and all legal
move argmax choices agreed. This supports a scale-sensitive float32 discrepancy;
it does not establish that the original random-input check passed.

The updated contract is explicit in SPEC §4:

- Seeded random stress inputs keep batches 1 and 7. Each raw logit must satisfy
  `abs(onnx - reference) <= 1e-4 + 1e-6 * abs(reference)`. A large logit elsewhere
  cannot hide a mismatch near zero.
- Bootstrap also checks up to 64 evenly spaced held-out Rust-encoded positions
  at batches 1, 7 and 32. Selected positions repeat to fill small/partial batches.
  These raw checks retain strict absolute `1e-4`, without a relative allowance.
- Both paths independently require softmax probability differences at most
  `1e-5` in both heads. Real policy probabilities use the stored legal-move mask.
- Nonfinite outputs, shape mismatches, empty sources, and invalid masks fail.

These are additional validation rules, not a claim that the original failure was
harmless on every possible input. Normal export on the owner's saved M1 model
still needs to pass. No checkpoint weights, architecture, tensor encoding or
runtime optimization settings change.

## Continue with the baseline matches

After successful export recovery, build the engine if needed and run:

```bash
cargo build --release --manifest-path engine/Cargo.toml
gen_dir=data_store/runs/main/gens/gen_0000
for opponent in random material; do
  /usr/bin/time -l engine/target/release/engine match \
    --a "$gen_dir/model.onnx" --b "$opponent" \
    --games 20 --sims 200 --seed 0 \
    --batch 1 --ep auto --threads 1 \
    --out "$gen_dir/anchors/vs_${opponent}.json" \
    2>&1 | tee "/tmp/chessbot-anchor-${opponent}.log"
done
```

Compare each match's `a_score` against `0.95` for random and `0.70` for material.
Generation 0 also needs final `val.policy_top1 >= 0.35`, `val.wdl_loss <= 1.0`,
and measured maximum RSS no greater than 2.5 GB. Export success alone does not
complete the milestone. Batch 1 follows the recorded M1 benchmark as a starting
choice; `--ep auto` may fall back to CPU.

## Preserve the completed run

On the Mac that contains the actual model and match files, run from the repo root:

```bash
python pipeline/archive_generation_zero.py
```

The command copies both models, the training summary, both anchor JSON files,
bootstrap decision/run state, and available logs into a new timestamped directory
under `~/Library/Application Support/chessbot/archives/`, outside OneDrive. Source
files are retained. Each copied file is checked with SHA-256; the archive's
`manifest.json` records the hashes, exact match results, available RSS values and
any missing optional logs. Missing required artifacts abort instead of claiming
a complete archive. Original tee logs may be incomplete because of earlier
OneDrive timeouts. Temporary logs under `/tmp` are copied if they still exist.

`training/GENERATION_0_RESULTS.json` records the owner's confirmed milestone
metrics. It is a results record, not a substitute for copying the original model
files; the cloud workspace cannot perform this Mac-local copy itself.
