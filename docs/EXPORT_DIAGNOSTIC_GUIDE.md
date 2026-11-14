# Diagnose and recover a generation-1 export

The current candidate completed 1,200 self-play games and 4,000 training steps.
Its CPU PyTorch-to-ONNX comparison failed before matches began. Work from that
saved candidate. This procedure does not require new games or training.

The guide has two phases: collect evidence first, then apply and verify a fix
chosen from that evidence. Completing a diagnostic is not approval to publish an
experimental model. Current numerical acceptance limits remain unchanged.

## 1. Stop at the saved export stage

Use one Terminal window, from the repository folder. If the loop is still
running, press Ctrl-C once and wait for its prompt. Do not run bootstrap recovery:
`bootstrap.py --export-only` is for generation 0.

After the diagnostic update is available on `codex/m2-run-analysis`, update it:

```bash
git switch codex/m2-run-analysis &&
git pull --ff-only origin codex/m2-run-analysis
```

The new command below is `training/export_diagnostics.py`. If it is absent, stop:
the installed revision does not yet include this guide's diagnostic support.

Set a new evidence directory outside OneDrive:

```bash
export EXPORT_PY="$HOME/.venvs/chessbot/bin/python"
export EXPORT_RUN="$PWD/data_store/runs/main"
export EXPORT_EVIDENCE="$HOME/Library/Logs/chessbot/export-gen1-$(date +%Y%m%d-%H%M%S)"
```

Use a new directory each time. The runner creates it and refuses to overwrite an
existing one. Keep the saved Python environment; changing package versions before
reproducing the problem would introduce another variable.

## 2. Run the bounded diagnostic

```bash
"$EXPORT_PY" training/export_diagnostics.py \
  --run "$EXPORT_RUN" \
  --out "$EXPORT_EVIDENCE" \
  --failing-row 2 \
  --timeout 60
```

The runner checks that the saved stage is `export`, the training summary shows
all configured steps, and all configured self-play chunks are complete. It uses
the loop lock throughout the procedure and refuses a live loop or worker.

Before loading the network, it copies the checkpoint and available metadata into
`backup/`, verifies their hashes, and saves `preservation.json`. The preservation
manifest fingerprints the checkpoint, training summary, available history and
generation metadata, run state, every complete self-play chunk pair and the
held-out parity source.

Five fresh worker processes run sequentially, with at most 60 seconds each:

| Profile | Exported graph | CPU execution |
| --- | --- | --- |
| `baseline` | Normal export | Original/default settings |
| `single` | Normal export | PyTorch and ONNX Runtime each use one thread |
| `no_ort` | Normal export | One thread; ONNX Runtime graph optimizations disabled |
| `no_fold` | Export folding disabled, if supported | One thread; normal ONNX Runtime optimizations |
| `no_fold_no_ort` | Export folding disabled, if supported | One thread; ONNX Runtime optimizations disabled |

Actual graph operation counts are recorded. A deprecated export flag may have no
effect in a particular PyTorch version; the graph must show whether the intended
change happened before that comparison is interpreted.

The five workers have a combined five-minute timeout budget. Hashing, backup and
report writing add setup time. Each worker writes its own unpublished ONNX file,
partial evidence and logs under the evidence directory. No worker writes the
loop's official `model.onnx` or changes its saved stage.

## 3. Understand the output before taking another action

Expected top-level output includes each profile's result, the path to
`report.json`, and `Protected artifacts: UNCHANGED`.

- `PARITY FAIL` means the comparison completed and failed a consistency check.
  The other comparisons still run. This reproduces evidence rather than losing
  it after the first error.
- `PASS` means that profile passed the existing gate cases. A passing experimental
  profile does not override a failing baseline.
- `INCOMPLETE` means an execution error, unsupported operation or timeout stopped
  that profile. Its partial records and stderr log remain available.

Exit status 0 means the baseline passed, profiles completed and protected files
were preserved. Exit status 1 means completed diagnostics reproduced baseline
parity failure. Exit status 2 means an operational or preservation problem;
inspect the report and logs. Ctrl-C exits 130 after cleanup.

The full report and preservation manifest are local:

```bash
ls "$EXPORT_EVIDENCE"
```

Keep `backup/model.pt`. Share the console summary and `report.json` for review;
the model weights do not need to be uploaded to interpret these measurements.
Stop here until the report has been assessed.

## 4. What the report must establish

The random inputs preserve the exporter's exact sequence: generator seed 0,
batch 1 first, then batch 7. Independently reseeding batch 7 would test different
inputs. Row 2 identifies the failing row in the reported generation-1 failure.
The same row is also tested alone and compared with its result inside batch 7,
within each runtime as well as between runtimes.

Each profile retains both output heads' numerical evidence even when the policy
raw-logit check fails. It also checks the existing held-out Rust-encoded positions
at batches 1, 7 and 32. Python reads those positions through the existing memmap
decoder; it does not encode new chess positions.

The required evidence includes raw errors and per-value violation counts,
probability differences, legal-move choices and their margins on real positions,
and outcome-probability/value differences. Random tensors have no legal chess
move mask. A near-tie change in preferred move is diagnostic information rather
than an additional automatic rejection rule.

Record checkpoint/input/ONNX hashes, package versions, CPU provider settings,
thread counts and graph operation counts. A numerical difference without this
context is insufficient to choose a fix.

## 5. Choose the fix from the controlled comparisons

| Evidence | Next investigation |
| --- | --- |
| Controlled one-thread cases pass but baseline fails | Isolate thread/backend settings individually |
| Disabling ONNX Runtime optimizations helps | Identify the responsible runtime optimization |
| Disabling export folding helps and the graph actually differs | Inspect the export transformation and its rounding effects |
| Only disabling both helps | Investigate both transformation stages |
| None helps | Use a supported PyTorch backend comparison or a higher-precision reference calculation |

Higher precision is an explanation tool, not a replacement acceptance reference.
Use the same saved float32 inputs converted to higher precision, and a separate
in-memory copy of the checkpoint. Keep the saved weights intact.

Prefer a narrowly demonstrated fix that works with the engine's intended runtime
settings. Passing an easier diagnostic configuration does not establish that a
different production configuration is valid. Do not tune tolerances to this one
checkpoint, omit batch 7, or substitute different stress inputs.

## 6. Verify the implementation before recovery

After an evidence-based fix is implemented, its tests must demonstrate that an
early numerical failure still produces a complete report, fails overall, and
never publishes a model. Recovery must preserve the trained checkpoint and
completed chunks and invoke neither self-play nor training again.

Run the applicable existing checks and the new diagnostic tests separately:

```bash
"$EXPORT_PY" -m pytest training/tests/test_onnx_parity.py \
  training/tests/test_real_parity.py training/tests/test_parity_diagnostics.py \
  training/tests/test_export_diagnostics.py

"$EXPORT_PY" -m pytest pipeline/tests/test_export_recovery.py \
  pipeline/tests/test_stage_recovery.py
```

The implementer must also verify the selected production runtime configuration,
including any Rust change, and rerun its relevant gates. Keep every Python/Rust
source file below the repository's 250-line limit.

## 7. Require a normal export and engine checks to pass

Only after review and implementation of the fix, create a new recovery folder:

```bash
export EXPORT_RECOVERY="$HOME/Library/Logs/chessbot/export-recovery-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$EXPORT_RECOVERY"
set -o pipefail

"$EXPORT_PY" training/export_onnx.py \
  --checkpoint "$EXPORT_RUN/gens/gen_0001/model.pt" \
  --out "$EXPORT_RECOVERY/checked.onnx" \
  --parity-source "$EXPORT_RUN/supervised_val" \
  2>&1 | tee "$EXPORT_RECOVERY/export.log"
```

Require exit status 0 and the normal `parity passed` message, including the real
held-out checks. This uses an evidence path, leaving official publication to the
loop. A failed experimental ONNX must remain unpublished.

Then check the production fixed-batch adapter on that verified artifact:

```bash
"$EXPORT_PY" pipeline/infer_probe.py \
  --model "$EXPORT_RECOVERY/checked.onnx" \
  --providers cpu coreml --verify-fixed 32 --timeout 60
```

Require both PASS results. The two checks have different purposes: normal export
checks CPU PyTorch against CPU ONNX Runtime; this second command checks exact
versus fixed inference through the Rust engine. Selected CoreML can still fall
back to CPU. Do not describe either check as proving every possible backend.

## 8. Resume the saved loop and inspect its decision

The diagnostic process must have exited and released its lock. Resume normally:

```bash
caffeinate -i "$EXPORT_PY" -u pipeline/loop.py \
  --run "$EXPORT_RUN" --engine engine/target/release/engine --until-gen 1 \
  2>&1 | tee "$EXPORT_RECOVERY/resume.log"
```

The loop retries export from the saved checkpoint. It should not replay the 1,200
games or repeat the 4,000 optimizer steps. After successful publication it runs
candidate-versus-incumbent matches, records a promotion decision, performs
applicable anchors and pruning, and stops after generation 1.

Refresh the analytical report after completion:

```bash
"$EXPORT_PY" pipeline/analyze.py --run "$EXPORT_RUN" \
  --out "$HOME/Library/Logs/chessbot/analysis"
open "$HOME/Library/Logs/chessbot/analysis/index.html"
```

Promotion or rejection are both valid evaluation outcomes. The earlier slow
attempts remain in the time ledger, so this generation remains a trial for M2's
eight-hour criterion. Normal recovery legitimately updates run state and can
prune a rejected candidate's native model files; keep the diagnostic backup.
