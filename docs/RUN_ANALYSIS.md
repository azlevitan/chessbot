# Analyzing an RL run

Run from the repository root on the Mac that owns the run:

```bash
~/.venvs/chessbot/bin/python pipeline/analyze.py --run data_store/runs/main
open data_store/runs/main/analysis/index.html
```

The exporter uses only Python's standard library. It writes CSV tables, an input-diagnostics JSON file, and a standalone HTML report containing SVG plots. The report works offline and requires no plotting service or additional packages. Rerun the command to refresh it, including while training is active. `--out /path/to/directory` chooses another output directory. Source models and sample contents are never opened.

Pipeline verification tests use temporary runs with the `ci_tiny.json` preset. Its evaluation games are capped at 40 individual moves (`max_plies: 40`) with match batch size 1; temporary training/export jobs use one CPU thread. These limits reduce the test workload on slower machines while retaining the existing timeout. The owner's Mac rerun is still required to verify its runtime there. These test settings do not change an overnight run's saved configuration. If a verification check fails, rerun that check alone before repeating the full suite.

The HTML charts show training objective, fixed training-probe and held-out policy/WDL losses, policy accuracy, held-out WDL Brier score, generation validation, baseline match scores, candidate Elo with confidence intervals, the promoted ladder, SPRT progress, self-play outcomes, durations, and measured process memory. Hover over points for their values. Missing measurements remain unavailable rather than becoming zero.

## Plot-ready tables

| File | One row represents | Useful plots |
| --- | --- | --- |
| `training.csv` | One baseline or epoch evaluation within an optimizer attempt | Loss, accuracy, learning rate and WDL probability quality versus optimizer step |
| `training_attempts.csv` | One training attempt, with outcome and source/probe metadata | Separate retries; identify the attempt published in `train.json` |
| `generations.csv` | One candidate, including rejected and unfinished candidates | Validation, promotion, anchors, relative Elo, resource costs versus generation |
| `matches.csv` | One match block with cumulative results | Cumulative W/D/L, score, Elo/CI and SPRT LLR versus games played |
| `match_games.csv` | One saved evaluation game | Game length and termination distributions, grouped by color/opening/generation |
| `selfplay.csv` | One complete chunk, or its preserved metadata snapshot | Draw/resignation/termination rates, samples per game and audit results |
| `stages.csv` | One stage invocation, including interrupted and failed attempts | Attempt durations and captured effective configuration |
| `processes.csv` | One measured process/worker invocation | Peak RSS, command/unit, exit status and measurement provenance |
| `stage_summary.csv` | One stage, aggregating its attempts | Full stage duration and resource-measurement completeness |

CSV missing values are `NA`. Nested provenance is JSON inside quoted CSV fields. `diagnostics.json` lists ignored partial history records, malformed optional artifacts, missing diagnostic counters and unavailable historic SPRT settings. Check that file before drawing conclusions from a sparse plot.

For example, in a separate plotting environment with pandas:

```python
import pandas as pd

history = pd.read_csv("data_store/runs/main/analysis/training.csv")
candidate = history[(history.gen == 1) & history.published]
candidate.plot(x="steps", y=["train_probe_policy_loss", "val_policy_loss"])
```

## Interpreting training and overfitting

`train_loss`, `train_policy_loss` and `train_wdl_loss` are optimizer block means. In RL the WDL objective blends game outcomes with search Q values. These optimizer losses cannot be treated as directly comparable to hard-outcome validation WDL loss.

`train_probe_*` and `val_*` are evaluations with the model in evaluation mode and Q blending disabled. The probe is a fixed, bounded sample of that attempt's training mixture. Validation uses an evenly spaced subset across the held-out supervised source. The baseline at optimizer step zero evaluates the incoming model before any update, so compare the candidate with that baseline on exactly the same held-out subset.

Watch for a sustained fall in training-probe loss accompanied by rising validation loss or falling validation accuracy. That is a reason to inspect training duration, learning rate, replay diversity and source weights. It is not proof of overfitting: the supervised holdout and evolving RL replay data have different distributions. Compare probe curves within an attempt; probe composition changes between generations. Policy targets in self-play are distributions, so their entropy affects policy cross-entropy. `*_policy_target_entropy` and `*_policy_kl` distinguish target entropy from fit. WDL Brier score measures probability quality against true outcomes; lower is better, with a three-class summed range of 0 to 2.

Every training invocation has a new `attempt_id`. An interrupted training stage restarts optimization rather than continuing the same optimizer state. Curves and CSVs keep those attempts separate; `published` identifies the attempt referenced by the completed `train.json`. `unfinished` means no terminal history event was recorded and can describe either an active process or an abrupt crash. A published summary establishes completion even if the final history append was lost.

Repeatedly choosing hyperparameters or training duration from the same validation results gradually adapts the workflow to that validation set. Keep a separate untouched test source for occasional final audits and avoid using its results to select checkpoints or tune settings. The current loop evaluates `supervised_val`; an independent final test set is not evaluated automatically.

Older generation-0 summaries contain only their saved final metrics. Past learning curves cannot be reconstructed from those summaries. Generation 0 may also have used a different held-out sample selector; its recorded validation score is not automatically comparable to a newly selected subset. Use each new attempt's step-zero baseline for a controlled before/after comparison.

## Interpreting model quality and self-play

Policy accuracy and validation losses measure prediction on a supervised holdout. Evaluation matches measure playing strength. Check both: a better imitation score alone does not establish a stronger engine. Candidate Elo compares each candidate to its own incumbent; ladder Elo is relative to generation 0 and is not an absolute Stockfish calibration. Candidate confidence intervals describe individual matches rather than accumulated ladder uncertainty. Anchor scores are only present for generations whose anchor matches were run, ordinarily promoted generations.

Match LLR uses effective settings captured in stage telemetry, or the generation's saved configuration when no matching telemetry exists. Differing match-stage settings make historical LLR ambiguous; the exporter leaves it unavailable. For legacy blocks without generation/stage provenance, match counts, empirical scores and relative Elo remain useful, but LLR is not reconstructed from today's potentially different run configuration.

Self-play snapshots are saved as `gens/gen_GGGG/selfplay_summary.json` outside the removable `selfplay/` directory. Chunk diagnostics therefore survive later replay pruning. Legacy runs can only preserve metadata that still exists. `samples_per_game` counts training samples per game; it is not an exact mean ply count, because a resignation can add a final sample without a played move. Exact evaluation-game plies are available in `match_games.csv`. A missing resignation playthrough denominator gives `NA`, not a zero false-positive rate.

## Time, memory and resumption

### Checking a quiet self-play run

The engine prints a progress line every five seconds, even before a game finishes. Read `moves` and `eval_positions` across successive lines: increasing values show actual moves or successful model evaluations. `first_game=1:ply12` means the first game has played 12 individual moves. `completed` includes finished games waiting for an earlier game; `emitted` counts games written in the required order. A saved chunk still requires the final `.bin` and `.json` files.

`phase=loading_model` identifies startup. `phase=evaluating` with a growing `in_flight` duration and unchanged evaluation/move counts shows one evaluation has not returned; a changing elapsed-time display alone does not establish progress. `since_progress` measures time since a move, successful evaluation, completion or write count changed. Keep several successive lines when diagnosing a stall. A short probe completing four moves does not establish full-game speed or an eight-hour generation ETA.

Existing run files and their measurements should be preserved after interruption. Restart through `pipeline/loop.py`; an unfinished chunk will be regenerated. Keep diagnostic outputs separate from the run. On a Mac with a OneDrive-backed repository, use a local directory such as `~/Library/Logs/chessbot` for console logs and probes. Moving a console log does not establish that dataset or checkpoint writes are healthy. The overnight preset remains a workload to measure, and M2's eight-hour target requires measured generation results.

### Measuring model-call overhead

If self-play advances slowly, pause the real loop before running this bounded comparison:

```bash
~/.venvs/chessbot/bin/python pipeline/infer_probe.py \
  --model data_store/runs/main/gens/gen_0000/model.onnx
```

On macOS, the script runs six cases using the actual Rust inference path: CPU and CoreML selection, each with ordinary inputs and memory patterns enabled (`exact_on`), ordinary inputs and memory patterns disabled (`exact_off`), and inputs padded to 32 positions with memory patterns enabled (`fixed32_on`). Production inference retains its existing settings. Linux defaults to the three CPU cases. `--providers cpu` selects CPU alone; `--timeout 30` limits each subprocess, so the default six-case Mac comparison has at most three minutes of engine work. A timeout marks an incomplete diagnostic rather than proving a deadlock.

Each case loads one model session, encodes the same position in Rust, and repeats the requested group sizes `32,31,32,20` five times with two calls per group. Timing covers tensor construction, model execution and result extraction. Input preparation, output validation and file writing occur outside those timings. The comparison is useful because repeated fixed-size calls can be fast even when changing sizes incurs setup costs. The ORT library recommends disabling memory patterns for varying input sizes; this experiment measures that setting rather than assuming it resolves the owner's slowdown.

Results go to a new local `~/Library/Logs/chessbot/infer-probe-*` directory on macOS (`work/` on Linux), or a new directory supplied by `--out`. Raw JSONL retains every call and session settings. `summary.json` and `summary.csv` record the model hash, startup time, first-cycle time, total model-call time, and median durations after a requested-size change and on immediate repeats. The two medians exclude the first cycle; startup and first-use costs remain available in the raw records. A zero-rounded duration in old game logs is not a zero-cost model call. The selected provider label still permits ORT CPU fallback.

The equivalent individual command is `engine infer-bench --model PATH --ep cpu --batches 32,31,32,20 --cycles 5 --repeat 2 --memory-pattern on`. Add `--pad-to 32` for fixed-size inputs or use `--memory-pattern off` to isolate that setting. Batches are capped at 64 and only one call's outputs are retained. These diagnostics establish model-call behavior; full-generation duration and actual process RSS still require an M2 run. Preserving complete chunks on resume remains a separate acceptance check.

Stage elapsed time is summed across all recorded attempts, including work lost to interruption or failure. Process durations are retained separately and must not be added again to stage totals. Measured stage spans include setup/validation and retries. `train_attempt_seconds` retains the trainer's narrower successful-attempt duration, including its probe/validation evaluations; it is not pure optimizer time. `train_seconds` prefers complete stage telemetry and falls back to that legacy duration only when no training stage ledger exists.

`stage_summary.csv` reports `rss_complete` per stage. Process records include measurement method, platform and byte-normalized peak RSS. `recorded_rss_complete` in the generation table describes available attempted stages only. Full-generation `rss_complete` also requires a finished generation and all seven stage boundaries, so an early live run cannot accidentally claim complete memory evidence. Peak RSS is the largest recorded process peak, not the sum of independently attained peaks and not a measure of total unified GPU/system memory.

Pre-telemetry work, abrupt unfinalized attempts and missing resource records keep evidence incomplete. Historical work is never retroactively assigned zero time or memory. The native milestone report remains `report.tsv`; the analysis tables provide the detail needed to audit it.
