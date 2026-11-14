"""Bounded, same-session inference comparisons; no RL run files are modified."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from run_layout import ENGINE_BIN


TUNINGS = (("exact_on", "on", None), ("exact_off", "off", None),
           ("fixed32_on", "on", 32))


def summarize(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    session = next(record for record in records if record.get("kind") == "session")
    summary = next(record for record in records if record.get("kind") == "summary")
    calls = [record for record in records if record.get("kind") == "call"]
    expected = len(session["batches"]) * session["cycles"] * session["repeat"]
    if len(calls) != expected or summary["calls"] != expected:
        raise ValueError("incomplete inference diagnostic")
    total = summary["inference_seconds"]
    if not isinstance(total, (int, float)) or not math.isfinite(total) or total <= 0:
        raise ValueError("invalid total inference duration")
    changed, repeated, first_cycle = [], [], []
    previous = None
    for record in calls:
        batch, seconds = record["requested_batch"], record["seconds"]
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid model-call duration")
        if record["cycle"] == 1:
            first_cycle.append(seconds)
        elif previous is not None:
            (changed if batch != previous else repeated).append(seconds)
        previous = batch
    return {
        "calls": len(calls), "load_seconds": session["load_seconds"],
        "inference_seconds": summary["inference_seconds"],
        "first_cycle_seconds": sum(first_cycle),
        "change_ms": statistics.median(changed) * 1000 if changed else None,
        "repeat_ms": statistics.median(repeated) * 1000 if repeated else None,
        "requested_positions": summary["requested_positions"],
        "physical_positions": summary["physical_positions"],
        "requested_positions_per_sec": (summary["requested_positions"]
                                         / summary["inference_seconds"]),
        "session": session,
    }


def run_case(engine, model, directory, provider, tuning, timeout):
    label, memory, padding = tuning
    name = provider + "_" + label
    output, errors = directory / (name + ".jsonl"), directory / (name + ".stderr.log")
    command = [str(engine), "infer-bench", "--model", str(model), "--ep", provider,
               "--threads", "1", "--batches", "32,31,32,20", "--cycles", "5",
               "--repeat", "2", "--memory-pattern", memory]
    if padding is not None:
        command.extend(["--pad-to", str(padding)])
    row = {"provider": provider, "tuning": label, "status": "failed",
           "command": command, "jsonl": str(output), "stderr": str(errors)}
    print(f"Testing {provider} / {label} (limit {timeout:g}s)...", flush=True)
    try:
        with output.open("w") as stdout, errors.open("w") as stderr:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout)
        if completed.returncode:
            raise ValueError(f"engine exited with code {completed.returncode}; see {errors}")
        row.update(summarize(output), status="complete")
        print(f"  after size change: {row['change_ms']:.3f} ms; "
              f"same-size repeat: {row['repeat_ms']:.3f} ms; "
              f"all model calls: {row['inference_seconds']:.3f}s", flush=True)
    except subprocess.TimeoutExpired:
        row.update(status="timeout", error=f"exceeded {timeout:g} seconds")
        print(f"  TIMEOUT: {row['error']}; partial records retained", flush=True)
    except (OSError, ValueError, KeyError, StopIteration) as error:
        row["error"] = str(error)
        print(f"  FAILED: {error}", flush=True)
    return row


def positive(value):
    result = float(value)
    if not 0 < result <= 300:
        raise argparse.ArgumentTypeError("timeout must be in (0, 300] seconds")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=ENGINE_BIN)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeout", type=positive, default=30)
    parser.add_argument("--providers", nargs="+", choices=("cpu", "coreml"),
                        default=["cpu", "coreml"] if sys.platform == "darwin" else ["cpu"])
    args = parser.parse_args(argv)
    model, engine = args.model.resolve(), args.engine.resolve()
    for path in (model, engine):
        if not path.is_file():
            parser.error(f"file not found: {path}")
    base = (Path.home() / "Library/Logs/chessbot" if sys.platform == "darwin"
            else Path.cwd() / "work")
    directory = (args.out or base / datetime.now().strftime("infer-probe-%Y%m%d-%H%M%S-%f")).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    with model.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    rows = [run_case(engine, model, directory, provider, tuning, args.timeout)
            for provider in args.providers for tuning in TUNINGS]
    result = {"format": "chessbot-infer-probe-v1", "model": str(model),
              "model_sha256": digest.hexdigest(), "timeout_seconds": args.timeout,
              "note": "Provider labels describe selection; CPU fallback remains possible.",
              "cases": rows}
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = ("provider", "tuning", "status", "calls", "load_seconds", "inference_seconds",
              "first_cycle_seconds", "change_ms", "repeat_ms", "requested_positions",
              "physical_positions", "requested_positions_per_sec")
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved raw timings and summary: {directory}", flush=True)
    return int(any(row["status"] != "complete" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
