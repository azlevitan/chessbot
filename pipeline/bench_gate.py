#!/usr/bin/env python3
"""GATE 7: release uniform-search throughput and real batch utilization (stdlib).

Run from any directory. Model loading/build time is outside the engine's timer.
The fixed floors are intentionally enforced even when Python runs with -O.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

from run_layout import ENGINE_BIN, ENGINE_DIR

FIELDS = {"sims", "evals", "seconds", "sims_per_sec", "evals_per_sec", "avg_batch", "ep"}
EXPECTED_SIMS = 6 * 800


def validate_result(result: object, batch: int) -> dict:
    """Reject malformed results as well as performance regressions."""
    if not isinstance(result, dict) or set(result) != FIELDS:
        raise ValueError("bench must return the SPEC §8 JSON object")
    if result["ep"] != "uniform":
        raise ValueError("gate must use the uniform evaluator")
    if type(result["sims"]) is not int or result["sims"] != EXPECTED_SIMS:
        raise ValueError(f"expected {EXPECTED_SIMS} completed simulations")
    if type(result["evals"]) is not int or not 0 < result["evals"] <= EXPECTED_SIMS:
        raise ValueError("invalid evaluation count")
    for field in ("seconds", "sims_per_sec", "evals_per_sec", "avg_batch"):
        value = result[field]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field} must be finite and positive")
    if not 1 <= result["avg_batch"] <= batch:
        raise ValueError(f"average batch must be between 1 and {batch}")
    for rate, count in (("sims_per_sec", "sims"), ("evals_per_sec", "evals")):
        if not math.isclose(result[rate], result[count] / result["seconds"], rel_tol=1e-6):
            raise ValueError(f"{rate} disagrees with measured time and count")
    if batch == 1 and result["sims_per_sec"] < 50_000:
        raise ValueError(f"batch 1 throughput {result['sims_per_sec']:.1f} is below 50,000 sims/s")
    if batch == 8 and result["avg_batch"] < 6.0:
        raise ValueError(f"batch 8 average {result['avg_batch']:.3f} is below 6.0")
    return result


def run_benchmark(batch: int) -> dict:
    process = subprocess.run(
        [str(ENGINE_BIN), "bench", "--model", "uniform", "--sims", "800", "--batch", str(batch)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    if len(process.stdout.splitlines()) != 1:
        raise ValueError("bench must print exactly one JSON line on stdout")
    return validate_result(json.loads(process.stdout), batch)


def main() -> int:
    try:
        subprocess.run(
            ["cargo", "build", "--release", "--manifest-path", str(ENGINE_DIR / "Cargo.toml")],
            check=True, stdout=sys.stderr,
        )
        for batch in (1, 8):
            result = run_benchmark(batch)
            print(f"PASS batch {batch}: {json.dumps(result, sort_keys=True, allow_nan=False)}")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"FAIL GATE 7: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
