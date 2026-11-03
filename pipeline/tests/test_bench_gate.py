"""The throughput gate rejects regressions, malformed JSON and failed commands."""

import json
import subprocess
from types import SimpleNamespace

import pytest

import bench_gate


def result(batch=1, rate=50_000):
    seconds = 4800 / rate
    return {"sims": 4800, "evals": 4790, "seconds": seconds,
            "sims_per_sec": rate, "evals_per_sec": 4790 / seconds,
            "avg_batch": 1.0 if batch == 1 else 6.0, "ep": "uniform"}


def test_threshold_boundaries_pass_but_regressions_fail():
    for batch in (1, 8):
        assert bench_gate.validate_result(result(batch), batch)
    with pytest.raises(ValueError, match="50,000"):
        bench_gate.validate_result(result(rate=49_999), 1)
    with pytest.raises(ValueError, match="below 6.0"):
        bench_gate.validate_result({**result(8), "avg_batch": 5.99}, 8)


@pytest.mark.parametrize("change", [
    {"sims": 4799}, {"sims": 4800.0}, {"evals": True}, {"evals": 0}, {"evals": 4801},
    {"ep": "cpu"}, {"seconds": 0}, {"seconds": -1}, {"seconds": "0.096"},
    {"avg_batch": 0}, {"avg_batch": 9}, {"sims_per_sec": 100_000},
    {"evals_per_sec": 1}, {"seconds": float("nan")}, {"avg_batch": float("inf")},
    {"seconds": True}, {"unexpected": 1},
])
def test_malformed_or_inconsistent_metrics_are_rejected(change):
    with pytest.raises(ValueError):
        bench_gate.validate_result({**result(8), **change}, 8)


@pytest.mark.parametrize("stdout", ["", "{}\n{}\n", "not json\n", "[]\n", "null\n"])
def test_bad_stdout_fails_the_gate(monkeypatch, stdout):
    monkeypatch.setattr(bench_gate.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=stdout))
    with pytest.raises(ValueError):
        bench_gate.run_benchmark(1)


def test_main_builds_release_and_runs_both_batches_with_exact_arguments(monkeypatch, capsys):
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "cargo":
            return SimpleNamespace()
        batch = int(args[-1])
        return SimpleNamespace(stdout=json.dumps(result(batch)) + "\n")

    monkeypatch.setattr(bench_gate.subprocess, "run", run)
    assert bench_gate.main() == 0
    assert calls[0][0] == ["cargo", "build", "--release", "--manifest-path",
                           str(bench_gate.ENGINE_DIR / "Cargo.toml")]
    for (args, options), batch in zip(calls[1:], (1, 8)):
        assert args == [str(bench_gate.ENGINE_BIN), "bench", "--model", "uniform",
                        "--sims", "800", "--batch", str(batch)]
        assert options["check"] and options["timeout"] == 60
    assert capsys.readouterr().out.count("PASS batch") == 2


@pytest.mark.parametrize("build_fails", [False, True])
def test_failed_build_or_engine_returns_nonzero_with_diagnostics(monkeypatch, capsys, build_fails):
    def run(args, **kwargs):
        if args[0] == "cargo" and not build_fails:
            return SimpleNamespace()
        raise subprocess.CalledProcessError(17, args, stderr="fixture command failed")

    monkeypatch.setattr(bench_gate.subprocess, "run", run)
    assert bench_gate.main() == 1
    captured = capsys.readouterr()
    assert "FAIL GATE 7" in captured.err and "fixture command failed" in captured.err
    assert "PASS" not in captured.out
