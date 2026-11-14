"""Same-session shape comparisons, bounded failures and retained raw evidence."""

import csv
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from infer_probe import TUNINGS, main, run_case, summarize


def trace(*, fixed=True):
    """Warmup deliberately dominates; physical shape never changes in fixed mode."""
    records = [{"format": "chessbot-infer-bench-v1", "kind": "session",
                "batches": [32, 31, 32, 20], "cycles": 2, "repeat": 2,
                "load_seconds": 0.25}]
    sequence = [32, 32, 31, 31, 32, 32, 20, 20]
    timed = [0.011, 0.001, 0.022, 0.002, 0.033, 0.003, 0.044, 0.004]
    for cycle in (1, 2):
        for offset, batch in enumerate(sequence):
            records.append({"format": "chessbot-infer-bench-v1", "kind": "call",
                            "index": (cycle - 1) * 8 + offset + 1, "cycle": cycle,
                            "repeat_index": offset % 2 + 1, "requested_batch": batch,
                            "physical_batch": 32 if fixed else batch,
                            "shape_changed": False if fixed else offset % 2 == 0,
                            "first_for_shape": cycle == 1 and offset == 0,
                            "seconds": 100 + offset if cycle == 1 else timed[offset]})
    calls = records[1:]
    records.append({"format": "chessbot-infer-bench-v1", "kind": "summary",
                    "calls": 16, "requested_positions": 460,
                    "physical_positions": 512 if fixed else 460,
                    "inference_seconds": sum(row["seconds"] for row in calls)})
    return records


def save_trace(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


@pytest.mark.parametrize("fixed", [False, True])
def test_warmup_is_excluded_from_shape_comparison_but_included_in_full_cost(tmp_path, fixed):
    path = tmp_path / "timing.jsonl"
    records = trace(fixed=fixed)
    save_trace(path, records)
    result = summarize(path)

    # Changing requested sizes remains a transition when every physical call is 32.
    assert result["change_ms"] == pytest.approx(27.5)
    assert result["repeat_ms"] == pytest.approx(2.5)
    assert result["first_cycle_seconds"] == 828
    assert result["calls"] == 16
    assert result["load_seconds"] == 0.25
    assert result["inference_seconds"] == pytest.approx(828.12)
    assert result["requested_positions"] == 460
    assert result["physical_positions"] == (512 if fixed else 460)
    assert result["requested_positions_per_sec"] == pytest.approx(460 / 828.12)


@pytest.mark.parametrize("damage", ["missing_call", "wrong_summary_count", "missing_summary",
                                    "missing_session", "truncated_line"])
def test_malformed_or_incomplete_raw_trace_cannot_be_reported_as_complete(tmp_path, damage):
    records = trace()
    if damage == "missing_call":
        del records[4]
    elif damage == "wrong_summary_count":
        records[-1]["calls"] = 15
    elif damage == "missing_summary":
        records.pop()
    elif damage == "missing_session":
        records.pop(0)
    path = tmp_path / "partial.jsonl"
    save_trace(path, records)
    if damage == "truncated_line":
        with path.open("a") as stream:
            stream.write('{"kind":"call","seconds":')
    with pytest.raises((ValueError, KeyError, StopIteration)):
        summarize(path)


def test_timeout_retains_partial_stdout_stderr_and_command_context(tmp_path, monkeypatch):
    def timed_out(command, *, stdout, stderr, timeout):
        stdout.write('{"kind":"session","phase":"model-loaded"}\n')
        stderr.write("infer-bench: call 1, requested 32, physical 32\n")
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr("infer_probe.subprocess.run", timed_out)
    result = run_case(Path("engine"), Path("model.onnx"), tmp_path,
                      "cpu", TUNINGS[2], timeout=0.1)
    assert result["status"] == "timeout"
    assert "0.1" in result["error"]
    assert Path(result["jsonl"]).read_text().endswith('"model-loaded"}\n')
    assert "call 1" in Path(result["stderr"]).read_text()
    assert result["command"][-2:] == ["--pad-to", "32"]
    assert "calls" not in result
    assert "change_ms" not in result


@pytest.mark.parametrize("exit_code", [0, 23])
def test_engine_failure_or_partial_success_preserves_diagnostic_files(tmp_path, monkeypatch, exit_code):
    def failed(command, *, stdout, stderr, timeout):
        stdout.write('{"kind":"session"}\n')
        stderr.write("specific engine diagnostic\n")
        return subprocess.CompletedProcess(command, exit_code)

    monkeypatch.setattr("infer_probe.subprocess.run", failed)
    result = run_case(Path("engine"), Path("model.onnx"), tmp_path,
                      "cpu", TUNINGS[0], timeout=1)
    assert result["status"] == "failed"
    assert "error" in result
    assert Path(result["jsonl"]).read_text() == '{"kind":"session"}\n'
    assert Path(result["stderr"]).read_text() == "specific engine diagnostic\n"
    assert "requested_positions_per_sec" not in result


def test_cpu_probe_runs_three_bounded_cases_and_publishes_auditable_summary(
        tmp_path, engine_bin, repo_root):
    model = repo_root / "engine/tests/fixtures/match_uniform.onnx"
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    output = tmp_path / "probe"
    assert main(["--model", str(model), "--engine", str(engine_bin),
                 "--providers", "cpu", "--timeout", "15", "--out", str(output)]) == 0

    report = json.loads((output / "summary.json").read_text())
    assert report["format"] == "chessbot-infer-probe-v1"
    assert report["model"] == str(model.resolve())
    assert report["model_sha256"] == model_hash
    assert hashlib.sha256(model.read_bytes()).hexdigest() == model_hash
    assert {row["tuning"] for row in report["cases"]} == {row[0] for row in TUNINGS}
    assert len(report["cases"]) == 3
    for case in report["cases"]:
        assert case["status"] == "complete"
        assert case["provider"] == "cpu"
        assert case["calls"] == 40
        assert case["requested_positions"] == 1150
        records = [json.loads(line) for line in Path(case["jsonl"]).read_text().splitlines()]
        calls = [row for row in records if row["kind"] == "call"]
        assert len(calls) == 40
        assert case["first_cycle_seconds"] == pytest.approx(sum(row["seconds"] for row in calls[:8]))
        assert case["inference_seconds"] == pytest.approx(sum(row["seconds"] for row in calls))
        assert case["change_ms"] >= 0 and case["repeat_ms"] >= 0
        assert case["session"]["memory_pattern"] == (case["tuning"] != "exact_off")
        if case["tuning"] == "fixed32_on":
            assert case["physical_positions"] == 1280
            assert {row["physical_batch"] for row in calls} == {32}
        else:
            assert case["physical_positions"] == 1150
            assert all(row["physical_batch"] == row["requested_batch"] for row in calls)
    with (output / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert {row["status"] for row in rows} == {"complete"}
    assert {row["calls"] for row in rows} == {"40"}


def test_cpu_parity_reports_only_successful_complete_checks(tmp_path, engine_bin, repo_root):
    model = repo_root / "engine/tests/fixtures/match_uniform.onnx"
    output = tmp_path / "parity"
    assert main(["--model", str(model), "--engine", str(engine_bin), "--providers", "cpu",
                 "--verify-fixed", "32", "--out", str(output)]) == 0
    report = json.loads((output / "summary.json").read_text())
    assert report["verify_fixed"] == 32 and len(report["cases"]) == 1
    case = report["cases"][0]
    assert case["status"] == "complete" and case["batches_checked"] == 7
    assert "change_ms" not in case
    raw = [json.loads(line) for line in Path(case["jsonl"]).read_text().splitlines()]
    assert [r["requested_batch"] for r in raw if r["kind"] == "parity"] == [1, 7, 20, 31, 32, 33, 65]
    assert raw[-1]["pass"] is True
