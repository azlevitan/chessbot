"""Diagnostic evidence cannot publish candidates, lose checkpoints or hide failures."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

import export_diagnostics as diagnostics


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


@pytest.fixture
def candidate(tmp_path):
    run = tmp_path / "run"
    gen = run / "gens/gen_0001"
    gen.mkdir(parents=True)
    (gen / "model.pt").write_bytes(b"precious completed generation-one weights")
    write(gen / "train.json", {"steps": 4000, "samples_seen": 2048000,
                               "attempt_id": "trained", "final_loss": 2.4})
    (gen / "train_history.jsonl").write_text('{"event":"complete","attempt_id":"trained"}\n')
    cfg = {"seed": 0, "train": {"steps": 4000},
           "selfplay": {"games": 4, "games_per_chunk": 2}}
    write(run / "run.json", {"format": "chessbot-run-v1", "config": cfg,
          "best_gen": 0, "current_gen": 1, "stage": "export", "resign_threshold": -.95})
    write(gen / "generation.json", {"format": "chessbot-generation-v1", "gen": 1, "config": cfg})
    incumbent = run / "gens/gen_0000/model.onnx"
    incumbent.parent.mkdir()
    incumbent.write_bytes(b"unchanged incumbent")
    write(gen / "selfplay/.inputs.json", {"model": str(incumbent), "resign_threshold": -.95})
    for index in range(2):
        binary = gen / "selfplay" / f"chunk_{index:06d}.bin"
        binary.write_bytes(bytes(492))
        write(binary.with_suffix(".json"), {"format": "chessbot-chunk-v1", "kind": "selfplay",
            "samples": 1, "games": 2, "model": str(incumbent), "seed": 1000000 + 1000 * index,
            "sims": 200, "max_plies": 512, "temp_plies": 30,
            "resign": {"threshold": -.95, "playthrough_games": 0, "false_positives": 0}})
    validation = run / "supervised_val/chunk_000000.bin"
    validation.parent.mkdir()
    validation.write_bytes(bytes(492))
    write(validation.with_suffix(".json"), {"format": "chessbot-chunk-v1", "kind": "pgn",
                                           "samples": 1, "games": 1})
    return run


def snapshot(run):
    return {str(path.relative_to(run)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in run.rglob("*") if path.is_file() and path.name != "loop.lock"}


def profile(name, run, evidence, timeout, failing_row):
    return {"profile": name, "completed": True, "gate_pass": True, "cases": []}


def test_backup_precedes_workers_and_preserves_every_completed_artifact(candidate, tmp_path, monkeypatch):
    before = snapshot(candidate)
    output, calls = tmp_path / "evidence", []

    def inspect(name, run, evidence, timeout, failing_row):
        calls.append(name)
        assert int((candidate / "loop.lock").read_text()) == os.getpid()
        assert (output / "backup/model.pt").read_bytes() == before["gens/gen_0001/model.pt"][0]
        assert (output / "backup/train.json").read_bytes() == before["gens/gen_0001/train.json"][0]
        assert (output / "backup/run.json").read_bytes() == before["run.json"][0]
        return profile(name, run, evidence, timeout, failing_row)

    monkeypatch.setattr(diagnostics, "_run_profile", inspect)
    result = diagnostics.diagnose(candidate, output)
    assert calls == list(diagnostics.PROFILE_NAMES)
    assert result["format"] == "chessbot-export-diagnostics-v1"
    assert result["baseline_pass"] is True and result["protected_unchanged"] is True
    assert snapshot(candidate) == before
    assert not (candidate / "loop.lock").exists()
    assert not (candidate / "gens/gen_0001/model.onnx").exists()
    assert not (candidate / "gens/gen_0001/model.onnx.tmp").exists()
    manifest_path = Path(result["preservation_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = output / manifest_path
    manifest = json.loads(manifest_path.read_text())
    protected = ["run.json", "gens/gen_0001/model.pt", "gens/gen_0001/train.json"]
    protected += [key for key in before if "/selfplay/chunk_" in key or key.startswith("supervised_val/")]
    for key in protected:
        assert manifest["files"][key] == {"bytes": len(before[key][0]), "mtime_ns": before[key][1],
                                          "sha256": hashlib.sha256(before[key][0]).hexdigest()}
    assert json.loads((output / "report.json").read_text())["baseline_pass"] is True


def test_experimental_pass_never_overrides_failed_original_gate(candidate, tmp_path, monkeypatch):
    calls = []
    def failed_baseline(name, run, evidence, timeout, failing_row):
        calls.append(name)
        return {**profile(name, run, evidence, timeout, failing_row), "gate_pass": name != "baseline"}
    monkeypatch.setattr(diagnostics, "_run_profile", failed_baseline)
    output = tmp_path / "numeric-failure"
    assert diagnostics.main(["--run", str(candidate), "--out", str(output)]) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["baseline_pass"] is False
    assert report["protected_unchanged"] is True
    assert calls == list(diagnostics.PROFILE_NAMES)
    assert all(row["completed"] for row in report["profiles"])
    assert read_state(candidate)["stage"] == "export"


def read_state(run):
    return json.loads((run / "run.json").read_text())


def test_incomplete_profile_retains_raw_evidence_and_other_profiles_continue(candidate, tmp_path, monkeypatch):
    output, calls = tmp_path / "partial-profile", []
    def timed_out(name, run, evidence, timeout, failing_row):
        calls.append(name)
        if name != "single":
            return profile(name, run, evidence, timeout, failing_row)
        raw, stderr = output / "single.partial.json", output / "single.stderr.log"
        raw.write_text('{"case":"random7","completed":false}\n')
        stderr.write_text("last completed case was random1\n")
        return {"profile": name, "completed": False, "error": "timeout", "json": str(raw),
                "stderr": str(stderr)}
    monkeypatch.setattr(diagnostics, "_run_profile", timed_out)
    assert diagnostics.main(["--run", str(candidate), "--out", str(output)]) == 2
    report = json.loads((output / "report.json").read_text())
    assert report["baseline_pass"] is True and report["protected_unchanged"] is True
    assert calls == list(diagnostics.PROFILE_NAMES)
    assert "random7" in (output / "single.partial.json").read_text()
    assert "random1" in (output / "single.stderr.log").read_text()
    assert next(row for row in report["profiles"] if row["profile"] == "single")["completed"] is False
    assert not (candidate / "loop.lock").exists()


def test_changed_source_is_reported_and_not_silently_restored(candidate, tmp_path, monkeypatch):
    source = candidate / "gens/gen_0001/model.pt"
    original = source.read_bytes()
    def mutation(name, run, evidence, timeout, failing_row):
        if name == "baseline":
            source.write_bytes(original + b" external change")
        return profile(name, run, evidence, timeout, failing_row)
    monkeypatch.setattr(diagnostics, "_run_profile", mutation)
    output = tmp_path / "mutation"
    assert diagnostics.main(["--run", str(candidate), "--out", str(output)]) == 2
    report = json.loads((output / "report.json").read_text())
    assert report["protected_unchanged"] is False
    assert report["preservation_errors"]
    assert (output / "backup/model.pt").read_bytes() == original
    assert source.read_bytes() == original + b" external change"
    assert not (candidate / "loop.lock").exists()


@pytest.mark.parametrize("fault", ["wrong_stage", "wrong_steps", "missing_checkpoint", "incomplete_chunk"])
def test_preflight_fails_before_workers_or_evidence_creation(candidate, tmp_path, monkeypatch, fault):
    if fault == "wrong_stage":
        saved = read_state(candidate)
        saved["stage"] = "train"
        write(candidate / "run.json", saved)
    elif fault == "wrong_steps":
        write(candidate / "gens/gen_0001/train.json", {"steps": 3999})
    elif fault == "missing_checkpoint":
        (candidate / "gens/gen_0001/model.pt").unlink()
    else:
        (candidate / "gens/gen_0001/selfplay/chunk_000001.bin").write_bytes(b"partial")
    before = snapshot(candidate)
    monkeypatch.setattr(diagnostics, "_run_profile", lambda *a: pytest.fail("worker ran"))
    output = tmp_path / "must-not-exist"
    with pytest.raises((ValueError, OSError)):
        diagnostics.diagnose(candidate, output)
    assert snapshot(candidate) == before
    assert not output.exists()
    assert not (candidate / "loop.lock").exists()


def test_live_owner_and_output_inside_run_are_refused(candidate, tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "_run_profile", lambda *a: pytest.fail("worker ran"))
    lock = candidate / "loop.lock"
    lock.write_text(str(os.getpid()))
    with pytest.raises(diagnostics.LoopLockError, match="running"):
        diagnostics.diagnose(candidate, tmp_path / "locked")
    assert lock.read_text() == str(os.getpid())
    lock.unlink()
    with pytest.raises((ValueError, OSError)):
        diagnostics.diagnose(candidate, candidate / "evidence")
    assert not (candidate / "evidence").exists()


def test_success_exit_code_requires_original_gate_and_preservation(candidate, tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "_run_profile", profile)
    assert diagnostics.main(["--run", str(candidate), "--out", str(tmp_path / "success")]) == 0


def test_worker_timeout_keeps_partial_logs_and_returns_incomplete(candidate, tmp_path, monkeypatch):
    evidence = tmp_path / "timeout-logs"
    evidence.mkdir()
    def timeout(command, *, stdout, stderr, timeout):
        stdout.write('case random1 completed; random7 started\n')
        stderr.write("worker diagnostic before timeout\n")
        raise subprocess.TimeoutExpired(command, timeout)
    monkeypatch.setattr(diagnostics.subprocess, "run", timeout)
    result = diagnostics._run_profile("baseline", candidate, evidence, .05, 2)
    assert result["completed"] is False and result["gate_pass"] is False
    assert "exceeded" in result["error"]
    assert "random7 started" in Path(result["stdout"]).read_text()
    assert "before timeout" in Path(result["stderr"]).read_text()


def test_worker_operational_failure_keeps_its_partial_comparisons(candidate, tmp_path, monkeypatch):
    evidence = tmp_path / "partial-details"
    evidence.mkdir()
    def failed(command, *, stdout, stderr, timeout):
        path = Path(command[command.index("--report") + 1])
        write(path, {"profile": "baseline", "completed": False, "gate_pass": False,
                     "cases": [{"name": "random_1", "completed": True, "gate_pass": True}],
                     "error": {"type": "RuntimeError", "message": "later operation failed"}})
        return subprocess.CompletedProcess(command, 2)
    monkeypatch.setattr(diagnostics.subprocess, "run", failed)
    result = diagnostics._run_profile("baseline", candidate, evidence, 1, 2)
    assert result["completed"] is False and result["worker_exit_code"] == 2
    assert result["cases"][0]["name"] == "random_1"
    assert result["error"]["message"] == "later operation failed"
    assert Path(result["report_path"]).is_file()
