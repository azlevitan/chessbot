"""Report publication shares the loop's existing completion and retry boundary."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

import loop
from run_layout import generation_dir, read_run_json, write_json_atomic
from test_loop import bare_run, invoke
from test_loop_state import arguments


def test_reporting_follows_successful_pruning_before_done_checkpoint(tmp_path, monkeypatch):
    bare_run(tmp_path, stage="prune")
    events = []

    def prune(run, engine, state):
        assert state["stage"] == "prune"
        events.append("prune")
        return deepcopy(state)

    def report(run):
        saved = read_run_json(run)
        assert saved["current_gen"] == 1
        events.append("report:" + saved["stage"])

    monkeypatch.setattr(loop, "run_stage", prune)
    monkeypatch.setattr(loop, "write_report", report)
    loop.run_loop(arguments(tmp_path, "--until-gen", "1"))
    assert events == ["prune", "report:prune", "report:done"]
    state = read_run_json(tmp_path)
    assert (state["current_gen"], state["stage"]) == (2, "selfplay")


@pytest.mark.parametrize("stage", ["prune", "done"])
@pytest.mark.parametrize("failure", [OSError, ValueError, KeyboardInterrupt])
def test_report_failure_keeps_checkpoint_releases_lock_and_can_retry(
        tmp_path, monkeypatch, stage, failure):
    bare_run(tmp_path, stage=stage)
    before = (tmp_path / "run.json").read_bytes()
    monkeypatch.setattr(loop, "run_stage", lambda run, engine, state: deepcopy(state))

    def fail(run):
        assert read_run_json(run)["stage"] == stage
        raise failure("publication interrupted")

    monkeypatch.setattr(loop, "write_report", fail)
    with pytest.raises(failure, match="publication interrupted"):
        loop.run_loop(arguments(tmp_path, "--until-gen", "1"))
    assert (tmp_path / "run.json").read_bytes() == before
    assert not (tmp_path / "loop.lock").exists()

    monkeypatch.setattr(loop, "write_report", lambda run: None)
    loop.run_loop(arguments(tmp_path, "--until-gen", "1"))
    state = read_run_json(tmp_path)
    assert (state["current_gen"], state["stage"]) == (2, "selfplay")


def test_failed_prune_preserves_reports_and_checkpoint(tmp_path, monkeypatch):
    bare_run(tmp_path, stage="prune")
    before = (tmp_path / "run.json").read_bytes()
    paths = [tmp_path / "ladder.json", tmp_path / "report.tsv"]
    for path in paths:
        path.write_text("previous report\n")
    stamps = {path: path.stat().st_mtime_ns for path in paths}

    def fail(*args):
        raise subprocess.CalledProcessError(1, ["prune"])

    monkeypatch.setattr(loop, "run_stage", fail)
    monkeypatch.setattr(loop, "write_report", lambda run: pytest.fail("unexpected report"))
    with pytest.raises(subprocess.CalledProcessError):
        loop.run_loop(arguments(tmp_path, "--until-gen", "1"))
    assert (tmp_path / "run.json").read_bytes() == before
    for path in paths:
        assert path.read_text() == "previous report\n"
        assert path.stat().st_mtime_ns == stamps[path]
    assert not (tmp_path / "loop.lock").exists()


@pytest.mark.parametrize("stage", ["prune", "done"])
def test_real_tail_crash_publishes_reports_then_rebuilds_on_resume(tmp_path, stage):
    bare_run(tmp_path, stage=stage, best_gen=1)
    write_json_atomic(generation_dir(tmp_path, 0) / "decision.json", {
        "promoted": True, "reason": "bootstrap",
    })
    write_json_atomic(generation_dir(tmp_path, 1) / "decision.json", {
        "promoted": True, "reason": "sprt_accept", "games": 40, "score": 0.6,
        "elo": 50.0, "elo_ci95": [10.0, 90.0], "llr": 3.0,
    })
    checkpoint = (tmp_path / "run.json").read_bytes()
    crashed = invoke(tmp_path, Path(sys.executable), "--until-gen", 1,
                     crash=f"1:{stage}", timeout=10)
    assert crashed.returncode == 17, crashed.stdout + crashed.stderr
    assert (tmp_path / "run.json").read_bytes() == checkpoint
    paths = [tmp_path / "ladder.json", tmp_path / "report.tsv"]
    contents = {path: path.read_bytes() for path in paths}
    assert [entry["elo"] for entry in json.loads(contents[paths[0]])] == [0, 50]
    # Rebuild both derived files even if one disappeared and another is stale.
    paths[0].unlink()
    paths[1].write_text("stale\n")
    for path in paths:
        path.with_name(path.name + ".tmp").write_text("interrupted\n")

    resumed = invoke(tmp_path, Path(sys.executable), "--until-gen", 1, timeout=10)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    state = read_run_json(tmp_path)
    assert (state["current_gen"], state["stage"], state["best_gen"]) == (2, "selfplay", 1)
    assert {path: path.read_bytes() for path in paths} == contents
    assert not (tmp_path / "loop.lock").exists() and not list(tmp_path.rglob("*.tmp"))


def test_already_reached_generation_bound_does_not_rebuild(tmp_path, monkeypatch):
    bare_run(tmp_path, stage="done", current_gen=2)
    monkeypatch.setattr(loop, "write_report", lambda run: pytest.fail("unexpected report"))
    loop.run_loop(arguments(tmp_path, "--until-gen", "1"))
    assert read_run_json(tmp_path)["stage"] == "done"
