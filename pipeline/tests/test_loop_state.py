"""Loop checkpoint, validation and generation-boundary behavior without engines."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest

import loop
from run_layout import read_run_json
from test_loop import bare_run, invoke

REAL_STAGES = ("selfplay", "train", "export", "match", "decide", "anchors", "prune")


def arguments(run, *options):
    return loop.parse_args(["--run", str(run), "--engine", sys.executable, *options])


@pytest.mark.parametrize("options,expected", [
    (("--generations", "1"), 2),
    (("--generations", "2"), 3),
    (("--until-gen", "2"), 3),
    (("--generations", "4", "--until-gen", "2"), 3),
    (("--generations", "1", "--until-gen", "4"), 2),
])
def test_limits_count_the_resumed_generation_and_stop_after_rollover(
        tmp_path, monkeypatch, options, expected):
    original = bare_run(tmp_path, stage="train")
    calls = []

    def work(run, engine, state):
        assert read_run_json(run) == state
        assert Path(engine) == Path(sys.executable).resolve()
        calls.append((state["current_gen"], state["stage"]))
        proposed = deepcopy(state)
        if state["stage"] == "decide":
            proposed["best_gen"] = state["current_gen"]
        return proposed

    monkeypatch.setattr(loop, "run_stage", work)
    loop.run_loop(arguments(tmp_path, *options))
    final = read_run_json(tmp_path)
    assert final["current_gen"] == expected and final["stage"] == "selfplay"
    assert final["best_gen"] == expected - 1 and final["config"] == original["config"]
    expected_calls = [(1, stage) for stage in REAL_STAGES[1:]]
    expected_calls += [(gen, stage) for gen in range(2, expected) for stage in REAL_STAGES]
    assert calls == expected_calls
    assert not (tmp_path / "loop.lock").exists()


def test_done_resume_rolls_over_without_stage_work(tmp_path, monkeypatch):
    bare_run(tmp_path, stage="done", current_gen=4, best_gen=4)
    monkeypatch.setattr(loop, "run_stage", lambda *args: pytest.fail("unexpected stage"))
    loop.run_loop(arguments(tmp_path, "--generations", "1"))
    state = read_run_json(tmp_path)
    assert (state["current_gen"], state["stage"], state["best_gen"]) == (5, "selfplay", 4)


def test_state_mutations_commit_together_with_stage_advancement(tmp_path, monkeypatch):
    bare_run(tmp_path)

    def work(run, engine, state):
        proposed = deepcopy(state)
        if state["stage"] == "selfplay":
            proposed["resign_threshold"] = -0.97
        else:
            assert state["resign_threshold"] == -0.97
        if state["stage"] == "decide":
            proposed["best_gen"] = 1
        if state["stage"] == "anchors":
            assert read_run_json(run)["best_gen"] == 1
        return proposed

    monkeypatch.setattr(loop, "run_stage", work)
    loop.run_loop(arguments(tmp_path, "--generations", "1"))
    state = read_run_json(tmp_path)
    assert state["resign_threshold"] == -0.97 and state["best_gen"] == 1


@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError])
def test_failed_work_does_not_commit_mutated_state(tmp_path, monkeypatch, failure):
    bare_run(tmp_path)
    before = (tmp_path / "run.json").read_bytes()

    def fail(run, engine, state):
        state["best_gen"] = 1
        state["resign_threshold"] = -0.99
        raise failure("during stage")

    monkeypatch.setattr(loop, "run_stage", fail)
    with pytest.raises(failure):
        loop.run_loop(arguments(tmp_path, "--generations", "1"))
    assert (tmp_path / "run.json").read_bytes() == before
    assert not (tmp_path / "loop.lock").exists()


@pytest.mark.parametrize("updates", [
    {"current_gen": 0}, {"current_gen": True}, {"current_gen": "1"},
    {"best_gen": -1}, {"best_gen": True}, {"best_gen": 2},
    {"stage": "unknown"}, {"stage": None}, {"config": []},
    {"config": {"selfplay": {"games_per_chunk": 1001}}},
    {"resign_threshold": 0.1}, {"resign_threshold": -1.1},
    {"resign_threshold": "-0.95"},
])
def test_invalid_checkpoint_fails_without_writing(tmp_path, updates):
    bare_run(tmp_path, **updates)
    before = (tmp_path / "run.json").read_bytes()
    result = invoke(tmp_path, sys.executable, "--generations", 1, timeout=10)
    assert result.returncode != 0
    assert (tmp_path / "run.json").read_bytes() == before
    assert not (tmp_path / "loop.lock").exists()


@pytest.mark.parametrize("option,value", [
    ("--generations", "0"), ("--generations", "-1"),
    ("--until-gen", "0"), ("--until-gen", "-1"),
])
def test_invalid_cli_bounds_fail_before_work(tmp_path, option, value):
    bare_run(tmp_path)
    result = invoke(tmp_path, sys.executable, option, value, timeout=10)
    assert result.returncode != 0
    assert not (tmp_path / "loop.lock").exists()


def test_stage_process_failure_preserves_checkpoint_and_releases_lock(tmp_path):
    bare_run(tmp_path)
    before = (tmp_path / "run.json").read_bytes()
    engine = tmp_path / "failing-engine"
    engine.write_text(f"#!{sys.executable}\nraise SystemExit(23)\n")
    engine.chmod(0o755)
    result = invoke(tmp_path, engine, "--generations", 1, timeout=10)
    assert result.returncode != 0 and result.returncode != 130
    assert (tmp_path / "run.json").read_bytes() == before
    assert not (tmp_path / "loop.lock").exists()
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("stage", ["prune", "done"])
def test_crash_at_tail_preserves_checkpoint_then_recovers_stale_lock(tmp_path, stage):
    bare_run(tmp_path, stage=stage)
    before = (tmp_path / "run.json").read_bytes()
    crashed = invoke(tmp_path, sys.executable, "--until-gen", 1,
                     crash=f"1:{stage}", timeout=10)
    assert crashed.returncode == 17, crashed.stderr
    assert (tmp_path / "run.json").read_bytes() == before
    assert (tmp_path / "loop.lock").is_file()
    (tmp_path / "run.json.tmp").write_text("interrupted metadata")
    resumed = invoke(tmp_path, sys.executable, "--until-gen", 1, timeout=10)
    assert resumed.returncode == 0, resumed.stderr
    state = read_run_json(tmp_path)
    assert (state["current_gen"], state["stage"]) == (2, "selfplay")
    assert not (tmp_path / "loop.lock").exists() and not list(tmp_path.rglob("*.tmp"))


def test_no_limit_keeps_advancing_until_interrupted(tmp_path, monkeypatch):
    bare_run(tmp_path)
    calls = []

    def work(run, engine, state):
        calls.append((state["current_gen"], state["stage"]))
        if state["current_gen"] == 2:
            raise KeyboardInterrupt
        return deepcopy(state)

    monkeypatch.setattr(loop, "run_stage", work)
    with pytest.raises(KeyboardInterrupt):
        loop.run_loop(arguments(tmp_path))
    assert calls == [(1, stage) for stage in REAL_STAGES] + [(2, "selfplay")]
    state = read_run_json(tmp_path)
    assert (state["current_gen"], state["stage"]) == (2, "selfplay")
    assert not (tmp_path / "loop.lock").exists()
