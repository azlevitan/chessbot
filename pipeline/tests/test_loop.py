"""Real ci_tiny crash/resume acceptance for the Phase 9.3 loop."""

from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_layout import (
    REPO_ROOT, generation_dir, read_json, read_run_json, write_run_json,
)


def command(run, engine, *options):
    return [sys.executable, str(REPO_ROOT / "pipeline/loop.py"),
            "--run", str(run), "--engine", str(engine), *map(str, options)]


def invoke(run, engine, *options, crash=None, timeout=180):
    env = os.environ.copy()
    env.pop("CHESSBOT_CRASH_AT", None)
    if crash is not None:
        env["CHESSBOT_CRASH_AT"] = crash
    return subprocess.run(command(run, engine, *options), env=env, cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=timeout)


def bare_run(directory, **updates):
    state = {"config": {}, "best_gen": 0, "current_gen": 1,
             "stage": "selfplay", "resign_threshold": -0.95}
    state.update(deepcopy(updates))
    write_run_json(directory, state)
    return read_run_json(directory)


def stamps(paths):
    return {path: path.stat().st_mtime_ns for path in paths}


def test_crash_after_train_resumes_through_generation_two(
        ci_tiny_bootstrap, engine_bin, tmp_path):
    run = tmp_path / "run"
    shutil.copytree(ci_tiny_bootstrap, run)
    original_config = deepcopy(read_run_json(run)["config"])
    crashed = invoke(run, engine_bin, "--until-gen", 2, crash="1:train")
    assert crashed.returncode == 17, crashed.stdout + crashed.stderr
    interrupted = read_run_json(run)
    assert interrupted["current_gen"] == 1 and interrupted["stage"] == "train"
    assert interrupted["best_gen"] == 0
    assert (run / "loop.lock").is_file()
    one = generation_dir(run, 1)
    assert (one / "model.pt").is_file() and (one / "train.json").is_file()
    completed = stamps([one / "model.pt", one / "train.json",
                        *(one / "selfplay").glob("chunk_*.*")])
    resumed = invoke(run, engine_bin, "--until-gen", 2)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    state = read_run_json(run)
    assert state["current_gen"] == 3 and state["stage"] == "selfplay"
    assert state["config"] == original_config and "engine" not in state["config"]
    assert stamps(completed) == completed
    promoted = [0]
    for generation in (1, 2):
        directory = generation_dir(run, generation)
        decision = read_json(directory / "decision.json")
        if decision["promoted"]:
            promoted.append(generation)
        binaries = sorted((directory / "selfplay").glob("chunk_*.bin"))
        sidecars = sorted((directory / "selfplay").glob("chunk_*.json"))
        assert [path.name for path in binaries] == ["chunk_000000.bin", "chunk_000001.bin"]
        assert len(sidecars) == 2
        for binary in binaries:
            metadata = read_json(binary.with_suffix(".json"))
            assert metadata["games"] == 2
            assert binary.stat().st_size == metadata["samples"] * 492
    assert state["best_gen"] == max(promoted)
    assert not list(run.rglob("*.tmp")) and not (run / "loop.lock").exists()
    assert not generation_dir(run, 3).exists()
    # An already reached inclusive bound must leave the checkpoint untouched.
    before = (run / "run.json").read_bytes(), (run / "run.json").stat().st_mtime_ns
    repeated = invoke(run, engine_bin, "--until-gen", 2)
    assert repeated.returncode == 0, repeated.stderr
    assert before == ((run / "run.json").read_bytes(), (run / "run.json").stat().st_mtime_ns)


def test_cli_rejects_an_existing_live_pid(tmp_path):
    run = tmp_path / "locked"
    bare_run(run)
    lock = run / "loop.lock"
    lock.write_text(str(os.getpid()) + "\n")
    before = (run / "run.json").read_bytes()
    result = invoke(run, Path(sys.executable), "--until-gen", 1, timeout=10)
    assert result.returncode != 0
    assert lock.read_text() == str(os.getpid()) + "\n"
    assert (run / "run.json").read_bytes() == before
