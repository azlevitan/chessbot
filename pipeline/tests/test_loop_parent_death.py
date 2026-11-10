"""A hard parent crash must end its isolated stage before stale-lock takeover."""

import os
import signal
import subprocess
import sys

import pytest

from run_layout import read_run_json, write_run_json
from test_loop import bare_run, command, invoke
from test_loop_signals import alive, process_ids, wait_for


@pytest.mark.parametrize("ignore", [False, True])
def test_parent_death_stops_stage_before_restart_can_reclaim_lock(tmp_path, ignore):
    run = tmp_path / "run"
    bare_run(run, current_gen=2,
             config={"selfplay": {"games": 2, "games_per_chunk": 2}})
    before = (run / "run.json").read_bytes()
    ids_file, ready = tmp_path / "pids.json", tmp_path / "child-ready"
    engine = tmp_path / "waiting-engine"
    child_code = (
        "import signal, time; from pathlib import Path; "
        + ("signal.signal(signal.SIGINT, signal.SIG_IGN); "
           "signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore else "")
        + f"Path({str(ready)!r}).touch(); time.sleep(60)"
    )
    engine.write_text(
        f"#!{sys.executable}\n"
        "import json, os, signal, subprocess, sys, time\n"
        "from pathlib import Path\n"
        + ("signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
           "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore else "")
        + f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        + f"while not Path({str(ready)!r}).exists(): time.sleep(0.01)\n"
        + f"Path({str(ids_file)!r}).write_text(json.dumps([os.getpid(), child.pid, os.getppid()]))\n"
        + "try:\n    child.wait()\nfinally:\n    child.kill()\n    child.wait()\n"
    )
    engine.chmod(0o755)
    env = os.environ.copy()
    env.pop("CHESSBOT_CRASH_AT", None)
    parent = subprocess.Popen(command(run, engine, "--until-gen", 2), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ids = []
    try:
        ids = wait_for(lambda: process_ids(ids_file))
        assert all(alive(pid) for pid in ids)
        parent.kill()
        assert parent.wait(timeout=5) == -signal.SIGKILL
        # A restart races orphan cleanup: it may report busy, or acquire only
        # once the old stage is gone. The earlier target prevents new work.
        early = invoke(run, engine, "--until-gen", 1, timeout=10)
        assert early.returncode in (0, 1), early.stdout + early.stderr
        assert (run / "run.json").read_bytes() == before
        wait_for(lambda: all(not alive(pid) for pid in ids))
        state = read_run_json(run)
        state["stage"] = "done"
        write_run_json(run, state)
        resumed = invoke(run, engine, "--until-gen", 2, timeout=10)
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        state = read_run_json(run)
        assert (state["current_gen"], state["stage"]) == (3, "selfplay")
        assert not (run / "loop.lock").exists()
        assert not list(run.rglob("*.tmp"))
    finally:
        if parent.poll() is None:
            parent.send_signal(signal.SIGINT)
            try:
                parent.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                parent.kill()
        ids = process_ids(ids_file) or ids
        for pid in ids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        parent.communicate(timeout=5)
