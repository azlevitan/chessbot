"""Real SIGINT delivery to the loop with active engine descendants."""

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from test_loop import bare_run, command


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A killed orphan can briefly remain a zombie under a container's PID 1.
    status = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                            capture_output=True, text=True, check=False)
    return bool(status.stdout.strip()) and not status.stdout.lstrip().startswith("Z")


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("timed out waiting for process readiness/termination")


def process_ids(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


@pytest.mark.parametrize("ignore", [False, True])
def test_sigint_stops_engine_and_grandchild_preserving_checkpoint(tmp_path, ignore):
    run = tmp_path / "run"
    bare_run(run)
    before = (run / "run.json").read_bytes()
    ids_file = tmp_path / "pids.json"
    grandchild_ready = tmp_path / "grandchild-ready"
    engine = tmp_path / "waiting-engine"
    child_code = (
        "import os, signal, time; from pathlib import Path; "
        + ("signal.signal(signal.SIGINT, signal.SIG_IGN); "
           "signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore else "")
        + f"Path({str(grandchild_ready)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    engine.write_text(
        f"#!{sys.executable}\n"
        "import json, os, signal, subprocess, sys, time\n"
        "from pathlib import Path\n"
        + ("signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
           "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore else "")
        + f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        + f"while not Path({str(grandchild_ready)!r}).exists(): time.sleep(0.01)\n"
        + f"Path({str(ids_file)!r}).write_text(json.dumps([os.getpid(), child.pid]))\n"
        + "try:\n    child.wait()\nfinally:\n    child.kill()\n    child.wait()\n"
    )
    engine.chmod(0o755)
    env = os.environ.copy()
    env.pop("CHESSBOT_CRASH_AT", None)
    process = subprocess.Popen(command(run, engine, "--generations", 1), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ids = []
    try:
        ids = wait_for(lambda: process_ids(ids_file))
        assert all(alive(pid) for pid in ids)
        assert (run / "loop.lock").read_text().strip() == str(process.pid)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 130, stdout + stderr
        wait_for(lambda: all(not alive(pid) for pid in ids))
        assert (run / "run.json").read_bytes() == before
        assert not (run / "loop.lock").exists()
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        process.communicate(timeout=5)
        ids = process_ids(ids_file) or ids
        for pid in ids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
