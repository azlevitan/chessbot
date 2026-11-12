"""Real child accounting, cancellation and the uninstrumented compatibility path."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from process_metrics import measured_run, own_peak_rss_bytes, rss_bytes
from telemetry import read_attempts


@pytest.fixture
def metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("CHESSBOT_METRICS_DIR", str(tmp_path / "telemetry"))
    monkeypatch.setenv("CHESSBOT_STAGE", "selfplay")
    monkeypatch.setenv("CHESSBOT_STAGE_ATTEMPT", "parent-span")
    return tmp_path


def test_units_are_normalized_without_guessing_from_magnitude():
    assert rss_bytes(1234, "darwin") == 1234
    assert rss_bytes(1234, "linux") == 1234 * 1024
    assert own_peak_rss_bytes() > 0


def test_uninstrumented_call_preserves_original_run_interface(tmp_path, monkeypatch):
    monkeypatch.delenv("CHESSBOT_METRICS_DIR", raising=False)
    observed = []
    sentinel = object()
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        observed.append((args, kwargs)) or sentinel)
    command = ["engine", "match"]
    assert measured_run(command, unit="block", check=True, cwd=tmp_path) is sentinel
    assert observed == [((command,), {"check": True, "cwd": tmp_path})]


def test_direct_process_peak_and_output_are_preserved(metrics):
    output = metrics / "output.txt"
    code = "x = bytearray(24 * 1024 * 1024); print('complete')"
    with output.open("w") as stream:
        result = measured_run([sys.executable, "-c", code], unit="chunk_000000",
                              check=True, stdout=stream, cwd=metrics)
    assert result.returncode == 0
    assert output.read_text().strip() == "complete"
    record, = read_attempts(metrics)
    assert record["status"] == "complete"
    assert record["peak_rss_bytes"] >= 24 * 1024 * 1024
    assert record["seconds"] > 0
    assert record["stage_attempt"] == "parent-span"
    assert record["resource_method"] == "wait4"
    assert record["pid"] > 0
    assert not list(metrics.rglob("*.tmp"))


@pytest.mark.parametrize("check", [False, True])
def test_failure_exit_status_and_measurements_are_retained(metrics, check):
    command = [sys.executable, "-c", "raise SystemExit(7)"]
    if check:
        with pytest.raises(subprocess.CalledProcessError) as error:
            measured_run(command, unit="failure", check=True)
        assert error.value.returncode == 7
        assert error.value.cmd == command
    else:
        assert measured_run(command, unit="failure").returncode == 7
    record, = read_attempts(metrics)
    assert record["status"] == "failed"
    assert record["exit_code"] == 7
    assert record["peak_rss_bytes"] > 0


def test_spawn_failure_is_not_misrepresented_as_zero_memory(metrics):
    with pytest.raises(FileNotFoundError):
        measured_run([str(metrics / "missing-program")], unit="spawn", check=True)
    record, = read_attempts(metrics)
    assert record["status"] == "failed"
    assert record["peak_rss_bytes"] is None
    assert record["rss_required"] is False
    assert record["error_type"] == "FileNotFoundError"


def test_timeout_reaps_child_and_keeps_resource_measurement(metrics):
    with pytest.raises(subprocess.TimeoutExpired):
        measured_run([sys.executable, "-c", "import time; time.sleep(30)"],
                     unit="timeout", timeout=0.1, stderr=subprocess.DEVNULL)
    record, = read_attempts(metrics)
    assert record["status"] == "failed"
    assert record["peak_rss_bytes"] > 0
    with pytest.raises(ProcessLookupError):
        os.kill(record["pid"], 0)


def test_sigint_reaps_direct_child_and_records_interrupted_attempt(tmp_path):
    pid_file = tmp_path / "child.pid"
    command = [sys.executable, "-c", (
        "import os,time; from pathlib import Path; "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)")]
    env = {**os.environ, "CHESSBOT_METRICS_DIR": str(tmp_path / "telemetry"),
           "CHESSBOT_STAGE": "train", "CHESSBOT_STAGE_ATTEMPT": "span",
           "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    runner = subprocess.Popen([sys.executable, "-c", (
        "from process_metrics import measured_run; "
        f"measured_run({command!r}, unit='train', check=True)")],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_file.exists()
        runner.send_signal(signal.SIGINT)
        runner.wait(timeout=3)
        record, = read_attempts(tmp_path)
        assert record["status"] == "interrupted"
        assert record["peak_rss_bytes"] > 0
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)
    finally:
        if runner.poll() is None:
            runner.kill()
        runner.wait()
