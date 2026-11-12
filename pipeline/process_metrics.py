"""Checked subprocess calls with direct-child POSIX wait4 resource accounting.

No metrics environment means the original subprocess.run call is used unchanged.
Measured children inherit the worker's process group and parent-death handling.
"""

from __future__ import annotations

import os
import resource
import signal
import subprocess
import sys
import time

from run_layout import write_json_atomic
from telemetry import begin_attempt, finish_attempt


def rss_bytes(value, platform=None):
    """Darwin reports bytes; Linux and other supported POSIX hosts report KiB."""
    return int(value) * (1 if (platform or sys.platform) == "darwin" else 1024)


def own_peak_rss_bytes():
    return rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _wait(process, timeout=None):
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        try:
            pid, status, usage = os.wait4(process.pid, 0 if deadline is None else os.WNOHANG)
        except InterruptedError:
            continue
        if pid:
            process.returncode = os.waitstatus_to_exitcode(status)
            return usage
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired(process.args, timeout)
        time.sleep(0.01)


def _stop(process):
    """Reap a child without Popen.poll consuming its resource accounting first."""
    try:
        os.kill(process.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        return _wait(process, 0.5)
    except subprocess.TimeoutExpired:
        try:
            os.kill(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return _wait(process)


def measured_run(command, *, unit, **kwargs):
    """Run an inherited-output command and persist its exact process peak.

    The measured mode supports the launch options used by the pipeline (cwd,
    environment and inherited/file stdout/stderr). Capturing pipes, input and
    shell commands are intentionally rejected: buffering belongs to a caller.
    """
    directory = os.environ.get("CHESSBOT_METRICS_DIR")
    if not directory:
        return subprocess.run(command, **kwargs)
    unsupported = (kwargs.get("capture_output") or kwargs.get("input") is not None
                   or kwargs.get("stdout") == subprocess.PIPE or kwargs.get("stderr") == subprocess.PIPE
                   or kwargs.get("shell"))
    if unsupported:
        raise ValueError("measured_run requires direct commands with inherited or file output")
    options = dict(kwargs)
    check = options.pop("check", False)
    timeout = options.pop("timeout", None)
    options.pop("capture_output", None)
    options.pop("input", None)
    path, record = begin_attempt(
        directory, kind="process", stage=os.environ.get("CHESSBOT_STAGE", "unknown"),
        stage_attempt=os.environ.get("CHESSBOT_STAGE_ATTEMPT"), unit=str(unit),
        command=list(map(str, command)), pid=None, rss_required=False,
        resource_method="wait4", resource_platform=sys.platform)
    started, process, usage = time.monotonic(), None, None
    status, exception = "failed", None
    try:
        process = subprocess.Popen(command, **options)
        record.update(pid=process.pid, rss_required=True)
        write_json_atomic(path, record)
        usage = _wait(process, timeout)
        status = "complete" if process.returncode == 0 else "failed"
        if check and process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)
        return subprocess.CompletedProcess(command, process.returncode)
    except BaseException as error:
        exception = error
        if isinstance(error, KeyboardInterrupt):
            status = "interrupted"
        if process is not None and process.returncode is None:
            previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                usage = _stop(process)
            finally:
                signal.signal(signal.SIGINT, previous)
        raise
    finally:
        finish_attempt(path, record, status, time.monotonic() - started,
                       peak_rss_bytes=rss_bytes(usage.ru_maxrss) if usage is not None else None,
                       exit_code=process.returncode if process is not None else None,
                       error_type=type(exception).__name__ if exception is not None else None)
