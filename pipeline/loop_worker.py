"""Isolated stage execution; the parent alone owns locks and run checkpoints.

The result travels through an unlinked temporary file descriptor, keeping stage
logs on stdout and leaving no extra metadata files after a hard parent crash.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading

from config import normalize_config
from loop_lock import LoopLockError, stage_lock
from run_layout import read_run_json


def _signal_group(process, signum) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _stop(process) -> None:
    # A second Ctrl-C must not interrupt cleanup and leave inference running.
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        _signal_group(process, signal.SIGINT)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # The leader may already have exited while a grandchild remains.
            _signal_group(process, signal.SIGKILL)
            process.wait()
    finally:
        signal.signal(signal.SIGINT, previous)


@contextmanager
def _lifeline():
    reader, writer = os.pipe()
    try:
        yield reader
    finally:
        os.close(reader)
        os.close(writer)


def _watch_parent(descriptor: int) -> None:
    # Only the parent owns the write end. Even SIGKILL closes it, so an orphan
    # cannot continue training or writing chunks after the loop disappears.
    while os.read(descriptor, 1):
        pass
    os.killpg(os.getpgrp(), signal.SIGKILL)


def run_stage(run: Path, engine: Path, state: dict) -> dict:
    """Run one stage and return proposed state, without modifying run.json."""
    command = [sys.executable, str(Path(__file__).resolve()), str(run), str(engine)]
    with tempfile.TemporaryFile(mode="w+") as result, _lifeline() as lifeline:
        process = None
        # Defer SIGINT across Popen so even an interrupt during spawn has an
        # owned process to reap. The worker unblocks it before doing any work.
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        try:
            try:
                process = subprocess.Popen(
                    [*command, str(result.fileno()), str(lifeline), str(os.getpid())],
                    pass_fds=(result.fileno(), lifeline), start_new_session=True,
                )
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous)
            status = process.wait()
            if status:
                raise subprocess.CalledProcessError(status, command)
            result.seek(0)
            proposed = json.load(result)
            if not isinstance(proposed, dict):
                raise ValueError(f"{state['stage']}: worker returned invalid run state")
            return proposed
        except BaseException:
            if process is not None:
                _stop(process)
            raise


def main() -> int:
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
    import stages  # stdlib stages; training libraries stay in their subprocesses

    run, engine, descriptor, lifeline, parent = sys.argv[1:]
    # The watcher inherits blocked SIGINT; only the main thread handles Ctrl-C,
    # including the short masked ownership transactions in stage_lock.
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        threading.Thread(target=_watch_parent, args=(int(lifeline),), daemon=True).start()
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)
    try:
        with stage_lock(run, int(parent)):
            state = read_run_json(run)
            cfg = normalize_config(state["config"])
            cfg["engine"] = engine
            stage = state["stage"]
            if stage not in ("selfplay", "train", "export", "match", "decide", "anchors"):
                raise ValueError(f"unsupported worker stage: {stage}")
            getattr(stages, stage)(run, state["current_gen"], cfg, state)
            with os.fdopen(int(descriptor), "w", encoding="utf-8") as result:
                json.dump(state, result, allow_nan=False)
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, LoopLockError, subprocess.SubprocessError) as error:
        print(f"loop stage: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
