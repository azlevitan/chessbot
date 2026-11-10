"""PID lock for one resumable loop; acquisition is serialized on the run directory.

The directory advisory lock is held only while inspecting or changing loop.lock.
It avoids unlink/open races on a lock file that must disappear on clean exit.
This helper targets the project's Linux and macOS hosts.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal


class LoopLockError(RuntimeError):
    """Another process owns the run, or its lock cannot be safely inspected."""


@contextmanager
def _serialized(directory):
    # Defer Ctrl-C until the PID file and our ownership record agree.
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        fcntl.flock(directory, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(directory, fcntl.LOCK_UN)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _identity(path):
    info = path.lstat()
    return info.st_dev, info.st_ino


def _stage_running(path):
    """A worker's shared lease survives abrupt death of its loop parent."""
    try:
        stream = path.open("rb")
    except FileNotFoundError:
        return False
    with stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


@contextmanager
def stage_lock(run_dir: str | Path, parent_pid: int):
    """Keep the parent's lock unreclaimable until this worker finishes.

    Check ownership while serializing against stale-lock takeover, so a late
    worker from an old loop cannot join a replacement loop's lock file.
    """
    path = Path(run_dir) / "loop.lock"
    directory = os.open(run_dir, os.O_RDONLY | os.O_DIRECTORY)
    stream = None
    try:
        with _serialized(directory):
            if path.is_symlink():
                raise LoopLockError(f"{path}: refusing a symbolic-link lock")
            stream = path.open("r", encoding="ascii")
            if stream.read().strip() != str(parent_pid):
                raise LoopLockError(f"{path}: stage parent no longer owns the run")
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield
    finally:
        if stream is not None:
            stream.close()
        os.close(directory)


@contextmanager
def run_lock(run_dir: str | Path):
    """Reject a live owner, reclaim stale PID files, and remove only our lock.

    Empty/malformed files can be left by a crash during PID publication. All
    cooperating writers hold the directory lock, so none can still be writing
    such a file while it is inspected here. Unknown PID liveness fails closed.
    """
    run_dir = Path(run_dir)
    path = run_dir / "loop.lock"
    pid = os.getpid()
    owned = None
    published = False
    directory = os.open(run_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with _serialized(directory):
            if path.is_symlink():
                raise LoopLockError(f"{path}: refusing a symbolic-link lock")
            try:
                contents = path.read_text(encoding="ascii").strip()
            except FileNotFoundError:
                contents = ""
            except UnicodeError:
                contents = ""
            try:
                owner = int(contents)
            except ValueError:
                owner = 0
            # Python rejects integers outside pid_t; those cannot name an owner.
            try:
                alive = owner > 0 and _alive(owner)
            except OverflowError:
                alive = False
            if alive:
                raise LoopLockError(f"{path}: loop already running with PID {owner}")
            if _stage_running(path):
                raise LoopLockError(f"{path}: previous stage still running; retry")
            path.unlink(missing_ok=True)
            with path.open("x", encoding="ascii") as stream:
                info = os.fstat(stream.fileno())
                owned = info.st_dev, info.st_ino
                stream.write(f"{pid}\n")
                stream.flush()
                os.fsync(stream.fileno())
                published = True
        yield
    finally:
        try:
            if owned is not None:
                with _serialized(directory):
                    try:
                        ours = (_identity(path) == owned and not path.is_symlink()
                                and (not published or path.read_text(
                                    encoding="ascii").strip() == str(pid)))
                    except (FileNotFoundError, UnicodeError):
                        ours = False
                    if ours and not _stage_running(path):
                        path.unlink()
        finally:
            os.close(directory)
