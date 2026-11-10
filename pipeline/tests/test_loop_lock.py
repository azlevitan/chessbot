"""Lock ownership, hard-crash recovery and competing starts; no engine needed."""

from contextlib import contextmanager, ExitStack
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from loop_lock import LoopLockError, run_lock, stage_lock
from run_layout import REPO_ROOT


def command(code, directory):
    return [sys.executable, "-c", "import sys; sys.path.insert(0, "
            f"{str(REPO_ROOT / 'pipeline')!r}); " + code, str(directory)]


def wait_for(path, process):
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
        assert process.poll() is None
        time.sleep(.01)
    assert path.exists()


@contextmanager
def held_lock(directory):
    ready = directory / "ready"
    code = ("from pathlib import Path; import time; from loop_lock import run_lock; "
            "exec(\"with run_lock(Path(sys.argv[1])):\\n "
            "Path(sys.argv[1], 'ready').touch()\\n time.sleep(20)\")")
    child = subprocess.Popen(command(code, directory))
    try:
        wait_for(ready, child)
        yield child
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGINT)
        child.wait(timeout=5)


def test_lock_publishes_own_pid_and_disappears(tmp_path):
    with run_lock(tmp_path):
        assert (tmp_path / "loop.lock").read_text().strip() == str(os.getpid())
    assert not (tmp_path / "loop.lock").exists()


@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
def test_context_failure_removes_lock(tmp_path, exception):
    with pytest.raises(exception):
        with run_lock(tmp_path):
            raise exception()
    assert not (tmp_path / "loop.lock").exists()


def test_live_dummy_pid_is_rejected_and_preserved(tmp_path):
    dummy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    path = tmp_path / "loop.lock"
    try:
        path.write_text(str(dummy.pid))
        with pytest.raises(LoopLockError, match=str(dummy.pid)):
            with run_lock(tmp_path):
                pytest.fail("a live dummy must block the loop")
        assert path.read_text() == str(dummy.pid)
    finally:
        dummy.terminate()
        dummy.wait(timeout=5)


def test_permission_denied_pid_counts_as_live(tmp_path, monkeypatch):
    path = tmp_path / "loop.lock"
    path.write_text("12345")

    def denied(*args):
        raise PermissionError("other user")

    monkeypatch.setattr(os, "kill", denied)
    with pytest.raises(LoopLockError, match="12345"):
        with run_lock(tmp_path):
            pytest.fail("unproven liveness must not be reclaimed")
    assert path.read_text() == "12345"


def test_pid_publication_failure_removes_owned_lock(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("fsync failed")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError, match="fsync failed"):
        with run_lock(tmp_path):
            pytest.fail("failed publication must not enter loop")
    assert not (tmp_path / "loop.lock").exists()


@pytest.mark.parametrize("contents", ["", "partial", "-1", "0", "9" * 50, "\N{SNOWMAN}"])
def test_incomplete_pid_publication_is_reclaimed(tmp_path, contents):
    path = tmp_path / "loop.lock"
    path.write_text(contents)
    with run_lock(tmp_path):
        assert path.read_text().strip() == str(os.getpid())
    assert not path.exists()


def test_hard_crash_leaves_a_stale_lock_that_is_reclaimed(tmp_path):
    code = ("from pathlib import Path; import os; from loop_lock import run_lock; "
            "exec(\"with run_lock(Path(sys.argv[1])):\\n os._exit(17)\")")
    result = subprocess.run(command(code, tmp_path), check=False, timeout=5)
    assert result.returncode == 17
    assert (tmp_path / "loop.lock").exists()
    with run_lock(tmp_path):
        assert (tmp_path / "loop.lock").read_text().strip() == str(os.getpid())
    assert not (tmp_path / "loop.lock").exists()


def test_second_start_rejects_a_real_lock_owner(tmp_path):
    with held_lock(tmp_path) as owner:
        with pytest.raises(LoopLockError, match=str(owner.pid)):
            with run_lock(tmp_path):
                pytest.fail("second loop entered")
        assert (tmp_path / "loop.lock").read_text().strip() == str(owner.pid)
    assert not (tmp_path / "loop.lock").exists()


def test_cleanup_preserves_replaced_lock(tmp_path):
    path = tmp_path / "loop.lock"
    with run_lock(tmp_path):
        replacement = tmp_path / "replacement"
        replacement.write_text("12345")
        replacement.replace(path)
    assert path.read_text() == "12345"


def test_cleanup_preserves_changed_owner_on_same_inode(tmp_path):
    path = tmp_path / "loop.lock"
    with run_lock(tmp_path):
        path.write_text("12345")
    assert path.read_text() == "12345"


def test_symbolic_link_lock_is_rejected(tmp_path):
    target = tmp_path / "unrelated"
    target.write_text("partial")
    (tmp_path / "loop.lock").symlink_to(target)
    with pytest.raises(LoopLockError, match="symbolic-link"):
        with run_lock(tmp_path):
            pytest.fail("symlink locks must be left intact")
    assert target.read_text() == "partial"


def test_simultaneous_starts_have_exactly_one_owner(tmp_path):
    # All contenders wait for one barrier, then the winner holds its lock until
    # every loser has reported. Holding the winner avoids serial successes.
    code = """
import sys, time
from pathlib import Path
from loop_lock import LoopLockError, run_lock
root = Path(sys.argv[1])
while not (root / 'go').exists():
    time.sleep(.005)
try:
    with run_lock(root):
        (root / ('winner_' + str(__import__('os').getpid()))).touch()
        while not (root / 'release').exists():
            time.sleep(.005)
except LoopLockError:
    (root / ('loser_' + str(__import__('os').getpid()))).touch()
"""
    children = [subprocess.Popen(command(code, tmp_path)) for _ in range(6)]
    try:
        (tmp_path / "go").touch()
        deadline = time.monotonic() + 5
        while len(list(tmp_path.glob("loser_*"))) < 5 and time.monotonic() < deadline:
            time.sleep(.01)
        assert len(list(tmp_path.glob("winner_*"))) == 1
        assert len(list(tmp_path.glob("loser_*"))) == 5
    finally:
        (tmp_path / "release").touch()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
    assert all(child.returncode == 0 for child in children)
    assert not (tmp_path / "loop.lock").exists()


def test_worker_lease_blocks_stale_takeover_until_worker_finishes(tmp_path, monkeypatch):
    path = tmp_path / "loop.lock"
    path.write_text(str(os.getpid()))

    def dead(*args):
        raise ProcessLookupError()

    with stage_lock(tmp_path, os.getpid()):
        monkeypatch.setattr(os, "kill", dead)
        with pytest.raises(LoopLockError, match="previous stage still running"):
            with run_lock(tmp_path):
                pytest.fail("a surviving worker still owns the files")
        assert path.read_text() == str(os.getpid())
    with run_lock(tmp_path):
        assert path.read_text().strip() == str(os.getpid())
    assert not path.exists()


def test_late_worker_cannot_join_replacement_loop(tmp_path):
    path = tmp_path / "loop.lock"
    path.write_text(str(os.getpid()))
    with pytest.raises(LoopLockError, match="parent no longer owns"):
        with stage_lock(tmp_path, os.getpid() + 1):
            pytest.fail("worker belongs to the previous loop")
    assert path.read_text() == str(os.getpid())


def test_parent_cleanup_preserves_active_worker_lease(tmp_path):
    path = tmp_path / "loop.lock"
    with ExitStack() as worker:
        with run_lock(tmp_path):
            worker.enter_context(stage_lock(tmp_path, os.getpid()))
        assert path.exists()
    assert path.exists()
    path.unlink()
