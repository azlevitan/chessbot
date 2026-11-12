"""Repository paths and the real release engine for pipeline integration tests."""

import subprocess
import sys

import pytest

from run_layout import ENGINE_BIN, ENGINE_DIR, REPO_ROOT


@pytest.fixture(scope="session", autouse=True)
def bounded_cpu_threads():
    """Keep temporary CPU training/export jobs from competing for excess threads."""
    with pytest.MonkeyPatch.context() as patch:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            patch.setenv(name, "1")
        yield


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def engine_bin():
    subprocess.run(["cargo", "build", "--release", "--manifest-path",
                    str(ENGINE_DIR / "Cargo.toml")], check=True)
    assert ENGINE_BIN.is_file(), f"engine binary not found at {ENGINE_BIN}"
    return ENGINE_BIN


@pytest.fixture(scope="session")
def ci_tiny_bootstrap(engine_bin, tmp_path_factory):
    """One real CPU bootstrap shared by resumable-stage integration tests."""
    run = tmp_path_factory.mktemp("ci-tiny-bootstrap") / "run"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "pipeline/bootstrap.py"),
         "--pgn", str(REPO_ROOT / "training/tests/fixtures/sample.pgn"),
         "--run", str(run), "--config", str(REPO_ROOT / "pipeline/configs/ci_tiny.json"),
         "--val-games", "1", "--steps", "20", "--batch-size", "16", "--device", "cpu"],
        check=True, timeout=180,
    )
    return run
