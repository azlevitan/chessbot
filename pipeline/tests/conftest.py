"""Repository paths and the real release engine for pipeline integration tests."""

import subprocess

import pytest

from run_layout import ENGINE_BIN, ENGINE_DIR, REPO_ROOT


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


@pytest.fixture(scope="session")
def engine_bin():
    subprocess.run(["cargo", "build", "--release", "--manifest-path",
                    str(ENGINE_DIR / "Cargo.toml")], check=True)
    assert ENGINE_BIN.is_file(), f"engine binary not found at {ENGINE_BIN}"
    return ENGINE_BIN
