"""Fixtures: build the Rust engine and dump the contract test vectors."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = REPO_ROOT / "engine"
ENGINE_BIN = ENGINE_DIR / "target" / "release" / "engine"


@pytest.fixture(scope="session")
def engine_bin() -> Path:
    """Build the release engine binary (fails, never skips, if cargo is missing)."""
    subprocess.run(
        ["cargo", "build", "--release", "--manifest-path", str(ENGINE_DIR / "Cargo.toml")],
        check=True,
    )
    assert ENGINE_BIN.is_file(), f"engine binary not found at {ENGINE_BIN}"
    return ENGINE_BIN


@pytest.fixture(scope="session")
def vectors_path(engine_bin: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run `engine dump-test-vectors` into a temp dir and return the .bin path."""
    out = tmp_path_factory.mktemp("vectors") / "test_vectors.bin"
    subprocess.run([str(engine_bin), "dump-test-vectors", "--out", str(out)], check=True)
    return out
