"""Standalone pruning preview, locking and failure behavior with fake run data."""

import os
import subprocess
import sys

from run_layout import REPO_ROOT, read_run_json, write_run_json
from test_prune import candidate, settings, sized, snapshot, usage


def saved_run(run, *, budget=20):
    state, cfg = settings(budget=budget)
    write_run_json(run, {**state, "config": cfg, "stage": "prune",
                         "resign_threshold": -0.95})
    return state


def invoke(run, *options):
    return subprocess.run([sys.executable, str(REPO_ROOT / "pipeline/prune.py"),
                           "--run", str(run), *options], capture_output=True,
                          text=True, timeout=10, cwd=REPO_ROOT)


def targets(output, prefix):
    return [line.removeprefix(prefix) for line in output.splitlines() if line.startswith(prefix)]


def test_dry_run_matches_live_targets_and_changes_no_files_or_checkpoint(tmp_path):
    saved_run(tmp_path)
    old = candidate(tmp_path, 1, promoted=False)
    current = candidate(tmp_path, 6, promoted=False)
    expired = old / "selfplay"
    sized(expired / "chunk_000000.bin")
    before = snapshot(tmp_path)
    state = read_run_json(tmp_path)
    preview = invoke(tmp_path, "--dry-run")
    assert preview.returncode == 0, preview.stderr
    assert snapshot(tmp_path) == before
    assert not (tmp_path / "loop.lock").exists()
    expected = [str(expired), str(old / "model.pt"), str(old / "model.onnx")]
    assert targets(preview.stdout, "would remove ") == expected
    assert "bytes:" in preview.stdout and "budget:" in preview.stdout
    live = invoke(tmp_path)
    assert live.returncode == 0, live.stderr
    assert targets(live.stdout, "removed ") == expected
    assert all(not path.exists() for path in (expired, old / "model.pt", old / "model.onnx"))
    assert (current / "model.pt").is_file() and (current / "model.onnx").is_file()
    assert read_run_json(tmp_path) == state
    assert not (tmp_path / "loop.lock").exists()
    repeated = invoke(tmp_path)
    assert repeated.returncode == 0, repeated.stderr
    assert targets(repeated.stdout, "removed ") == []


def test_live_cli_refuses_live_lock_and_preview_does_not_modify_it(tmp_path):
    saved_run(tmp_path)
    old = candidate(tmp_path, 2, promoted=False)
    lock = tmp_path / "loop.lock"
    lock.write_text(f"{os.getpid()}\n")
    before = snapshot(tmp_path)
    result = invoke(tmp_path)
    assert result.returncode != 0 and "running" in result.stderr
    assert snapshot(tmp_path) == before
    preview = invoke(tmp_path, "--dry-run")
    assert preview.returncode == 0, preview.stderr
    assert str(old / "model.pt") in preview.stdout
    assert snapshot(tmp_path) == before


def test_unreachable_dry_run_is_read_only_and_live_failure_releases_lock(tmp_path):
    saved_run(tmp_path, budget=1e-9)
    rejected = candidate(tmp_path, 2, promoted=False)
    sized(tmp_path / "supervised/chunk_000000.bin", 4000)
    before = snapshot(tmp_path)
    preview = invoke(tmp_path, "--dry-run")
    assert preview.returncode != 0 and "budget" in preview.stderr
    assert snapshot(tmp_path) == before
    assert str(rejected / "model.pt") in preview.stdout
    result = invoke(tmp_path)
    assert result.returncode != 0 and "budget" in result.stderr
    assert not (rejected / "model.pt").exists()
    assert not (rejected / "model.onnx").exists()
    assert not (tmp_path / "loop.lock").exists()
    assert read_run_json(tmp_path)["stage"] == "prune"
    assert (tmp_path / "supervised/chunk_000000.bin").stat().st_size == 4000


def test_invalid_saved_state_fails_before_deleting_anything(tmp_path):
    saved_run(tmp_path)
    candidate(tmp_path, 2, promoted=False)
    state = read_run_json(tmp_path)
    state["best_gen"] = state["current_gen"] + 1
    write_run_json(tmp_path, state)
    before = snapshot(tmp_path)
    result = invoke(tmp_path)
    assert result.returncode != 0
    assert snapshot(tmp_path) == before
    assert not (tmp_path / "loop.lock").exists()


def test_help_runs_without_site_packages():
    result = subprocess.run([sys.executable, "-S", str(REPO_ROOT / "pipeline/prune.py"),
                             "--help"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--dry-run" in result.stdout and "--run" in result.stdout


def test_exact_persistent_budget_matches_preview_and_live_with_transient_lock(tmp_path):
    saved_run(tmp_path, budget=0.00001)
    historical = candidate(tmp_path, 0)
    sized(tmp_path / "padding.bin", 10000 - usage(tmp_path))
    before = snapshot(tmp_path)
    preview = invoke(tmp_path, "--dry-run")
    assert preview.returncode == 0, preview.stderr
    assert targets(preview.stdout, "would remove ") == []
    assert "bytes: 10000 -> 10000; budget: 10000" in preview.stdout
    assert snapshot(tmp_path) == before
    live = invoke(tmp_path)
    assert live.returncode == 0, live.stderr
    assert targets(live.stdout, "removed ") == []
    assert "bytes: 10000 -> 10000; budget: 10000" in live.stdout
    assert snapshot(tmp_path) == before
    assert (historical / "model.pt").is_file()
    assert not (tmp_path / "loop.lock").exists()
