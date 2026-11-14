"""Locked backend changes preserve the original state and completed data."""

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from loop_lock import LoopLockError, run_lock
from run_layout import read_run_json, write_json_atomic, write_run_json
from set_inference_batch import main, update_run, verify_saved


@pytest.fixture
def saved_run(tmp_path):
    run = tmp_path / "run"
    state = {"current_gen": 1, "best_gen": 0, "stage": "selfplay",
             "resign_threshold": -.95, "extra": {"keep": [1, 2]},
             "config": {"seed": 12, "selfplay": {"games": 2}, "match": {"max_games": 4}}}
    write_run_json(run, state)
    directory = run / "gens/gen_0001/selfplay"
    directory.mkdir(parents=True)
    for index, samples in enumerate((1, 3)):
        binary = directory / f"chunk_{index:06d}.bin"
        binary.write_bytes(bytes([index + 1]) * (samples * 492))
        write_json_atomic(binary.with_suffix(".json"), {
            "format": "chessbot-chunk-v1", "kind": "selfplay", "samples": samples})
    return run


def snapshots(run):
    return {str(path.relative_to(run)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in run.rglob("*") if path.is_file()}


def test_change_preserves_other_settings_and_exact_backup_and_chunk_evidence(saved_run, tmp_path):
    before = snapshots(saved_run)
    original = read_run_json(saved_run)
    expected = deepcopy(original)
    for section in ("selfplay", "match"):
        expected["config"][section]["infer_batch"] = 32
    output = tmp_path / "evidence"

    assert update_run(saved_run, 32, output) == output
    assert read_run_json(saved_run) == expected
    assert (output / "run.before.json").read_bytes() == before["run.json"][0]
    assert not (saved_run / "loop.lock").exists()
    after = snapshots(saved_run)
    assert set(after) == set(before)
    assert {k: v for k, v in after.items() if k != "run.json"} == {
        k: v for k, v in before.items() if k != "run.json"}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["complete_chunks"] == 2
    assert manifest["generation"] == 1 and manifest["infer_batch"] == 32
    assert len(manifest["files"]) == 4
    assert manifest["skipped_incomplete_chunks"] == []
    unchanged = snapshots(saved_run)
    assert verify_saved(saved_run, output) == 2
    assert snapshots(saved_run) == unchanged


def test_existing_loop_owner_blocks_changes_and_verification(saved_run, tmp_path):
    output = update_run(saved_run, 32, tmp_path / "evidence")
    with run_lock(saved_run):
        before = snapshots(saved_run)
        with pytest.raises(LoopLockError, match="already running"):
            update_run(saved_run, 16, tmp_path / "refused")
        with pytest.raises(LoopLockError, match="already running"):
            verify_saved(saved_run, output)
        assert snapshots(saved_run) == before
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize("batch", [0, 65, True, False, 32.0, "32"])
def test_invalid_batch_never_changes_state(saved_run, tmp_path, batch):
    before = snapshots(saved_run)
    with pytest.raises(ValueError, match="batch"):
        update_run(saved_run, batch, tmp_path / "evidence")
    assert snapshots(saved_run) == before
    assert not (tmp_path / "evidence").exists()


@pytest.mark.parametrize("damage", ["bad_json", "bad_format", "bad_samples", "wrong_size"])
def test_corrupt_published_chunk_refuses_migration(saved_run, tmp_path, damage):
    binary = saved_run / "gens/gen_0001/selfplay/chunk_000000.bin"
    sidecar = binary.with_suffix(".json")
    if damage == "bad_json":
        sidecar.write_text("{")
    elif damage == "wrong_size":
        binary.write_bytes(b"short")
    else:
        metadata = json.loads(sidecar.read_text())
        metadata["format" if damage == "bad_format" else "samples"] = "invalid"
        sidecar.write_text(json.dumps(metadata))
    before = snapshots(saved_run)
    with pytest.raises(ValueError):
        update_run(saved_run, 32, tmp_path / "evidence")
    assert snapshots(saved_run) == before
    assert not (tmp_path / "evidence").exists()


def test_incomplete_chunks_are_skipped_and_left_intact(saved_run, tmp_path):
    directory = saved_run / "gens/gen_0001/selfplay"
    (directory / "chunk_000002.bin").write_bytes(b"incomplete")
    (directory / "chunk_000003.json").write_text("{}")
    (directory / "chunk_000004.bin.tmp").write_bytes(b"pending")
    before = snapshots(saved_run)
    output = update_run(saved_run, 32, tmp_path / "evidence")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["complete_chunks"] == 2
    assert manifest["skipped_incomplete_chunks"] == ["chunk_000002", "chunk_000003"]
    assert {k: v for k, v in snapshots(saved_run).items() if k != "run.json"} == {
        k: v for k, v in before.items() if k != "run.json"}


@pytest.mark.parametrize("change", ["bytes", "mtime"])
def test_verification_detects_rewritten_or_touched_complete_chunks(saved_run, tmp_path, change):
    output = update_run(saved_run, 32, tmp_path / "evidence")
    binary = saved_run / "gens/gen_0001/selfplay/chunk_000000.bin"
    info = binary.stat()
    if change == "bytes":
        binary.write_bytes(bytes([9]) * info.st_size)
        os.utime(binary, ns=(info.st_atime_ns, info.st_mtime_ns))
        message = "sha256"
    else:
        os.utime(binary, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
        message = "mtime_ns"
    before = snapshots(saved_run)
    with pytest.raises(ValueError, match=message):
        verify_saved(saved_run, output)
    assert snapshots(saved_run) == before


def test_backup_must_be_new_and_outside_run(saved_run, tmp_path):
    before = snapshots(saved_run)
    with pytest.raises(ValueError, match="outside"):
        update_run(saved_run, 32, saved_run / "backup")
    with pytest.raises(FileExistsError):
        update_run(saved_run, 32, tmp_path)
    assert snapshots(saved_run) == before


def test_cli_verify_and_error_statuses(saved_run, tmp_path):
    output = tmp_path / "evidence"
    assert main(["--run", str(saved_run), "--batch", "32", "--out", str(output)]) == 0
    assert main(["--run", str(saved_run), "--verify-saved", str(output)]) == 0
    assert main(["--run", str(saved_run), "--batch", "32", "--out", str(output)]) == 1
    with pytest.raises(SystemExit) as caught:
        main(["--run", str(saved_run), "--batch", "0"])
    assert caught.value.code == 2
