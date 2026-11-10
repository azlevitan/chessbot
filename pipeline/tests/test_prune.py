"""Disk-budget policy over small fake runs; no engine or model readers needed."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from config import normalize_config
import prune as pruning
from prune import DiskBudgetError, prune
from run_layout import generation_dir


def sized(path, size=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.truncate(size)
    return path


def candidate(run, gen, promoted=True, checkpoint=1000):
    directory = generation_dir(run, gen)
    sized(directory / "model.pt", checkpoint)
    sized(directory / "model.onnx", 100)
    for relative in ("train.json", "match/block_00.json", "anchors/vs_random.json"):
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    decision = {"promoted": promoted, "reason": "bootstrap" if gen == 0 else "max_games"}
    (directory / "decision.json").write_text(json.dumps(decision))
    return directory


def settings(current=6, best=5, window=4, budget=20):
    return ({"current_gen": current, "best_gen": best},
            normalize_config({"window": {"generations": window}, "disk_budget_gb": budget}))


def usage(run):
    return sum(path.stat().st_size for path in run.rglob("*") if path.is_file())


def snapshot(run):
    return {str(path.relative_to(run)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in run.rglob("*") if path.is_file()}


def test_selfplay_expires_strictly_before_window_and_retry_is_noop(tmp_path):
    state, cfg = settings()
    directories = []
    for generation in range(1, 8):
        directory = generation_dir(tmp_path, generation) / "selfplay"
        sized(directory / "chunk_000000.bin")
        sized(directory / "chunk_000000.json", 20)
        sized(directory / "chunk_000001.bin.tmp", 30)
        directories.append(directory)
    assert prune(tmp_path, state, cfg) == directories[:2]
    assert all(not path.exists() for path in directories[:2])
    assert all(path.is_dir() for path in directories[2:])
    assert prune(tmp_path, state, cfg) == []


def test_rejected_models_removed_even_below_budget_and_jsons_preserved(tmp_path):
    rejected = candidate(tmp_path, 2, promoted=False)
    before = snapshot(tmp_path)
    state, cfg = settings()
    assert prune(tmp_path, state, cfg) == [rejected / "model.pt", rejected / "model.onnx"]
    expected = {path: info for path, info in before.items()
                if not path.endswith(("model.pt", "model.onnx"))}
    assert snapshot(tmp_path) == expected


def test_cleanup_order_precedes_oldest_promoted_checkpoint_budget_fallback(tmp_path):
    promoted = [candidate(tmp_path, gen) for gen in (0, 1, 3, 5, 6)]
    rejected = candidate(tmp_path, 2, promoted=False)
    expired = generation_dir(tmp_path, 1) / "selfplay"
    sized(expired / "chunk_000000.bin", 500)
    state, cfg = settings(budget=(usage(tmp_path) - 500 - 1100 - 500) / 1e9)
    assert prune(tmp_path, state, cfg) == [expired, rejected / "model.pt",
                                         rejected / "model.onnx", promoted[0] / "model.pt"]
    assert all((path / "model.pt").is_file() for path in promoted[1:])
    assert all((path / "model.onnx").is_file() for path in promoted)


@pytest.mark.parametrize("extra_byte,removed_count", [(0, 0), (1, 1)])
def test_decimal_budget_boundary_and_bootstrap_checkpoint(tmp_path, extra_byte, removed_count):
    oldest = candidate(tmp_path, 0)
    newest = candidate(tmp_path, 2)
    total = usage(tmp_path)
    state, cfg = settings(current=3, best=2, budget=(total - extra_byte) / 1e9)
    assert prune(tmp_path, state, cfg) == [oldest / "model.pt"][:removed_count]
    assert (newest / "model.pt").is_file()
    assert (oldest / "model.onnx").is_file()


def test_budget_deletes_promoted_checkpoints_by_generation_not_mtime(tmp_path):
    later = candidate(tmp_path, 3)
    oldest = candidate(tmp_path, 0)
    middle = candidate(tmp_path, 1)
    state, cfg = settings(budget=(usage(tmp_path) - 1500) / 1e9)
    assert prune(tmp_path, state, cfg) == [oldest / "model.pt", middle / "model.pt"]
    assert (later / "model.pt").is_file()


def test_entire_best_and_current_future_and_supervised_are_protected(tmp_path):
    best = candidate(tmp_path, 1, promoted=False)
    current = candidate(tmp_path, 6, promoted=False)
    future = candidate(tmp_path, 7, promoted=False)
    for directory in (best, current, future):
        sized(directory / "selfplay/chunk_000000.bin")
    for name in ("supervised", "supervised_val", "supervised_backup"):
        sized(tmp_path / name / "chunk_000000.bin")
    before = snapshot(tmp_path)
    state, cfg = settings(best=1, window=1, budget=1e-9)
    with pytest.raises(DiskBudgetError):
        prune(tmp_path, state, cfg)
    assert snapshot(tmp_path) == before


def test_rejected_current_models_become_eligible_next_generation(tmp_path):
    rejected = candidate(tmp_path, 6, promoted=False)
    state, cfg = settings()
    assert prune(tmp_path, state, cfg) == []
    state["current_gen"] = 7
    assert prune(tmp_path, state, cfg) == [rejected / "model.pt", rejected / "model.onnx"]


@pytest.mark.parametrize("contents", [None, "malformed", "[]", "{}", '{"promoted":0}',
                                      '{"promoted":"false"}', '{"promoted":null}'])
def test_unusable_decisions_do_not_authorize_model_deletion(tmp_path, contents):
    directory = candidate(tmp_path, 2)
    decision = directory / "decision.json"
    if contents is None:
        decision.unlink()
    else:
        decision.write_text(contents)
    before = snapshot(tmp_path)
    state, cfg = settings()
    assert prune(tmp_path, state, cfg) == []
    assert snapshot(tmp_path) == before


def test_unreachable_budget_removes_eligible_work_then_raises_on_every_retry(tmp_path):
    old = candidate(tmp_path, 0)
    rejected = candidate(tmp_path, 2, promoted=False)
    sized(tmp_path / "supervised/chunk_000000.bin", 4000)
    state, cfg = settings(budget=1e-9)
    for _ in range(2):
        with pytest.raises(DiskBudgetError, match="budget"):
            prune(tmp_path, state, cfg)
    assert not (old / "model.pt").exists()
    assert (old / "model.onnx").is_file()
    assert not (rejected / "model.pt").exists()
    assert not (rejected / "model.onnx").exists()
    assert (tmp_path / "supervised/chunk_000000.bin").stat().st_size == 4000


def test_pruning_reads_only_metadata_and_does_not_mutate_inputs(tmp_path, monkeypatch):
    candidate(tmp_path, 2, promoted=False)
    sized(generation_dir(tmp_path, 1) / "selfplay/chunk_000000.bin")
    state, cfg = settings()
    original_state, original_cfg = deepcopy(state), deepcopy(cfg)
    original_open = Path.open

    def metadata_only(path, *args, **kwargs):
        if path.suffix in (".bin", ".pt", ".onnx"):
            pytest.fail(f"prune read binary contents: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", metadata_only)
    prune(tmp_path, state, cfg)
    assert state == original_state and cfg == original_cfg


def test_symbolic_links_never_delete_or_read_outside_targets(tmp_path):
    run, outside = tmp_path / "run", tmp_path / "outside"
    external = candidate(outside, 2, promoted=False)
    sized(external / "selfplay/chunk_000000.bin")
    (run / "gens").mkdir(parents=True)
    generation_dir(run, 2).symlink_to(external, target_is_directory=True)
    local = candidate(run, 1, promoted=False)
    (local / "selfplay").symlink_to(external / "selfplay", target_is_directory=True)
    (local / "model.pt").unlink()
    (local / "model.pt").symlink_to(external / "model.pt")
    before = snapshot(outside)
    state, cfg = settings()
    try:
        prune(run, state, cfg)
    except (OSError, ValueError):
        pass
    assert snapshot(outside) == before


def test_interrupted_directory_cleanup_resumes_from_remaining_files(tmp_path, monkeypatch):
    expired = generation_dir(tmp_path, 1) / "selfplay"
    binary = sized(expired / "chunk_000000.bin")
    sidecar = sized(expired / "chunk_000000.json", 20)
    rejected = candidate(tmp_path, 2, promoted=False)
    state, cfg = settings()

    def interrupt(path):
        assert path == expired
        binary.unlink()
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(pruning.shutil, "rmtree", interrupt)
        with pytest.raises(KeyboardInterrupt):
            prune(tmp_path, state, cfg)
    assert not binary.exists() and sidecar.is_file()
    assert (rejected / "model.pt").is_file()
    assert prune(tmp_path, state, cfg) == [expired, rejected / "model.pt",
                                         rejected / "model.onnx"]
    assert not expired.exists()
    assert prune(tmp_path, state, cfg) == []


def test_links_inside_expired_real_selfplay_are_unlinked_without_target_changes(tmp_path):
    run, outside = tmp_path / "run", tmp_path / "outside"
    protected = sized(outside / "important.bin", 400)
    directory = generation_dir(run, 1) / "selfplay"
    directory.mkdir(parents=True)
    (directory / "linked.bin").symlink_to(protected)
    (directory / "linked-dir").symlink_to(outside, target_is_directory=True)
    before = snapshot(outside)
    state, cfg = settings()
    assert prune(run, state, cfg) == [directory]
    assert not directory.exists()
    assert snapshot(outside) == before


def test_nested_file_named_loop_lock_still_counts_toward_budget(tmp_path):
    historical = candidate(tmp_path, 0)
    nested = sized(tmp_path / "metadata/loop.lock", 4000)
    state, cfg = settings(budget=0.000001)
    with pytest.raises(DiskBudgetError):
        prune(tmp_path, state, cfg)
    assert not (historical / "model.pt").exists()
    assert nested.stat().st_size == 4000
