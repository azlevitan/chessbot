"""Phase 6.3: real Rust chunks, lazy multi-file reads and weighted loading."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from data.chunk_dataset import ChunkDataset, list_complete_chunks, read_sidecar
from data.replay_buffer import SAMPLE_DTYPE, ReplayBuffer, make_loader
from data.sampler import WeightedSourceSampler


@pytest.fixture(scope="module")
def chunks(engine_bin, tmp_path_factory):
    directory = tmp_path_factory.mktemp("chunks")
    args = ["selfplay", "--games", "2", "--sims", "8", "--max-plies", "20", "--seed", "5"]
    for name, command in (
        ("selfplay", args),
        ("pgn", ["convert-pgn", "--input", str(Path(__file__).parent / "fixtures/sample.pgn")]),
        ("single2", ["selfplay", "--games", "1", "--sims", "8", "--max-plies", "100", "--seed", "2"]),
        ("single5", ["selfplay", "--games", "1", "--sims", "8", "--max-plies", "100", "--seed", "5"]),
    ):
        subprocess.run([str(engine_bin), *command, "--out", str(directory / f"{name}.bin")], check=True)
    return directory


def assert_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key].dtype == expected[key].dtype, key
        assert torch.equal(actual[key], expected[key]), key


def test_every_item_matches_replay_buffer(chunks):
    paths = [chunks / "selfplay.bin", chunks / "pgn.bin"]
    dataset = ChunkDataset(paths)
    assert len(dataset) == sum(read_sidecar(p)["samples"] for p in paths) == 100
    offset = 0
    for path in paths:
        reference = ReplayBuffer(path)
        for i in range(len(reference)):
            item = dataset[offset + i]
            assert_equal(item, reference[i])
            assert item["q_value"].dtype == torch.float32
            assert -1 <= item["q_value"].item() <= 1
            assert item["q"] is item["q_value"]
            assert item["policy"].sum().item() == pytest.approx(1, abs=1e-5)
        offset += len(reference)
    assert all(isinstance(m, np.memmap) and m.mode == "r" for m in dataset._samples)


def test_single_game_sidecars_match_both_perspectives(chunks):
    labels = set()
    for name in ("single2", "single5"):
        path = chunks / f"{name}.bin"
        results = read_sidecar(path)["results"]
        dataset = ChunkDataset([path])
        assert sum(results.values()) == 1
        for ply in range(len(dataset)):
            expected = 1 if results["draw"] else (2 if bool(results["white"]) == (ply % 2 == 0) else 0)
            assert dataset[ply]["wdl"].item() == expected
            labels.add(expected)
    assert labels == {0, 1, 2}


def test_batch_matches_stack_without_scalar_fallback(chunks, monkeypatch):
    dataset = ChunkDataset([chunks / "selfplay.bin", chunks / "pgn.bin"])
    indices = [99, 40, 0, 39, 40, -1, -100]
    items = [dataset[i] for i in indices]
    expected = {key: torch.stack([item[key] for item in items]) for key in items[0]}
    monkeypatch.setattr(dataset, "__getitem__", lambda i: pytest.fail("scalar fallback"))
    assert_equal(dataset.__getitems__(indices), expected)
    empty = dataset.__getitems__([])
    for key, value in expected.items():
        assert empty[key].shape == (0, *value.shape[1:])
        assert empty[key].dtype == value.dtype
    for indices in ([100], [-101], [0, 100], [2**64 - 1]):
        with pytest.raises(IndexError):
            dataset.__getitems__(indices)
    with pytest.raises(TypeError):
        dataset.__getitems__([1.5])


def test_discovery_ignores_partial_writes_and_sorts(chunks, tmp_path):
    for name in ("chunk_000009", "chunk_000002"):
        for suffix in (".bin", ".json"):
            shutil.copyfile(chunks / f"pgn{suffix}", tmp_path / f"{name}{suffix}")
    (tmp_path / "missing.bin").write_bytes(b"")
    (tmp_path / "stray.bin.tmp").write_bytes(b"")
    (tmp_path / "stray.json.tmp").write_text("{}")
    (tmp_path / "pending.bin").write_bytes(b"")
    (tmp_path / "pending.json.tmp").write_text('{"samples": 0}')
    expected = [tmp_path / f"chunk_{i:06}.bin" for i in (2, 9)]
    assert list_complete_chunks(tmp_path) == expected
    with pytest.raises(FileNotFoundError):
        ChunkDataset([tmp_path / "missing.bin"])


@pytest.mark.parametrize("metadata", [
    "{", "[]", '{"format":"other","samples":0}',
    *[json.dumps({"format": "chessbot-chunk-v1", "samples": n}) for n in (None, -1, 1.5, True, 1)],
])
def test_invalid_sidecars_are_excluded_but_explicit_paths_fail(tmp_path, metadata):
    path = tmp_path / "invalid.bin"
    path.write_bytes(b"")
    path.with_suffix(".json").write_text(metadata)
    assert list_complete_chunks(tmp_path) == []
    with pytest.raises(ValueError):
        ChunkDataset([path])


def test_empty_chunks_and_extra_metadata(chunks, tmp_path):
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    metadata = {"format": "chessbot-chunk-v1", "samples": 0, "extra": "allowed"}
    empty.with_suffix(".json").write_text(json.dumps(metadata))
    assert read_sidecar(empty) == metadata
    assert list_complete_chunks(tmp_path) == [empty]
    dataset = ChunkDataset([empty, chunks / "pgn.bin", empty])
    assert len(dataset) == 60
    assert_equal(dataset[-1], ReplayBuffer(chunks / "pgn.bin")[-1])
    for paths in ([], [empty]):
        dataset = ChunkDataset(paths)
        assert len(dataset) == 0
        assert dataset.__getitems__([])["planes"].shape == (0, 18, 8, 8)
        with pytest.raises(IndexError):
            dataset[0]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.01, 1.01])
def test_invalid_q_values_are_rejected(chunks, tmp_path, value):
    path = tmp_path / "corrupt.bin"
    for suffix in (".bin", ".json"):
        shutil.copyfile(chunks / f"pgn{suffix}", path.with_suffix(suffix))
    records = np.memmap(path, dtype=SAMPLE_DTYPE, mode="r+")
    records[0]["q_value"] = value
    records.flush()
    dataset = ChunkDataset([path])
    for fetch in (lambda: dataset[0], lambda: dataset.__getitems__([0]), lambda: ReplayBuffer(path)[0]):
        with pytest.raises(ValueError, match="q_value"):
            fetch()


def test_weighted_sources_are_reproducible_and_uniform_within_source():
    sampler = WeightedSourceSampler([(0, 4), (4, 16)], [0.25, 0.75], 4000, seed=7)
    indices = list(sampler)
    assert len(indices) == len(sampler) == 4000
    assert indices == list(sampler)
    assert indices == list(WeightedSourceSampler([(0, 4), (4, 16)], [0.25, 0.75], 4000, 7))
    assert indices != list(WeightedSourceSampler([(0, 4), (4, 16)], [0.25, 0.75], 4000, 8))
    counts = np.bincount(indices, minlength=16) / 4000
    assert counts[:4].sum() == pytest.approx(0.25, abs=0.03)
    assert counts[4:].sum() == pytest.approx(0.75, abs=0.03)
    assert np.all(np.abs(counts - 1 / 16) < 0.02)
    active = WeightedSourceSampler([(0, 10), (10, 10), (30, 31)], [0, 100, 2], 25, 1)
    assert list(active) == [30] * 25
    assert list(WeightedSourceSampler([], [], 0, 0)) == []
    assert len(list(WeightedSourceSampler([(0, 1), (1, 2)], [1e308, 1e308], 10, 0))) == 10


@pytest.mark.parametrize("ranges,weights,n", [
    ([(0, 1)], [], 1), ([(0, 1)], [-1], 1), ([(0, 1)], [float("nan")], 1),
    ([(0, 1)], [float("inf")], 1), ([(0, 1)], [0], 1), ([(0, 0)], [1], 1),
    ([(-1, 1)], [1], 1), ([(2, 1)], [1], 1), ([(0, 1)], [1], -1),
])
def test_invalid_sampler_configuration(ranges, weights, n):
    with pytest.raises(ValueError):
        WeightedSourceSampler(ranges, weights, n, seed=0)


@pytest.mark.parametrize("multi_chunk", [False, True])
def test_weighted_loader_preserves_batches_and_guardrails(chunks, multi_chunk):
    path = chunks / "pgn.bin"
    dataset = ChunkDataset([path, chunks / "selfplay.bin"]) if multi_chunk else ReplayBuffer(path)
    sampler = WeightedSourceSampler([(0, 20), (20, len(dataset))], [1, 3], 23, seed=4)
    loader = make_loader(dataset, batch_size=8, sampler=sampler)
    assert loader.num_workers == 0 and loader.pin_memory is False
    batches = list(loader)
    assert [len(batch["wdl"]) for batch in batches] == [8, 8, 7]
    expected_items = [dataset[i] for i in sampler]
    assert_equal(
        {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]},
        {key: torch.stack([item[key] for item in expected_items]) for key in expected_items[0]},
    )
