"""Durable curves, comparable hard-label probes and interrupted attempt lineage."""

import json
from pathlib import Path
import subprocess

import pytest
import torch

import train
import train_loop
from data.chunk_dataset import ChunkDataset
from data.replay_buffer import make_loader
from train_history import TrainingHistory, fixed_loader, train_probe, validation_indices


@pytest.fixture(scope="module")
def history_chunk(engine_bin, tmp_path_factory):
    root = tmp_path_factory.mktemp("history_chunk")
    path = root / "chunk_000000.bin"
    subprocess.run([str(engine_bin), "convert-pgn", "--input",
                    str(Path(__file__).parent / "fixtures/sample.pgn"),
                    "--out", str(path)], check=True)
    return path


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = torch.nn.Parameter(torch.zeros(4168))
        self.wdl = torch.nn.Parameter(torch.tensor([0.0, 1.0, -1.0]))

    def forward(self, planes):
        return self.policy.expand(len(planes), -1), self.wdl.expand(len(planes), -1)


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_history_preserves_attempts_and_recovers_only_truncated_tail(tmp_path):
    path = tmp_path / "train_history.jsonl"
    first = TrainingHistory(path, {"seed": 1})
    first.record({"steps": 0})
    before = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b'{"partial":' + b"x" * 5000)
    second = TrainingHistory(path, {"seed": 2})
    second.failed(KeyboardInterrupt())
    assert path.read_bytes().startswith(before)
    rows = read_rows(path)
    assert [row["event"] for row in rows] == ["start", "metrics", "start", "interrupted"]
    assert rows[0]["attempt_id"] == rows[1]["attempt_id"]
    assert rows[2]["attempt_id"] == rows[3]["attempt_id"]
    assert rows[0]["attempt_id"] != rows[2]["attempt_id"]
    assert all(row["timestamp"].endswith("+00:00") for row in rows)


def test_fixed_probes_cover_sources_and_do_not_advance_training_rng(history_chunk):
    dataset = ChunkDataset([history_chunk])
    sources = [{"weight": 1.0}, {"weight": 0.0}]
    first, count = train_probe(dataset, [(0, 30), (30, 60)], sources, 100, 7, 3)
    repeat, _ = train_probe(dataset, [(0, 30), (30, 60)], sources, 100, 7, 3)
    assert count == 60
    assert first.dataset.indices == repeat.dataset.indices
    assert all(0 <= index < 30 for index in first.dataset.indices)
    assert first.num_workers == 0 and first.pin_memory is False
    state = torch.random.get_rng_state()
    list(first)
    assert torch.equal(state, torch.random.get_rng_state())
    assert validation_indices(1000, 4) == [0, 333, 666, 999]
    assert validation_indices(2, 4) == [0, 1]


def test_baseline_epoch_and_comparable_probes(history_chunk, capsys):
    dataset = ChunkDataset([history_chunk])
    probe = fixed_loader(dataset, [0, 1, 2], 2, 7)
    loader = make_loader(dataset, 2, shuffle=False)
    records = []
    result = train_loop.train_steps(
        TinyNet(), loader, steps=3, device=torch.device("cpu"), steps_per_epoch=2,
        q_ratio=1.0, train_probe_loader=probe, val_loader=probe, on_record=records.append,
    )
    assert [row["steps"] for row in records] == [0, 2, 3]
    assert [row["phase"] for row in records] == ["baseline", "epoch", "epoch"]
    assert records[0]["train"] is None
    assert [row["samples_seen"] for row in records] == [0, 4, 6]
    assert [row["seconds"] for row in records] == sorted(row["seconds"] for row in records)
    for row in records:
        # Identical inputs prove both probes use hard labels even at q_ratio=1.
        assert row["train_probe"] == row["val"]
        assert row["val"]["loss"] == pytest.approx(
            row["val"]["policy_loss"] + row["val"]["wdl_loss"]
        )
        assert row["val"]["samples"] == 3
        assert 0 <= row["val"]["wdl_accuracy"] <= 1
        assert 0 <= row["val"]["wdl_brier"] <= 2
        assert row["val"]["policy_kl"] == pytest.approx(
            row["val"]["policy_loss"] - row["val"]["policy_target_entropy"]
        )
    assert result["train_probe"] == records[-1]["train_probe"]
    stdout = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["steps"] for row in stdout] == [2, 3]


def test_cli_records_failures_and_retry_lineage(history_chunk, tmp_path, monkeypatch):
    path = tmp_path / "curves.jsonl"
    args = ["--source", str(history_chunk), "--out", str(tmp_path / "model.pt"),
            "--steps", "1", "--device", "cpu", "--history-jsonl", str(path)]

    def interrupted(*args, **kwargs):
        kwargs["on_record"]({"phase": "baseline", "steps": 0})
        raise KeyboardInterrupt()

    monkeypatch.setattr(train, "train_steps", interrupted)
    with pytest.raises(KeyboardInterrupt):
        train.main(args)
    assert [row["event"] for row in read_rows(path)] == ["start", "metrics", "interrupted"]

    monkeypatch.setattr(train, "train_steps", lambda *a, **kw: {"steps": 1, "val": None})
    result = train.main(args)
    rows = read_rows(path)
    assert [row["event"] for row in rows] == ["start", "metrics", "interrupted", "start", "complete"]
    assert rows[0]["attempt_id"] != rows[-1]["attempt_id"] == result["attempt_id"]
    assert result["history_file"] == str(path.resolve())
    summary = json.loads((tmp_path / "train.json").read_text())
    assert summary["attempt_id"] == result["attempt_id"]
    assert rows[-1]["summary_file"] == str((tmp_path / "train.json").resolve())


@pytest.mark.parametrize("name", ["model.pt", "model.pt.tmp", "train.json", "train.json.tmp"])
def test_history_cannot_alias_model_or_summary(tmp_path, name):
    with pytest.raises(SystemExit):
        train.parse_args(["--source", "unused.bin", "--out", str(tmp_path / "model.pt"),
                          "--steps", "1", "--history-jsonl", str(tmp_path / name)])


def test_history_cannot_modify_source_chunks(history_chunk, tmp_path):
    before = history_chunk.read_bytes()
    with pytest.raises(ValueError, match="source chunks"):
        train.main(["--source", str(history_chunk), "--out", str(tmp_path / "model.pt"),
                    "--steps", "1", "--history-jsonl", str(history_chunk)])
    assert history_chunk.read_bytes() == before


def test_history_write_failure_does_not_mask_training_failure(history_chunk, tmp_path, monkeypatch):
    def broken_training(*args, **kwargs):
        raise KeyboardInterrupt("original interruption")

    def broken_failure_log(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(train, "train_steps", broken_training)
    monkeypatch.setattr(train.TrainingHistory, "failed", broken_failure_log)
    with pytest.raises(KeyboardInterrupt, match="original interruption"):
        train.main(["--source", str(history_chunk), "--out", str(tmp_path / "model.pt"),
                    "--steps", "1", "--device", "cpu"])
