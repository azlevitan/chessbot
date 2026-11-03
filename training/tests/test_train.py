"""Task 6.4: real Rust chunks, streaming training, validation and checkpoints."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

import train
import train_loop
from checkpoint import load_checkpoint
from data.chunk_dataset import ChunkDataset
from data.replay_buffer import make_loader
from data.sampler import WeightedSourceSampler
from losses import masked_logits, total_loss, wdl_loss
from schedule import lr_multiplier

SCRIPT = Path(train.__file__)
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def chunk(engine_bin, tmp_path_factory):
    directory = tmp_path_factory.mktemp("trainer_chunk")
    path = directory / "chunk_000000.bin"
    pgn = Path(__file__).parent / "fixtures/sample.pgn"
    subprocess.run([str(engine_bin), "convert-pgn", "--input", str(pgn), "--out", str(path)],
                   check=True)
    assert len(ChunkDataset([path])) == 60
    return path


def run_cli(chunk, directory, *extra):
    out = directory / "model.pt"
    command = [sys.executable, str(SCRIPT), "--source", str(chunk), "--out", str(out),
               "--steps", "60", "--batch-size", "16", "--lr", "3e-3", "--q-ratio", "0.3",
               "--device", "cpu", "--seed", "0", *map(str, extra)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return out, json.loads((directory / "train.json").read_text()), [
        json.loads(line) for line in result.stdout.splitlines()]


@pytest.fixture(scope="module")
def runs(chunk, tmp_path_factory):
    root = tmp_path_factory.mktemp("trainer_runs")
    args = ("--steps-per-epoch", "10", "--val", chunk, "--val-samples", "17")
    return [run_cli(chunk, root / name, *args) for name in ("first", "repeat")]


def test_training_loss_checkpoint_and_seed(runs):
    path, summary, records = runs[0]
    assert records[-1]["loss"] <= 0.7 * records[0]["loss"]
    assert summary["final_loss"] == runs[1][1]["final_loss"]
    assert summary["steps"] == 60 and summary["samples_seen"] == 60 * 16
    assert summary["final_loss"] == pytest.approx(summary["policy_loss"] + summary["wdl_loss"])
    assert summary["seconds"] > 0
    assert summary["sources"][0]["samples"] == 60
    assert [record["steps"] for record in records] == [10, 20, 30, 40, 50, 60]
    assert all(0 <= record["val"]["policy_top1"] <= 1 for record in records)
    restored = load_checkpoint(path)
    assert not restored.training
    repeat = load_checkpoint(runs[1][0])
    assert all(torch.equal(value, repeat.state_dict()[key])
               for key, value in restored.state_dict().items())
    assert not list(path.parent.glob("*.tmp"))


def test_init_starts_from_checkpoint(chunk, runs, tmp_path):
    initial = load_checkpoint(runs[0][0])
    out, summary, records = run_cli(chunk, tmp_path, "--init", runs[0][0], "--steps", "1",
                                    "--lr", "1e-12", "--weight-decay", "0")
    restored = load_checkpoint(out)
    for before, after in zip(initial.parameters(), restored.parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=1e-9)
    assert summary["val"] is None and len(records) == 1
    assert summary["samples_seen"] == 16


def test_wdl_hard_labels_soft_q_and_blend():
    logits = torch.tensor([[1., 2., -1.], [0., -2., 1.], [-1., 1., 2.]], requires_grad=True)
    labels = torch.tensor([2, 0, 1])
    q = torch.tensor([-2., 0., 2.])
    hard = F.cross_entropy(logits, labels)
    assert torch.equal(wdl_loss(logits, labels, q, 0), hard)
    assert torch.equal(wdl_loss(logits, labels, None, 0.3), hard)
    torch.testing.assert_close(wdl_loss(logits, labels, q, 1),
                               F.cross_entropy(logits, torch.tensor([0, 1, 2])))
    torch.testing.assert_close(wdl_loss(logits, labels, torch.zeros(3), 1),
                               F.cross_entropy(logits, torch.ones(3, dtype=torch.long)))
    q = torch.tensor([-0.4, 0.2, 0.7])
    soft = torch.tensor([[0.4, 0.6, 0.], [0., 0.8, 0.2], [0., 0.3, 0.7]])
    target = 0.7 * F.one_hot(labels, 3) + 0.3 * soft
    expected = -(target * logits.log_softmax(-1)).sum(-1).mean()
    actual = wdl_loss(logits, labels, q, 0.3)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_schedule_boundaries():
    factors = [lr_multiplier(i, 6, 2) for i in range(6)]
    assert factors[:3] == [0.5, 1.0, 1.0]
    assert factors[-1] == pytest.approx(0.1)
    assert factors[2:] == sorted(factors[2:], reverse=True)
    assert lr_multiplier(0, 6, 0) == 1
    assert lr_multiplier(5, 6, 0) == pytest.approx(0.1)
    assert [lr_multiplier(i, 3, 100) for i in range(3)] == pytest.approx([0.5, 1, 0.1])
    assert lr_multiplier(0, 1, 100) == pytest.approx(0.1)


def test_sources_directories_weights_and_partial_chunks(chunk, tmp_path):
    directory = tmp_path / "source:plain"
    directory.mkdir()
    for suffix in (".bin", ".json"):
        shutil.copyfile(chunk.with_suffix(suffix), directory / f"chunk_000001{suffix}")
    (directory / "orphan.bin").write_bytes(b"")
    (directory / "chunk_000002.bin.tmp").write_bytes(b"")
    empty = tmp_path / "empty"
    empty.mkdir()
    dataset, ranges, sources = train.build_sources([f"{directory},{chunk}:0.25", str(empty),
                                                   f"{chunk}:0"])
    assert len(dataset) == 180 and ranges == [(0, 120), (120, 120), (120, 180)]
    assert [source["weight"] for source in sources] == [0.25, 1, 0]
    assert len(train.build_sources([str(directory)])[0]) == 60
    assert all(isinstance(samples, np.memmap) for samples in dataset._samples)
    with pytest.raises(FileNotFoundError):
        train.build_sources([str(directory / "orphan.bin")])
    for spec in (f"{chunk}:nan", f"{chunk}:-1", f"{chunk}:inf", f"{chunk},"):
        with pytest.raises(ValueError):
            train.build_sources([spec])


class TinyNet(torch.nn.Module):
    """Cheap dual-head test model; all position decoding stays in the dataset."""

    def __init__(self):
        super().__init__()
        self.policy = torch.nn.Parameter(torch.zeros(4168))
        self.wdl = torch.nn.Parameter(torch.tensor([0., 1., -1.]))
        self.seen = []

    def forward(self, planes):
        if self.training:
            self.seen.append(planes.detach().clone())
        return self.policy.expand(len(planes), -1), self.wdl.expand(len(planes), -1)


def test_continuous_sampling_validation_and_partial_epoch(chunk, monkeypatch, capsys):
    dataset = ChunkDataset([chunk])
    sampler = WeightedSourceSampler([(0, 60)], [1], num_samples=10, seed=8)
    loader = make_loader(dataset, batch_size=2, sampler=sampler)
    assert loader.num_workers == 0 and loader.pin_memory is False
    # Uneven validation batches must be weighted by sample count, not batch count.
    validation = Subset(dataset, range(3))
    val_loader = make_loader(validation, batch_size=2, shuffle=False)
    cache_calls = []
    monkeypatch.setattr(train_loop, "empty_cache", cache_calls.append)
    model = TinyNet()
    result = train_loop.train_steps(model, loader, steps=5, device=CPU, steps_per_epoch=2,
                                   q_ratio=0.3, val_loader=val_loader)
    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [record["steps"] for record in records] == [2, 4, 5]
    assert cache_calls == [CPU] * 3 and model.training
    expected = dataset.__getitems__(list(sampler))
    assert torch.equal(torch.cat(model.seen), expected["planes"])
    batch = dataset.__getitems__([0, 1, 2])
    model.eval()
    with torch.no_grad():
        policy, wdl = model(batch["planes"])
        _, p_loss, v_loss = total_loss(policy, wdl, batch)
    val = train_loop.evaluate(model, val_loader, CPU)
    assert not model.training
    assert val["policy_loss"] == pytest.approx(float(p_loss))
    assert val["wdl_loss"] == pytest.approx(float(v_loss))
    top1 = (masked_logits(policy, batch["legal_mask"]).argmax(-1) == batch["policy"].argmax(-1))
    assert val["policy_top1"] == pytest.approx(float(top1.float().mean()))
    assert result["val"] == val and result["samples_seen"] == 10


def test_checkpoint_is_published_before_summary(chunk, tmp_path, monkeypatch):
    out, log = tmp_path / "model.pt", tmp_path / "metrics.json"
    out.write_bytes(b"old checkpoint")
    log.write_text("old summary")
    monkeypatch.setattr(train, "train_steps", lambda *a, **kw: {"val": None})
    original = train.save_checkpoint

    def save(model, temporary):
        assert temporary == tmp_path / "model.pt.tmp"
        assert out.read_bytes() == b"old checkpoint" and log.read_text() == "old summary"
        original(model, temporary)

    monkeypatch.setattr(train, "save_checkpoint", save)
    train.main(["--source", str(chunk), "--out", str(out), "--steps", "1",
                "--device", "cpu", "--log-json", str(log)])
    assert not load_checkpoint(out).training
    assert json.loads(log.read_text())["sources"][0]["samples"] == 60
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("flag,value", [("--steps", "0"), ("--q-ratio", "nan"),
    ("--q-ratio", "1.1"), ("--lr", "inf"), ("--warmup-steps", "-1"), ("--batch-size", "0"),
    ("--log-json", "unused.pt")])
def test_invalid_cli_options(flag, value):
    with pytest.raises(SystemExit) as exc:
        train.parse_args(["--source", "unused.bin", "--out", "unused.pt", "--steps", "1",
                          flag, value])
    assert exc.value.code == 2
