"""Subprocess arguments and training/export failure publication boundaries."""

from pathlib import Path
import subprocess
import sys

import pytest

from config import load_config
from run_layout import REPO_ROOT, generation_dir, write_json_atomic
import stages


@pytest.fixture
def cfg():
    cfg = load_config(REPO_ROOT / "pipeline/configs/ci_tiny.json")
    cfg.update(engine="selected engine", seed=7)
    return cfg


def state():
    return {"best_gen": 0, "current_gen": 1, "stage": "train", "resign_threshold": -.95}


def training_outputs(run, gen):
    directory = generation_dir(run, gen)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.pt").write_bytes(b"checkpoint")
    write_json_atomic(directory / "train.json", {"steps": 20})
    return directory


def dependent_outputs(directory):
    (directory / "model.onnx").write_bytes(b"old network")
    for path in (directory / "decision.json", directory / "match/block_00.json",
                 directory / "anchors/vs_random.json"):
        write_json_atomic(path, {"old": True})


def sample(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "chunk_000000.bin"
    path.write_bytes(bytes(492))
    write_json_atomic(path.with_suffix(".json"), {"format": "chessbot-chunk-v1", "samples": 1})
    return path


def test_train_forwards_weighted_sources_and_all_settings(tmp_path, cfg, monkeypatch):
    best = training_outputs(tmp_path, 0)
    candidate = generation_dir(tmp_path, 1)
    replay = sample(candidate / "selfplay")
    supervised = sample(tmp_path / "supervised")
    cfg["train"].update(lr=.002, weight_decay=.0003, warmup_steps=2, val_samples=8)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert kwargs["check"] is True
        training_outputs(tmp_path, 1)

    monkeypatch.setattr(stages.subprocess, "run", run)
    stages.train(tmp_path, 1, cfg, state())
    assert len(calls) == 1
    command = calls[0]
    assert command[:2] == [sys.executable, str(REPO_ROOT / "training/train.py")]
    assert command[2:6] == ["--source", f"{replay}:0.5", "--source", f"{supervised}:0.5"]
    options = dict(zip(command[6::2], command[7::2]))
    assert options == {"--init": str(best / "model.pt"), "--out": str(candidate / "model.pt"),
                       "--log-json": str(candidate / "train.json"),
                       "--val": str(tmp_path / "supervised_val"), "--seed": "7",
                       "--q-ratio": "0.3", "--steps": "20", "--batch-size": "16",
                       "--lr": "0.002", "--weight-decay": "0.0003", "--warmup-steps": "2",
                       "--steps-per-epoch": "10", "--device": "cpu", "--val-samples": "8"}


def test_training_failure_removes_candidate_outputs_before_retry(tmp_path, cfg, monkeypatch):
    best = training_outputs(tmp_path, 0)
    candidate = training_outputs(tmp_path, 1)
    dependent_outputs(candidate)
    sample(candidate / "selfplay")
    (candidate / "train.json").unlink()
    (candidate / "model.pt.tmp").write_bytes(b"unfinished")

    def fail(command, **kwargs):
        assert not (candidate / "model.onnx").exists()
        assert not (candidate / "decision.json").exists()
        (candidate / "model.pt.tmp").write_bytes(b"new unfinished")
        raise subprocess.CalledProcessError(17, command)

    monkeypatch.setattr(stages.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        stages.train(tmp_path, 1, cfg, state())
    assert not any((candidate / name).exists() for name in
                   ("model.pt", "train.json", "model.onnx", "decision.json"))
    assert not list(candidate.rglob("*.tmp"))
    assert (best / "model.pt").read_bytes() == b"checkpoint"


@pytest.mark.parametrize("condition", ["missing", "invalid", "temporary"])
def test_export_replaces_only_after_success_and_checks_parity_source(tmp_path, cfg, monkeypatch, condition):
    candidate = training_outputs(tmp_path, 1)
    dependent_outputs(candidate)
    onnx = candidate / "model.onnx"
    if condition == "missing":
        onnx.unlink()
    if condition == "temporary":
        (candidate / "model.onnx.tmp").write_bytes(b"unfinished")
    exports = []

    def run(command, **kwargs):
        assert kwargs["check"] is True
        if command[1] == "-c":
            raise subprocess.CalledProcessError(1, command)
        exports.append(command)
        options = dict(zip(command[2::2], command[3::2]))
        assert options == {"--checkpoint": str(candidate / "model.pt"),
                           "--out": str(candidate / "model.onnx.tmp"),
                           "--parity-source": str(tmp_path / "supervised_val")}
        Path(options["--out"]).write_bytes(b"validated network")

    monkeypatch.setattr(stages.subprocess, "run", run)
    stages.export(tmp_path, 1, cfg, state())
    assert len(exports) == 1 and onnx.read_bytes() == b"validated network"
    assert (candidate / "model.pt").read_bytes() == b"checkpoint"
    assert not (candidate / "decision.json").exists()
    assert not list(candidate.rglob("*.tmp"))


def test_valid_export_is_checked_in_subprocess_then_preserved(tmp_path, cfg, monkeypatch):
    candidate = training_outputs(tmp_path, 1)
    dependent_outputs(candidate)
    onnx, calls = candidate / "model.onnx", []
    before = onnx.stat().st_mtime_ns

    def run(command, **kwargs):
        calls.append(command)
        assert command[1] == "-c" and "check_contract" in command[2]
        assert command[-1] == str(onnx) and kwargs["check"] is True
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(stages.subprocess, "run", run)
    stages.export(tmp_path, 1, cfg, state())
    assert len(calls) == 1 and onnx.stat().st_mtime_ns == before
    assert (candidate / "decision.json").exists()


def test_export_failure_preserves_checkpoint_and_previous_model(tmp_path, cfg, monkeypatch):
    candidate = training_outputs(tmp_path, 1)
    dependent_outputs(candidate)
    (candidate / "model.onnx.tmp").write_bytes(b"interrupted")

    def fail(command, **kwargs):
        options = dict(zip(command[2::2], command[3::2]))
        Path(options["--out"]).write_bytes(b"bad export")
        raise subprocess.CalledProcessError(17, command)

    monkeypatch.setattr(stages.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        stages.export(tmp_path, 1, cfg, state())
    assert (candidate / "model.pt").read_bytes() == b"checkpoint"
    assert (candidate / "model.onnx").read_bytes() == b"old network"
    assert not (candidate / "decision.json").exists()
    assert not list(candidate.rglob("*.tmp"))


def test_match_forwards_generation_seed_and_inference_overrides(tmp_path, cfg, monkeypatch):
    cfg["match"].update(ep="cpu", threads=2, alpha=.1, beta=.2, elo0=-10, elo1=30)
    calls = []
    monkeypatch.setattr(stages.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    stages.match(tmp_path, 1, cfg, state())
    assert len(calls) == 1 and calls[0][1]["check"] is True
    command = calls[0][0]
    assert command[:2] == [sys.executable, str(REPO_ROOT / "pipeline/evaluate.py")]
    options = dict(zip(command[2::2], command[3::2]))
    assert options["--engine"] == "selected engine" and options["--seed-base"] == "1000000"
    assert options["--a"] == str(generation_dir(tmp_path, 1) / "model.onnx")
    assert options["--b"] == str(generation_dir(tmp_path, 0) / "model.onnx")
    assert options["--ep"] == "cpu" and options["--threads"] == "2"
    assert options["--alpha"] == "0.1" and options["--beta"] == "0.2"
    assert options["--elo0"] == "-10" and options["--elo1"] == "30"
