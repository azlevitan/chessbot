"""Real generation-one stages, no-op reruns, and selective crash recovery."""

from copy import deepcopy
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import bootstrap
from config import load_config
from run_layout import REPO_ROOT, generation_dir, read_json, read_run_json, write_json_atomic
import stages

STAGES = ("selfplay", "train", "export", "match", "decide", "anchors")


def mtimes(directory):
    return {str(path.relative_to(directory)): path.stat().st_mtime_ns
            for path in directory.rglob("*") if path.is_file()}


def configured(engine_bin):
    cfg = load_config(REPO_ROOT / "pipeline/configs/ci_tiny.json")
    cfg["engine"] = str(engine_bin)
    return cfg


def copy_run(original, target):
    """Copies used by recovery tests keep their recorded artifact paths local."""
    shutil.copytree(original, target)

    def relocated(value):
        if isinstance(value, dict):
            return {key: relocated(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocated(item) for item in value]
        return value.replace(str(original), str(target)) if isinstance(value, str) else value

    for path in target.rglob("*.json"):
        write_json_atomic(path, relocated(read_json(path)))


@pytest.fixture(scope="module")
def completed_gen_one(ci_tiny_bootstrap, engine_bin, tmp_path_factory):
    run = tmp_path_factory.mktemp("completed-generation") / "run"
    copy_run(ci_tiny_bootstrap, run)
    cfg, state = configured(engine_bin), read_run_json(run)
    commands = []
    actual_run = subprocess.run

    def record(command, **kwargs):
        commands.append(list(map(str, command)))
        return actual_run(command, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(stages.subprocess, "run", record)
        for name in STAGES:
            assert getattr(stages, name)(run, 1, cfg, state) is None
    return run, cfg, state, commands


def test_generation_one_outputs_and_forwarded_arguments(completed_gen_one):
    run, cfg, state, commands = completed_gen_one
    gen = generation_dir(run, 1)
    chunks = sorted((gen / "selfplay").glob("chunk_*.bin"))
    assert [path.name for path in chunks] == ["chunk_000000.bin", "chunk_000001.bin"]
    for index, binary in enumerate(chunks):
        metadata = read_json(binary.with_suffix(".json"))
        assert metadata["games"] == 2 and binary.stat().st_size == metadata["samples"] * 492
        assert metadata["seed"] == 1_000_000 + 1000 * index + cfg["seed"]
        assert metadata["temp_plies"] == 10
        assert Path(metadata["model"]) == generation_dir(run, 0) / "model.onnx"
    assert (gen / "model.pt").is_file() and bootstrap.valid_onnx(gen / "model.onnx")
    summary = read_json(gen / "train.json")
    assert summary["steps"] == 20 and summary["samples_seen"] == 320
    assert summary["val"] is not None
    trainer = next(command for command in commands if any(p.endswith("/train.py") for p in command))
    assert trainer[trainer.index("--q-ratio") + 1] == "0.3"
    assert Path(trainer[trainer.index("--init") + 1]) == generation_dir(run, 0) / "model.pt"
    assert "--source" in trainer and "--val" in trainer
    decision = read_json(gen / "decision.json")
    assert decision["reason"] in ("sprt_accept", "sprt_reject", "max_games")
    blocks = sorted((gen / "match").glob("block_*.json"))
    assert sum(read_json(path)["games"] for path in blocks) == decision["games"]
    assert state["best_gen"] == (1 if decision["promoted"] else 0)
    for opponent in ("random", "material"):
        anchor = gen / "anchors" / f"vs_{opponent}.json"
        assert anchor.exists() == decision["promoted"]
        if anchor.exists():
            report = read_json(anchor)
            assert report["b"] == opponent and report["games"] == cfg["anchors"]["games"]
            assert Path(report["a"]) == gen / "model.onnx"
    assert read_run_json(run)["best_gen"] == 0
    assert read_run_json(run)["stage"] == "selfplay"
    assert not list(run.rglob("*.tmp"))


def test_completed_stages_preserve_artifacts(completed_gen_one):
    run, cfg, state, _ = completed_gen_one
    before, repeated = mtimes(run), deepcopy(state)
    for name in STAGES:
        getattr(stages, name)(run, 1, cfg, repeated)
    assert mtimes(run) == before and repeated == state


def test_missing_selfplay_sidecar_regenerates_only_its_chunk(completed_gen_one, tmp_path, monkeypatch):
    original, cfg, state, _ = completed_gen_one
    run = tmp_path / "repair"
    copy_run(original, run)
    gen = generation_dir(run, 1)
    missing = gen / "selfplay/chunk_000001.json"
    missing.unlink()
    before = mtimes(run)
    calls, actual_run = [], subprocess.run

    def record(command, **kwargs):
        calls.append(list(map(str, command)))
        return actual_run(command, **kwargs)

    monkeypatch.setattr(stages.subprocess, "run", record)
    repeated = deepcopy(state)
    stages.selfplay(run, 1, cfg, repeated)
    assert len(calls) == 1 and calls[0][1] == "selfplay"
    assert calls[0][calls[0].index("--seed") + 1] == "1001000"
    after = mtimes(run)
    replaced = {"gens/gen_0001/selfplay/chunk_000001.bin"}
    assert all(after[path] == stamp for path, stamp in before.items() if path not in replaced)
    assert missing.is_file() and repeated["resign_threshold"] == state["resign_threshold"]
    assert not list(run.rglob("*.tmp"))


def test_stale_checkpoint_temp_forces_training_retry(completed_gen_one, tmp_path, monkeypatch):
    original, cfg, state, _ = completed_gen_one
    run = tmp_path / "retrain"
    copy_run(original, run)
    gen = generation_dir(run, 1)
    (gen / "model.pt.tmp").write_bytes(b"interrupted checkpoint")
    before = mtimes(run)
    commands, actual_run = [], subprocess.run

    def record(command, **kwargs):
        commands.append(list(map(str, command)))
        return actual_run(command, **kwargs)

    monkeypatch.setattr(stages.subprocess, "run", record)
    stages.train(run, 1, cfg, deepcopy(state))
    trainers = [command for command in commands if any(p.endswith("/train.py") for p in command)]
    assert len(trainers) == 1
    assert Path(trainers[0][trainers[0].index("--init") + 1]) == generation_dir(run, 0) / "model.pt"
    assert (gen / "model.pt").stat().st_mtime_ns != before["gens/gen_0001/model.pt"]
    assert (gen / "train.json").stat().st_mtime_ns != before["gens/gen_0001/train.json"]
    after = mtimes(run)
    assert all(after[path] == stamp for path, stamp in before.items()
               if "/selfplay/" in path or path.startswith("supervised"))
    assert not (gen / "model.onnx").exists() and not (gen / "decision.json").exists()
    assert not list((gen / "match").glob("block_*.json"))
    assert not list((gen / "anchors").glob("vs_*.json"))
    assert not list(run.rglob("*.tmp"))


def test_missing_decision_rebuilds_from_blocks_without_engine(completed_gen_one, tmp_path):
    original, cfg, _, _ = completed_gen_one
    run = tmp_path / "decision-replay"
    copy_run(original, run)
    gen = generation_dir(run, 1)
    decision = read_json(gen / "decision.json")
    (gen / "decision.json").unlink()
    blocks = mtimes(gen / "match")
    forbidden = tmp_path / "unexpected-engine"
    forbidden.write_text(f"#!{sys.executable}\nraise SystemExit(91)\n")
    forbidden.chmod(0o755)
    current = read_run_json(run)
    stages.decide(run, 1, {**cfg, "engine": str(forbidden)}, current)
    assert read_json(gen / "decision.json") == decision
    assert current["best_gen"] == int(decision["promoted"])
    assert mtimes(gen / "match") == blocks
    assert not list(run.rglob("*.tmp"))
