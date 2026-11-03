"""Generation-0 integration: Rust PGN data, real training/export, and crash recovery."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import chess
import pytest

import bootstrap
from run_layout import (
    REPO_ROOT, chunk_manifest, generation_dir, read_json, read_run_json,
    write_json_atomic, write_run_json,
)

PGN = REPO_ROOT / "training/tests/fixtures/sample.pgn"


def arguments(pgn, run, config):
    return ["--pgn", str(pgn), "--run", str(run), "--config", str(config),
            "--val-games", "1", "--steps", "20", "--batch-size", "16", "--device", "cpu"]


def invoke(pgn, run, config):
    return subprocess.run([sys.executable, str(REPO_ROOT / "pipeline/bootstrap.py"),
                           *arguments(pgn, run, config)], cwd=run.parent,
                          check=True, capture_output=True, text=True, timeout=120)


def mtimes(directory):
    return {str(path.relative_to(directory)): path.stat().st_mtime_ns
            for path in directory.rglob("*") if path.is_file()}


@pytest.fixture(scope="module")
def completed(engine_bin, tmp_path_factory):
    root = tmp_path_factory.mktemp("bootstrap")
    config = root / "config.json"
    write_json_atomic(config, {"seed": 7, "train": {"q_ratio": 0.3, "steps_per_epoch": 10}})
    run = root / "run"
    invoke(PGN, run, config)
    return run, config


def test_generation_zero_artifacts_and_split(completed):
    run, config = completed
    gen = generation_dir(run, 0)
    assert gen.name == "gen_0000"
    assert all((gen / name).is_file() for name in
               ("model.pt", "model.onnx", "train.json", "decision.json"))
    assert read_run_json(run) == {"format": "chessbot-run-v1", "config": read_json(config),
                                 "best_gen": 0, "current_gen": 1, "stage": "selfplay",
                                 "resign_threshold": -0.95}
    assert read_json(gen / "decision.json") == {"promoted": True, "reason": "bootstrap"}
    validation = chunk_manifest(run / "supervised_val")
    training = chunk_manifest(run / "supervised")
    assert sum(chunk["samples"] for chunk in validation + training) == 60
    assert read_json(run / "supervised_val/chunk_000000.json")["first_game"] == 0
    assert read_json(run / "supervised_val/chunk_000000.json")["games"] == 1
    assert read_json(run / "supervised/chunk_000000.json")["first_game"] == 1
    assert read_json(run / "supervised/chunk_000000.json")["games"] == 2
    summary = read_json(gen / "train.json")
    assert summary["steps"] == 20 and summary["samples_seen"] == 320
    assert summary["val"] is not None and 0 <= summary["val"]["policy_top1"] <= 1
    assert summary["sources"][0]["samples"] == sum(c["samples"] for c in training)
    assert not list(run.rglob("*.tmp"))


def test_exported_model_plays_over_cpu_uci(completed, engine_bin):
    run, _ = completed
    result = subprocess.run(
        [str(engine_bin), "uci", "--model", str(generation_dir(run, 0) / "model.onnx"),
         "--ep", "cpu"], input="uci\nisready\nposition startpos\ngo nodes 32\nquit\n",
        capture_output=True, text=True, check=True, timeout=60,
    )
    assert "uciok" in result.stdout and "readyok" in result.stdout
    assert "info string error" not in result.stdout
    best = next(line.split()[1] for line in result.stdout.splitlines()
                if line.startswith("bestmove "))
    assert chess.Move.from_uci(best) in chess.Board().legal_moves


def test_repeat_is_noop(completed):
    run, config = completed
    before = mtimes(run)
    invoke(PGN, run, config)
    assert mtimes(run) == before


def test_stale_temporary_metadata_is_cleaned(completed, tmp_path):
    original, config = completed
    run = tmp_path / "stale"
    shutil.copytree(original, run)
    before = mtimes(run)
    for path in (run / "run.json.tmp", generation_dir(run, 0) / "decision.json.tmp",
                 run / "supervised/chunk_000001.bin.tmp"):
        path.write_bytes(b"partial")
    invoke(PGN, run, config)
    assert mtimes(run) == before


def test_advanced_run_is_preserved(completed, tmp_path):
    original, config = completed
    run = tmp_path / "advanced"
    shutil.copytree(original, run)
    state = read_run_json(run)
    state.update(current_gen=3, best_gen=2, stage="train")
    write_run_json(run, state)
    before = mtimes(run)
    invoke(PGN, run, config)
    assert read_run_json(run) == state and mtimes(run) == before


@pytest.mark.parametrize("stage", ["conversion", "train", "export"])
def test_partial_outputs_redo_only_dependent_stages(completed, tmp_path, stage):
    original, config = completed
    run = tmp_path / "recover"
    shutil.copytree(original, run)
    gen = generation_dir(run, 0)
    before = mtimes(run)
    if stage == "conversion":
        (run / "supervised/.complete.json").unlink()
        (run / "supervised/chunk_000001.bin").write_bytes(b"partial")
        stale = run / "supervised.tmp"
        stale.mkdir()
        (stale / "chunk_000000.bin.tmp").write_bytes(b"partial")
    elif stage == "train":
        (gen / "train.json").unlink()
        (gen / "model.pt.tmp").write_bytes(b"partial")
    else:
        (gen / "model.onnx").write_bytes(b"invalid ONNX")
        (gen / "model.onnx.tmp").write_bytes(b"partial")
    invoke(PGN, run, config)
    after = mtimes(run)
    assert after["supervised_val/chunk_000000.bin"] == before["supervised_val/chunk_000000.bin"]
    if stage != "conversion":
        assert after["supervised/chunk_000000.bin"] == before["supervised/chunk_000000.bin"]
    if stage == "export":
        assert after["gens/gen_0000/model.pt"] == before["gens/gen_0000/model.pt"]
    else:
        assert after["gens/gen_0000/model.pt"] != before["gens/gen_0000/model.pt"]
    assert after["gens/gen_0000/model.onnx"] != before["gens/gen_0000/model.onnx"]
    assert bootstrap.valid_onnx(gen / "model.onnx")
    assert not list(run.rglob("*.tmp"))


def test_failed_training_cannot_publish_bootstrap(completed, tmp_path, monkeypatch):
    original, config = completed
    run = tmp_path / "failed"
    shutil.copytree(original, run)
    gen = generation_dir(run, 0)
    (run / "run.json").unlink()
    (gen / "train.json").unlink()

    def fail(*args):
        raise subprocess.CalledProcessError(17, "trainer")

    monkeypatch.setattr(bootstrap, "training_process", fail)
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap.main(arguments(PGN, run, config))
    assert not (gen / "model.onnx").exists() and not (gen / "decision.json").exists()
    assert not (run / "run.json").exists()


def test_data_rebuild_invalidates_model_before_conversion(completed, tmp_path, monkeypatch):
    original, config = completed
    run = tmp_path / "crash"
    shutil.copytree(original, run)
    (run / "supervised/.complete.json").unlink()

    def fail(*args):
        raise subprocess.CalledProcessError(17, "converter")

    monkeypatch.setattr(bootstrap, "convert_pgn", fail)
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap.main(arguments(PGN, run, config))
    gen = generation_dir(run, 0)
    assert not any((gen / name).exists() for name in
                   ("model.pt", "train.json", "model.onnx", "decision.json"))


@pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd CLI required for compressed PGN")
def test_zstd_bootstrap_and_bounded_stream(completed, engine_bin, tmp_path):
    _, config = completed
    compressed = tmp_path / "fixture.pgn.zst"
    with compressed.open("wb") as output:
        subprocess.run(["zstd", "-q", "-c", str(PGN)], stdout=output, check=True)
    run = tmp_path / "compressed"
    invoke(compressed, run, config)
    assert read_run_json(run)["best_gen"] == 0
    for directory in ("supervised", "supervised_val"):
        assert (run / directory / "chunk_000000.bin").read_bytes() == (
            completed[0] / directory / "chunk_000000.bin").read_bytes()
    # Force the converter to stop while the decoder still has megabytes to emit.
    large = tmp_path / "large.pgn"
    fixture = PGN.read_bytes()
    with large.open("wb") as output:
        for _ in range(4000):
            output.write(fixture)
    with compressed.open("wb") as output:
        subprocess.run(["zstd", "-q", "-c", str(large)], stdout=output, check=True)
    target = tmp_path / "bounded"
    assert bootstrap.convert_stage(compressed, target, "--max-games", "1")
    assert (target / "chunk_000000.bin").read_bytes() == (
        run / "supervised_val/chunk_000000.bin").read_bytes()
    assert not bootstrap.convert_stage(compressed, target, "--max-games", "1")
    compressed.write_bytes(b"invalid zstd")
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap.convert_stage(compressed, tmp_path / "bad", "--max-games", "1")
    assert not (tmp_path / "bad").exists()


def test_atomic_state_preserves_previous_value_on_failure(tmp_path, monkeypatch):
    state = {"best_gen": 0, "current_gen": 1, "stage": "selfplay"}
    write_run_json(tmp_path, state)
    previous = read_run_json(tmp_path)

    def fail(*args):
        raise OSError("simulated crash before rename")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail)
        with pytest.raises(OSError):
            write_run_json(tmp_path, {**state, "current_gen": 2})
    assert read_run_json(tmp_path) == previous
    write_run_json(tmp_path, state)
    assert not (tmp_path / "run.json.tmp").exists()
    (tmp_path / "run.json").write_text(json.dumps({"format": "unknown"}))
    with pytest.raises(ValueError, match="chessbot-run-v1"):
        read_run_json(tmp_path)


def test_empty_conversion_and_invalid_arguments(engine_bin, tmp_path):
    empty = tmp_path / "empty.pgn"
    empty.write_text("")
    with pytest.raises(ValueError, match="no samples"):
        bootstrap.convert_stage(empty, tmp_path / "empty", "--max-games", "1")
    assert not (tmp_path / "empty").exists()
    for flag in ("--val-games", "--max-samples", "--steps", "--batch-size"):
        with pytest.raises(SystemExit):
            bootstrap.parse_args([*arguments(PGN, tmp_path, tmp_path / "cfg"), flag, "0"])
    with pytest.raises(ValueError):
        generation_dir(tmp_path, -1)
