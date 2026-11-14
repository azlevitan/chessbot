"""Inference-only fixed batches propagate without changing cached training data."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from config import DEFAULTS, load_config, normalize_config
from generation_metadata import ensure_manifest
from run_layout import REPO_ROOT, generation_dir, read_run_json, write_json_atomic, write_run_json
import evaluate
import stages
from test_evaluate import arguments, report, snapshots


@pytest.mark.parametrize("section", ["selfplay", "match"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 65, "32", 1.0, 32.0])
def test_config_rejects_invalid_inference_batches(section, value):
    with pytest.raises(ValueError, match=f"{section}.infer_batch"):
        normalize_config({section: {"infer_batch": value}})


@pytest.mark.parametrize("value", [None, 1, 32, 64])
def test_optional_batches_preserve_explicit_none_and_legacy_defaults(value):
    default = normalize_config({})
    assert all("infer_batch" not in default[section] for section in ("selfplay", "match"))
    assert all("infer_batch" not in DEFAULTS[section] for section in ("selfplay", "match"))
    cfg = normalize_config({"selfplay": {"infer_batch": value}, "match": {"infer_batch": value}})
    assert cfg["selfplay"]["infer_batch"] == cfg["match"]["infer_batch"] == value
    overnight = load_config(REPO_ROOT / "pipeline/configs/m1_overnight.json")
    assert overnight["selfplay"]["infer_batch"] == overnight["match"]["infer_batch"] == 32
    tiny = load_config(REPO_ROOT / "pipeline/configs/ci_tiny.json")
    assert all("infer_batch" not in tiny[section] for section in ("selfplay", "match"))


def state():
    return {"best_gen": 0, "current_gen": 1, "stage": "selfplay", "resign_threshold": -0.95}


@pytest.mark.parametrize("selfplay_batch,match_batch,expected_match", [
    ("missing", "missing", None), (None, "missing", None),
    (32, "missing", 32), (32, None, None), (32, 16, 16), (None, 32, 32),
])
def test_stages_forward_inherited_override_or_explicit_opt_out(
        tmp_path, monkeypatch, selfplay_batch, match_batch, expected_match):
    cfg = load_config(REPO_ROOT / "pipeline/configs/ci_tiny.json")
    cfg["selfplay"].update(games=2, games_per_chunk=2)
    for section, value in (("selfplay", selfplay_batch), ("match", match_batch)):
        if value != "missing":
            cfg[section]["infer_batch"] = value
    calls = []

    def child(command, **kwargs):
        calls.append(command)
        options = dict(zip(command[2::2], command[3::2]))
        if command[1] == "selfplay":
            binary = Path(options["--out"])
            binary.write_bytes(bytes(492))
            write_json_atomic(binary.with_suffix(".json"), {
                "format": "chessbot-chunk-v1", "kind": "selfplay", "samples": 1,
                "games": 2, "model": options["--model"], "seed": int(options["--seed"]),
                **{key: int(options["--" + key.replace("_", "-")])
                   for key in ("sims", "max_plies", "temp_plies")},
                "resign": {"threshold": -.95, "playthrough_games": 0, "false_positives": 0}})
        elif command[1] == "match":
            write_json_atomic(Path(options["--out"]), report(options))

    monkeypatch.setattr(stages, "measured_run", child)
    stages.selfplay(tmp_path, 1, cfg, state())
    stages.match(tmp_path, 1, cfg, state())
    candidate = generation_dir(tmp_path, 1)
    write_json_atomic(candidate / "decision.json", {"promoted": True, "reason": "max_games",
        "games": 4, "score": .5, "elo": 0, "elo_ci95": [-1, 1], "llr": .1})
    stages.anchors(tmp_path, 1, cfg, state())

    assert len(calls) == 4  # self-play, evaluation driver, and both anchor engines
    expected_selfplay = None if selfplay_batch in ("missing", None) else selfplay_batch
    for command, expected in zip(calls, [expected_selfplay, expected_match, expected_match, expected_match]):
        assert "None" not in command
        if expected is None:
            assert "--infer-batch" not in command
        else:
            assert command[command.index("--infer-batch") + 1] == str(expected)

    # Backend tuning must not invalidate completed data or change its wire schema.
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in (candidate / "selfplay").glob("chunk_*.*")}
    cfg["selfplay"]["infer_batch"] = 32
    stages.selfplay(tmp_path, 1, cfg, state())
    assert len(calls) == 4
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == saved for path, saved in before.items())


@pytest.mark.parametrize("batch", [None, 32, "legacy"])
def test_evaluation_forwards_only_requested_batch_and_preserves_cached_blocks(tmp_path, monkeypatch, batch):
    args = evaluate.parse_args(arguments(tmp_path, **{"block-games": 4, "max-games": 4}))
    if batch == "legacy":
        del args.infer_batch
    else:
        args.infer_batch = batch
    calls = []
    def child(command, **kwargs):
        calls.append(command)
        options = dict(zip(command[2::2], command[3::2]))
        write_json_atomic(Path(options["--out"]), report(options))
    monkeypatch.setattr(evaluate, "measured_run", child)
    evaluate.evaluate(args)
    assert len(calls) == 1
    if batch == 32:
        assert calls[0][-2:] == ["--infer-batch", "32"]
    else:
        assert "--infer-batch" not in calls[0]
    before = snapshots(args.out_dir)
    assert "infer_batch" not in json.loads(next(args.out_dir.glob("*.json")).read_text())
    args.infer_batch = 32
    evaluate.evaluate(args)
    assert len(calls) == 1 and snapshots(args.out_dir) == before


@pytest.mark.parametrize("value", [True, False, 0, -1, 65, "32", 32.0])
def test_evaluate_namespace_rejects_invalid_inference_batches(tmp_path, value):
    args = evaluate.parse_args(arguments(tmp_path))
    args.infer_batch = value
    with pytest.raises(ValueError, match="infer-batch"):
        evaluate.evaluate(args)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("value", ["0", "65", "true", "32.0"])
def test_evaluate_cli_rejects_invalid_inference_batches(tmp_path, value):
    with pytest.raises(SystemExit):
        evaluate.parse_args(arguments(tmp_path, **{"infer-batch": value}))


def test_saved_generation_manifest_does_not_block_backend_only_migration(tmp_path):
    original = state()
    original["config"] = normalize_config({})
    directory = generation_dir(tmp_path, 1)
    ensure_manifest(directory, Path("engine"), original)
    manifest = directory / "generation.json"
    before = manifest.read_bytes(), manifest.stat().st_mtime_ns
    migrated = deepcopy(original)
    for section in ("selfplay", "match"):
        migrated["config"][section]["infer_batch"] = 32
    ensure_manifest(directory, Path("engine"), migrated)
    assert (manifest.read_bytes(), manifest.stat().st_mtime_ns) == before


def test_tiny_real_loop_accepts_fixed_inference_and_resumes_without_rewriting(
        tmp_path, ci_tiny_bootstrap, engine_bin, repo_root):
    run = tmp_path / "run"
    shutil.copytree(ci_tiny_bootstrap, run)
    saved = read_run_json(run)
    cfg = normalize_config(saved["config"])
    cfg["selfplay"].update(infer_batch=32, games=2, games_per_chunk=2, sims=2,
                           parallel=2, max_plies=4, temp_plies=0)
    cfg["train"].update(steps=1, batch_size=2, steps_per_epoch=1, val_samples=2)
    cfg["match"].update(infer_batch=32, block_games=2, max_games=2, sims=2, batch=1, max_plies=8)
    saved["config"] = cfg
    write_run_json(run, saved)
    command = [sys.executable, str(repo_root / "pipeline/loop.py"), "--run", str(run),
               "--engine", str(engine_bin), "--until-gen", "1"]
    subprocess.run(command, check=True, timeout=120)
    assert read_run_json(run)["current_gen"] == 2
    directory = generation_dir(run, 1)
    outputs = [directory / name for name in ("model.pt", "model.onnx", "train.json", "decision.json")]
    outputs += list((directory / "selfplay").glob("chunk_*.*"))
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in outputs}
    subprocess.run(command, check=True, timeout=15)
    assert all((path.stat().st_size, path.stat().st_mtime_ns) == previous
               for path, previous in before.items())
    assert not list(run.rglob("*.tmp"))
