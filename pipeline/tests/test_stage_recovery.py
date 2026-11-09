"""Small synthetic checks for stage state replay and interrupted subprocesses."""

from copy import deepcopy
from pathlib import Path
import subprocess

import pytest

from config import load_config
from run_layout import REPO_ROOT, generation_dir, write_json_atomic
import stages


@pytest.fixture
def cfg():
    return load_config(REPO_ROOT / "pipeline/configs/ci_tiny.json")


def state(threshold=-0.95):
    return {"best_gen": 0, "current_gen": 1, "stage": "selfplay",
            "resign_threshold": threshold}


def models(run):
    gen = generation_dir(run, 0)
    gen.mkdir(parents=True)
    (gen / "model.pt").write_bytes(b"checkpoint")
    (gen / "model.onnx").write_bytes(b"network")
    write_json_atomic(gen / "decision.json", {"promoted": True, "reason": "bootstrap"})


def promotion_decision(promoted):
    return {"promoted": promoted, "reason": "max_games", "games": 8,
            "score": .6 if promoted else .4, "elo": 70 if promoted else -70,
            "elo_ci95": [-20, 120], "llr": 1 if promoted else -1}


def chunk(run, index, cfg, threshold=-0.95, playthrough=2, false_positives=1):
    path = generation_dir(run, 1) / "selfplay" / f"chunk_{index:06d}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(492))
    games = min(cfg["selfplay"]["games_per_chunk"],
                cfg["selfplay"]["games"] - index * cfg["selfplay"]["games_per_chunk"])
    playthrough = min(playthrough, games)
    false_positives = min(false_positives, playthrough)
    write_json_atomic(path.with_suffix(".json"), {
        "format": "chessbot-chunk-v1", "kind": "selfplay", "samples": 1,
        "games": games,
        "model": str(generation_dir(run, 0) / "model.onnx"),
        "seed": 1_000_000 + 1000 * index + cfg["seed"],
        "sims": cfg["selfplay"]["sims"], "max_plies": cfg["selfplay"]["max_plies"],
        "temp_plies": cfg["selfplay"]["temp_plies"],
        "results": {"white": 0, "black": 0, "draw": games},
        "terminations": {"ply_cap": games},
        "resign": {"threshold": threshold, "consecutive": 3,
                   "playthrough_games": playthrough, "false_positives": false_positives},
    })
    return path


@pytest.mark.parametrize("threshold,playthrough,fp,expected", [
    (-0.95, 0, 0, -0.95), (-0.95, 20, 1, -0.95),
    (-0.95, 20, 2, -0.97), (-0.98, 20, 2, -0.99),
    (-0.949999988079071, 20, 2, -0.97),
])
def test_resignation_update_is_replayable(tmp_path, cfg, monkeypatch, threshold, playthrough, fp, expected):
    cfg["selfplay"].update(games=40, games_per_chunk=20)
    models(tmp_path)
    for index in range(2):
        chunk(tmp_path, index, cfg, threshold, playthrough, fp)
    monkeypatch.setattr(stages.subprocess, "run", lambda *a, **k: pytest.fail("complete selfplay reran"))
    current = state(threshold)
    stages.selfplay(tmp_path, 1, cfg, current)
    assert current["resign_threshold"] == pytest.approx(expected)
    stages.selfplay(tmp_path, 1, cfg, current)
    assert current["resign_threshold"] == pytest.approx(expected)
    saved = deepcopy(current)
    stages.selfplay(tmp_path, 1, cfg, saved)
    assert saved == current
    assert current["best_gen"] == 0 and current["stage"] == "selfplay"
    assert not (tmp_path / "run.json").exists()


def test_repaired_chunk_uses_generation_threshold_and_exact_flags(tmp_path, cfg, monkeypatch):
    models(tmp_path)
    preserved = chunk(tmp_path, 0, cfg)
    stamp = preserved.stat().st_mtime_ns
    calls = []

    def run(command, **kwargs):
        assert kwargs["check"] is True
        options = dict(zip(command[2::2], command[3::2]))
        calls.append(options)
        assert float(options["--resign-threshold"]) == pytest.approx(-0.95)
        chunk(tmp_path, 1, cfg)

    monkeypatch.setattr(stages.subprocess, "run", run)
    current = state(-0.97)
    stages.selfplay(tmp_path, 1, cfg, current)
    assert len(calls) == 1 and calls[0]["--seed"] == "1001000"
    assert calls[0]["--games"] == "2" and calls[0]["--temp-plies"] == "10"
    assert calls[0]["--parallel"] == "4" and calls[0]["--ep"] == "cpu"
    assert calls[0]["--model"] == str(generation_dir(tmp_path, 0) / "model.onnx")
    assert preserved.stat().st_mtime_ns == stamp
    assert current["resign_threshold"] == pytest.approx(-0.97)


def test_selfplay_nondivisible_game_count_runs_short_tail(tmp_path, cfg, monkeypatch):
    models(tmp_path)
    cfg["selfplay"]["games"] = 5
    for index in range(2):
        chunk(tmp_path, index, cfg)
    calls = []

    def run(command, **kwargs):
        options = dict(zip(command[2::2], command[3::2]))
        calls.append(options)
        chunk(tmp_path, 2, cfg, playthrough=1, false_positives=0)

    monkeypatch.setattr(stages.subprocess, "run", run)
    stages.selfplay(tmp_path, 1, cfg, state())
    assert len(calls) == 1
    assert calls[0]["--games"] == "1" and calls[0]["--seed"] == "1002000"
    assert calls[0]["--out"].endswith("chunk_000002.bin")


def test_selfplay_failure_preserves_finished_chunk_and_retries(tmp_path, cfg, monkeypatch):
    models(tmp_path)
    preserved = chunk(tmp_path, 0, cfg)
    before = preserved.stat().st_mtime_ns
    current, calls = state(), []

    def fail(command, **kwargs):
        calls.append(command)
        target = Path(command[command.index("--out") + 1])
        target.with_name(target.name + ".tmp").write_bytes(b"partial")
        raise subprocess.CalledProcessError(17, command)

    monkeypatch.setattr(stages.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        stages.selfplay(tmp_path, 1, cfg, current)
    assert current == state() and preserved.stat().st_mtime_ns == before
    assert not (preserved.parent / "chunk_000001.json").exists()

    def succeed(command, **kwargs):
        calls.append(command)
        chunk(tmp_path, 1, cfg)

    monkeypatch.setattr(stages.subprocess, "run", succeed)
    stages.selfplay(tmp_path, 1, cfg, current)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert preserved.stat().st_mtime_ns == before
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("promoted", [False, True])
def test_existing_decision_updates_best_without_evaluation(tmp_path, cfg, monkeypatch, promoted):
    models(tmp_path)
    decision = generation_dir(tmp_path, 1) / "decision.json"
    write_json_atomic(decision, promotion_decision(promoted))
    before = decision.stat().st_mtime_ns
    monkeypatch.setattr(stages.subprocess, "run", lambda *a, **k: pytest.fail("decision replay ran evaluation"))
    current = state()
    stages.decide(tmp_path, 1, cfg, current)
    assert current["best_gen"] == int(promoted)
    stages.decide(tmp_path, 1, cfg, current)
    assert current["best_gen"] == int(promoted)
    assert decision.stat().st_mtime_ns == before
    assert not (tmp_path / "run.json").exists()


def test_rejected_generation_has_no_anchor_subprocesses(tmp_path, cfg, monkeypatch):
    models(tmp_path)
    write_json_atomic(generation_dir(tmp_path, 1) / "decision.json", promotion_decision(False))
    monkeypatch.setattr(stages.subprocess, "run", lambda *a, **k: pytest.fail("rejected candidate ran anchors"))
    stages.anchors(tmp_path, 1, cfg, state())
    assert not (generation_dir(tmp_path, 1) / "anchors").exists()


def test_rejection_after_retraining_restores_previous_best(tmp_path, cfg, monkeypatch):
    models(tmp_path)
    write_json_atomic(generation_dir(tmp_path, 1) / "decision.json", promotion_decision(False))
    monkeypatch.setattr(stages.subprocess, "run", lambda *a, **k: pytest.fail("saved rejection reran"))
    current = {**state(), "best_gen": 1}
    stages.decide(tmp_path, 1, cfg, current)
    assert current["best_gen"] == 0


def test_promoted_anchors_resume_opponents_independently(tmp_path, cfg, monkeypatch):
    models(tmp_path)
    gen = generation_dir(tmp_path, 1)
    write_json_atomic(gen / "decision.json", promotion_decision(True))
    (gen / "model.onnx").write_bytes(b"candidate")
    calls = []

    def run(command, **kwargs):
        assert kwargs["check"] is True
        options = dict(zip(command[2::2], command[3::2]))
        calls.append(options["--b"])
        games, offset = int(options["--games"]), int(options.get("--opening-offset", 0))
        write_json_atomic(Path(options["--out"]), {
            "format": "chessbot-match-v1", "a": options["--a"], "b": options["--b"],
            "games": games, "a_wins": 0, "b_wins": 0, "draws": games, "a_score": .5,
            "sims": int(options["--sims"]), "seed": int(options["--seed"]),
            "opening_offset": offset, "max_plies": int(options["--max-plies"]),
            "results": [{"game": i, "opening": (offset + i // 2) % 50,
                         "white": "a" if i % 2 == 0 else "b", "result": "1/2-1/2",
                         "termination": "ply_cap", "plies": 10} for i in range(games)],
        })

    monkeypatch.setattr(stages.subprocess, "run", run)
    current = {**state(), "best_gen": 1}
    stages.anchors(tmp_path, 1, cfg, current)
    assert set(calls) == {"random", "material"}
    random = gen / "anchors/vs_random.json"
    stamp = random.stat().st_mtime_ns
    stages.anchors(tmp_path, 1, cfg, current)
    assert len(calls) == 2
    (gen / "anchors/vs_material.json").unlink()
    stages.anchors(tmp_path, 1, cfg, current)
    assert calls[-1] == "material" and len(calls) == 3
    assert random.stat().st_mtime_ns == stamp
    assert not list(tmp_path.rglob("*.tmp"))
