"""Real match acceptance, resumable blocks and atomic evaluation decisions."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

import elo
import evaluate


def arguments(directory, **changes):
    values = {"engine": "unused engine", "a": "material", "b": "random",
              "out-dir": directory / "blocks", "decision": directory / "decision.json",
              "sims": 16, "block-games": 40, "max-games": 40, "seed-base": 0,
              "ep": "cpu"}
    values.update(changes)
    return [item for key, value in values.items() if value is not None
            for item in (f"--{key}", str(value))]


def snapshots(directory):
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
            for p in directory.glob("block_*.json")}


def report(options, all_wins=False):
    n = int(options["--games"])
    wins, losses = (n, 0) if all_wins else (n // 2, n // 2)
    offset, cap = int(options["--opening-offset"]), int(options["--max-plies"])
    results = [{"game": i, "opening": (offset + i // 2) % 50,
                "white": "a" if i % 2 == 0 else "b",
                "result": "1-0" if (i < wins) == (i % 2 == 0) else "0-1",
                "termination": "checkmate", "plies": min(8, cap)} for i in range(n)]
    return {"format": "chessbot-match-v1", "a": options["--a"], "b": options["--b"],
            "games": n, "a_wins": wins, "b_wins": losses, "draws": 0,
            "a_score": wins / n, "sims": int(options["--sims"]),
            "seed": int(options["--seed"]), "opening_offset": offset,
            "max_plies": cap, "results": results}


def test_real_promotion_resume_and_zero_call_decision_rebuild(tmp_path, engine_bin, repo_root):
    args = arguments(tmp_path, engine=engine_bin, **{"block-games": 10})
    subprocess.run([sys.executable, str(repo_root / "pipeline" / "evaluate.py"), *args],
                   cwd=tmp_path, check=True, timeout=60)
    decision_path = tmp_path / "decision.json"
    before = decision_path.read_bytes()
    decision = json.loads(before)
    assert decision["promoted"] is True and decision["reason"] == "sprt_accept"
    assert 0 < decision["games"] <= 40 and decision["games"] % 10 == 0
    assert set(decision) == {"promoted", "reason", "games", "score", "elo", "elo_ci95", "llr"}
    blocks = snapshots(tmp_path / "blocks")
    assert blocks and evaluate.main(args) == 0
    assert snapshots(tmp_path / "blocks") == blocks
    calls = tmp_path / "calls"
    calls.write_text("0", encoding="utf-8")
    wrapper = tmp_path / "engine wrapper"
    wrapper.write_text(f"#!{sys.executable}\nfrom pathlib import Path\np = Path({str(calls)!r})\n"
                       "p.write_text(str(int(p.read_text()) + 1))\nraise SystemExit(91)\n",
                       encoding="utf-8")
    wrapper.chmod(0o755)
    decision_path.unlink()
    assert evaluate.main(arguments(tmp_path, engine=wrapper, **{"block-games": 10})) == 0
    assert calls.read_text() == "0" and decision_path.read_bytes() == before
    assert snapshots(tmp_path / "blocks") == blocks
    assert not list(tmp_path.rglob("*.tmp"))


def test_real_uniform_pair_is_rejected_with_cap_below_default_block(tmp_path, engine_bin):
    assert evaluate.main(arguments(tmp_path, engine=engine_bin, a="uniform", b="uniform",
                                   **{"block-games": None, "max-games": 20})) == 0
    decision = json.loads((tmp_path / "decision.json").read_text())
    block = json.loads((tmp_path / "blocks" / "block_00.json").read_text())
    assert decision["promoted"] is False and decision["score"] == 0.5
    assert decision["games"] == block["games"] == 20


def test_exact_arguments_short_final_block_and_cumulative_statistics(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        options = dict(zip(command[2::2], command[3::2]))
        calls.append((command, options, kwargs))
        Path(options["--out"]).write_text(json.dumps(report(options)))

    monkeypatch.setattr(evaluate.subprocess, "run", run)
    args = arguments(tmp_path, a="model with spaces.onnx", **{
        "block-games": 6, "max-games": 10, "seed-base": 73, "sims": 32,
        "batch": 3, "max-plies": 60, "threads": 2, "elo0": -10, "elo1": 30,
        "alpha": 0.1, "beta": 0.2})
    decision = evaluate.evaluate(evaluate.parse_args(args))
    assert len(calls) == 2
    for b, (command, options, kwargs) in enumerate(calls):
        assert command[:2] == ["unused engine", "match"] and kwargs["check"] is True
        assert options == {"--a": "model with spaces.onnx", "--b": "random",
                           "--games": str(6 if b == 0 else 4), "--sims": "32",
                           "--seed": str(73 + b), "--opening-offset": str(20 * b),
                           "--out": str(tmp_path / "blocks" / f"block_{b:02d}.json.tmp"),
                           "--batch": "3", "--max-plies": "60", "--ep": "cpu",
                           "--threads": "2"}
    assert decision == {"promoted": False, "reason": "max_games", "games": 10,
                        "score": 0.5, "elo": 0.0, "elo_ci95": list(elo.elo_ci95(5, 0, 5)),
                        "llr": elo.llr(5, 0, 5, -10, 30)}


@pytest.mark.parametrize("corruption", ["not JSON", {"promoted": 1}, {"games": 4.0}])
def test_resume_stops_at_earliest_cached_decision_and_repairs_it(tmp_path, monkeypatch, corruption):
    directory = tmp_path / "blocks"
    directory.mkdir()
    options = {"--a": "material", "--b": "random", "--games": "4", "--sims": "16",
               "--seed": "0", "--opening-offset": "0", "--max-plies": "300"}
    (directory / "block_00.json").write_text(json.dumps(report(options, all_wins=True)))
    (directory / "block_01.json").write_text("unused later block")
    monkeypatch.setattr(evaluate.subprocess, "run", lambda *a, **k: pytest.fail("engine called"))
    args = evaluate.parse_args(arguments(tmp_path, **{"block-games": 4}))
    decision = evaluate.evaluate(args)
    assert decision["reason"] == "sprt_accept" and decision["games"] == 4
    path = tmp_path / "decision.json"
    canonical = path.read_bytes()
    path.write_text(corruption if isinstance(corruption, str)
                    else json.dumps({**decision, **corruption}))
    assert evaluate.evaluate(args) == decision
    assert path.read_bytes() == canonical


def test_child_failure_preserves_complete_blocks_and_retries_partial_block(tmp_path, monkeypatch):
    seeds = []
    failing = True

    def run(command, **kwargs):
        options = dict(zip(command[2::2], command[3::2]))
        seeds.append(int(options["--seed"]))
        path = Path(options["--out"])
        path.write_text("partial JSON")
        if failing and options["--seed"] == "1":
            raise subprocess.CalledProcessError(17, command)
        path.write_text(json.dumps(report(options)))

    monkeypatch.setattr(evaluate.subprocess, "run", run)
    args = arguments(tmp_path, **{"block-games": 4, "max-games": 8})
    assert evaluate.main(args) != 0
    directory = tmp_path / "blocks"
    first = snapshots(directory)
    assert list(first) == ["block_00.json"] and not (tmp_path / "decision.json").exists()
    (directory / "block_01.json.tmp").write_text("stale")
    failing = False
    assert evaluate.main(args) == 0 and seeds == [0, 1, 1]
    assert snapshots(directory)["block_00.json"] == first["block_00.json"]
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("change", [
    {"format": "wrong"}, {"a": "other"}, {"games": 2}, {"a_wins": True},
    {"draws": -1}, {"b_wins": 0}, {"seed": 5}, {"sims": 2},
    {"opening_offset": 20}, {"max_plies": 100}, {"a_score": 0.9}, {"results": []},
    {"a_score": 10**500}, {"a_score": float("nan")}, {"a_score": float("inf")},
])
def test_bad_cached_or_fresh_reports_are_rejected(tmp_path, monkeypatch, change, cached):
    directory = tmp_path / "blocks"
    directory.mkdir()
    options = {"--a": "material", "--b": "random", "--games": "40", "--sims": "16",
               "--seed": "0", "--opening-offset": "0", "--max-plies": "300"}
    bad = {**report(options), **change}

    def run(command, **kwargs):
        assert not cached, "corrupt completed block must not be silently replaced"
        Path(command[command.index("--out") + 1]).write_text(json.dumps(bad))

    monkeypatch.setattr(evaluate.subprocess, "run", run)
    if cached:
        (directory / "block_00.json").write_text(json.dumps(bad))
    assert evaluate.main(arguments(tmp_path)) != 0
    assert not (tmp_path / "decision.json").exists()
    assert (directory / "block_00.json").exists() is cached


@pytest.mark.parametrize("flag,value", [
    ("block-games", "0"), ("block-games", "3"), ("max-games", "-2"), ("max-games", "3"),
    ("sims", "1"), ("batch", "0"), ("threads", "0"), ("max-plies", "0"),
    ("seed-base", "-1"), ("alpha", "nan"), ("alpha", "0"), ("beta", "1"),
    ("elo0", "40"), ("elo1", "inf"), ("ep", "gpu"),
])
def test_invalid_arguments_fail_before_outputs(tmp_path, flag, value):
    with pytest.raises((SystemExit, ValueError)):
        evaluate.parse_args(arguments(tmp_path, **{flag: value}))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", [
    "block_00.json", "block_00.json.tmp", "block_00.json/decision.json",
])
def test_decision_cannot_alias_block_outputs(tmp_path, name):
    args = arguments(tmp_path, decision=tmp_path / "blocks" / name)
    try:
        result = evaluate.main(args)
    except SystemExit as error:
        result = error.code
    assert result != 0 and not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["block_00.json", "block_00.json.tmp"])
def test_dangling_block_symlinks_do_not_hide_reserved_decision_paths(tmp_path, monkeypatch, name):
    directory = tmp_path / "blocks"
    directory.mkdir()
    targets = {directory / "block_00.json": tmp_path / "missing final",
               directory / "block_00.json.tmp": tmp_path / "missing temporary"}
    for path, target in targets.items():
        path.symlink_to(target)
    monkeypatch.setattr(evaluate.subprocess, "run", lambda *a, **k: pytest.fail("engine called"))
    assert evaluate.main(arguments(tmp_path, decision=directory / name)) != 0
    assert set(directory.iterdir()) == set(targets)
    for path, target in targets.items():
        assert path.is_symlink() and path.readlink() == target and not path.exists()
