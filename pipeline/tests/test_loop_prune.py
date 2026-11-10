"""Real pruning workers preserve loop checkpoints across crashes and failures."""

from pathlib import Path
import sys

from run_layout import generation_dir, read_run_json, write_json_atomic, write_run_json
from test_loop import bare_run, invoke, stamps


def generation(run, gen, promoted):
    directory = generation_dir(run, gen)
    directory.mkdir(parents=True)
    for name in ("model.pt", "model.onnx"):
        (directory / name).write_bytes(b"model bytes")
    write_json_atomic(directory / "train.json", {"loss": 0.1})
    write_json_atomic(directory / "decision.json", {
        "promoted": promoted, "reason": "bootstrap" if gen == 0 else "max_games",
        "games": 4, "score": 0.5, "elo": 0.0, "elo_ci95": [-1.0, 1.0], "llr": 0.0,
    })
    write_json_atomic(directory / "match" / "block_00.json", {"games": 4})
    if gen:
        binary = directory / "selfplay" / "chunk_000000.bin"
        binary.parent.mkdir()
        binary.write_bytes(bytes(492))
        write_json_atomic(binary.with_suffix(".json"), {
            "format": "chessbot-chunk-v1", "kind": "selfplay", "samples": 1,
        })
    return directory


def fingerprint(paths):
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}


def test_prune_deletions_survive_crash_before_checkpoint_and_resume(tmp_path):
    run = tmp_path / "run"
    bare_run(run, current_gen=6, best_gen=2, stage="prune")
    expired = generation(run, 1, False)
    best = generation(run, 2, True)
    rejected = generation(run, 3, False)
    promoted = generation(run, 4, True)
    current = generation(run, 6, False)
    protected = [path for directory in (best, current, promoted)
                 for path in directory.rglob("*") if path.is_file()]
    historical = [directory / name for directory in (expired, rejected)
                  for name in ("train.json", "decision.json", "match/block_00.json")]
    before = fingerprint([*protected, *historical])
    checkpoint = (run / "run.json").read_bytes()

    crashed = invoke(run, Path(sys.executable), "--until-gen", 6,
                     crash="6:prune", timeout=10)
    assert crashed.returncode == 17, crashed.stdout + crashed.stderr
    assert (run / "run.json").read_bytes() == checkpoint
    assert (run / "loop.lock").is_file()
    assert not (expired / "selfplay").exists()
    for directory in (expired, rejected):
        assert not (directory / "model.pt").exists()
        assert not (directory / "model.onnx").exists()
    assert (rejected / "selfplay").is_dir()
    assert fingerprint(before) == before

    resumed = invoke(run, Path(sys.executable), "--until-gen", 6, timeout=10)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    state = read_run_json(run)
    assert (state["current_gen"], state["stage"], state["best_gen"]) == (7, "selfplay", 2)
    assert fingerprint(before) == before
    assert not (run / "loop.lock").exists() and not list(run.rglob("*.tmp"))


def test_unreachable_budget_keeps_prune_checkpoint_and_releases_lock(tmp_path):
    run = tmp_path / "run"
    bare_run(run, current_gen=4, best_gen=3, stage="prune",
             config={"disk_budget_gb": 0.000001})
    oldest = generation(run, 0, True)
    rejected = generation(run, 1, False)
    best = generation(run, 3, True)
    current = generation(run, 4, False)
    supervised = run / "supervised" / "chunk_000000.bin"
    supervised.parent.mkdir()
    supervised.write_bytes(bytes(8192))
    protected = [supervised, *[path for directory in (best, current)
                              for path in directory.rglob("*") if path.is_file()]]
    before = fingerprint(protected)
    metadata = stamps([oldest / "decision.json", rejected / "decision.json"])
    checkpoint = (run / "run.json").read_bytes()

    failed = invoke(run, Path(sys.executable), "--until-gen", 4, timeout=10)
    assert failed.returncode == 1, failed.stdout + failed.stderr
    assert "budget" in failed.stderr.lower() and "Traceback" not in failed.stderr
    assert (run / "run.json").read_bytes() == checkpoint
    assert not (run / "loop.lock").exists() and not list(run.rglob("*.tmp"))
    assert not (oldest / "model.pt").exists() and (oldest / "model.onnx").is_file()
    assert not (rejected / "model.pt").exists() and not (rejected / "model.onnx").exists()
    assert fingerprint(before) == before and stamps(metadata) == metadata

    state = read_run_json(run)
    state["config"]["disk_budget_gb"] = 20
    write_run_json(run, state)
    resumed = invoke(run, Path(sys.executable), "--until-gen", 4, timeout=10)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    state = read_run_json(run)
    assert (state["current_gen"], state["stage"]) == (5, "selfplay")
    assert fingerprint(before) == before
    assert not (run / "loop.lock").exists()
