"""Durable generation inputs survive loss of every selfplay commit marker."""

from pathlib import Path
import struct

import pytest

from config import load_config
from run_layout import REPO_ROOT, generation_dir, read_json, write_json_atomic
import stages


@pytest.mark.parametrize("games", [2, 4])
def test_all_missing_sidecars_reuse_original_inputs_after_promotion(tmp_path, monkeypatch, games):
    cfg = load_config(REPO_ROOT / "pipeline/configs/ci_tiny.json")
    cfg["selfplay"]["games"] = games
    best = generation_dir(tmp_path, 0)
    best.mkdir(parents=True)
    (best / "model.onnx").write_bytes(b"incumbent")
    write_json_atomic(best / "decision.json", {"promoted": True, "reason": "bootstrap"})
    current = {"best_gen": 0, "resign_threshold": -.95}
    directory = generation_dir(tmp_path, 1) / "selfplay"
    calls = []

    def run(command, **kwargs):
        assert kwargs["check"] is True
        options = dict(zip(command[2::2], command[3::2]))
        calls.append(options)
        # Persist before the child can publish any chunk or crash.
        assert read_json(directory / ".inputs.json")["resign_threshold"] == -.95
        binary = Path(options["--out"])
        binary.write_bytes(bytes(492))
        count = int(options["--games"])
        threshold = struct.unpack("f", struct.pack("f", float(options["--resign-threshold"])))[0]
        write_json_atomic(binary.with_suffix(".json"), {
            "format": "chessbot-chunk-v1", "kind": "selfplay", "samples": 1,
            "model": options["--model"], "games": count,
            **{name: int(options["--" + name.replace("_", "-")])
               for name in ("seed", "sims", "max_plies", "temp_plies")},
            "resign": {"threshold": threshold, "consecutive": 3,
                       "playthrough_games": count, "false_positives": 1},
        })

    monkeypatch.setattr(stages.subprocess, "run", run)
    stages.selfplay(tmp_path, 1, cfg, current)
    assert current["resign_threshold"] == -.97
    inputs = directory / ".inputs.json"
    stamp = inputs.stat().st_mtime_ns
    for sidecar in directory.glob("chunk_*.json"):
        sidecar.unlink()
    current["best_gen"] = 1
    stages.selfplay(tmp_path, 1, cfg, current)
    assert len(calls) == games  # Two passes, two games per chunk.
    assert all(float(call["--resign-threshold"]) == -.95 for call in calls)
    assert all(call["--model"] == str(best / "model.onnx") for call in calls)
    assert current == {"best_gen": 1, "resign_threshold": -.97}
    assert inputs.stat().st_mtime_ns == stamp
    # An interrupted metadata publication cannot perturb an already complete run.
    (directory / ".inputs.json.tmp").write_bytes(b"partial metadata")
    stages.selfplay(tmp_path, 1, cfg, current)
    assert len(calls) == games and inputs.stat().st_mtime_ns == stamp
    assert not list(tmp_path.rglob("*.tmp"))
