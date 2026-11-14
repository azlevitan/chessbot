"""Useful, reproducible diagnostics from synthetic live and pruned run artifacts."""

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from analysis_data import collect
from analysis_selfplay import archive_selfplay, snapshot_run_selfplay
from analyze import export_analysis
import elo


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def generation(run, gen, *, promoted=None):
    directory = run / "gens" / f"gen_{gen:04d}"
    directory.mkdir(parents=True)
    if promoted is not None:
        write(directory / "decision.json", {"promoted": promoted,
              "reason": "bootstrap" if gen == 0 else "max_games", "games": 4,
              "score": 0.75, "elo": 50, "elo_ci95": [10, 90]})
    return directory


def chunk(directory, index=0, *, incomplete=False):
    path = directory / "selfplay" / f"chunk_{index:06d}.json"
    metadata = {"format": "chessbot-chunk-v1", "kind": "selfplay", "samples": 20, "games": 2,
                "model": "best.onnx", "seed": index, "sims": 200, "max_plies": 512, "temp_plies": 30,
                "results": {"white": 1, "black": 0, "draw": 1, "extra": "ignored"},
                "terminations": {"checkmate": 1, "resign": 0, "rule_draw": 0, "repetition": 1,
                                 "insufficient_material": 0, "ply_cap": 0},
                "resign": {"threshold": -0.95, "playthrough_games": 1, "false_positives": 0}}
    write(path, metadata)
    with path.with_suffix(".bin").open("wb") as stream:
        stream.truncate(492 * 20 - int(incomplete))
    return path


def metrics(attempt, step, *, event="metrics", **fields):
    return {"format": "chessbot-train-history-v1", "attempt_id": attempt, "event": event,
            "phase": "baseline" if step == 0 else "epoch", "steps": step, "samples_seen": step * 16,
            "seconds": step, "lr": 0.001, "q_ratio": 0.3, "train": {"loss": 0.4},
            "train_probe": {"policy_loss": 0.1, "wdl_loss": 0.2, "policy_top1": 0.9,
                            "policy_kl": 0.08, "policy_target_entropy": 0.02},
            "val": {"policy_loss": 0.5, "wdl_loss": 0.7, "policy_top1": 0.6, "wdl_brier": 0.4},
            **fields}


def ledger(directory, kind, identity, status="complete", **fields):
    value = {"format": "chessbot-metrics-v1", "kind": kind, "attempt_id": identity,
             "stage": "selfplay", "status": status, "seconds": 10, **fields}
    write(directory / "telemetry" / f"{kind}_{identity}.json", value)


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_training_attempts_hard_label_curves_and_live_tail(tmp_path):
    directory = generation(tmp_path, 1)
    write(directory / "generation.json", {"config": {"train": {
        "q_ratio": 0.1, "batch_size": 16, "device": "cpu"}}})
    write(directory / "train.json", {"attempt_id": "new", "seconds": 8, "final_loss": 0.4})
    history = [metrics("old", 0), metrics("old", 3), metrics("old", 3, event="failed"),
               metrics("new", 0, event="start", metadata={"q_ratio": 0.3, "batch_size": 32, "device": "mps"}),
               metrics("new", 0), metrics("new", 5)]
    (directory / "train_history.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in history) + '{"partial":', encoding="utf-8")
    out = export_analysis(tmp_path)
    training = rows(out / "training.csv")
    assert len(training) == 4
    assert [row["attempt_status"] for row in training] == ["failed", "failed", "complete", "complete"]
    assert [row["published"] for row in training] == ["False", "False", "True", "True"]
    assert float(training[-1]["train_loss"]) == 0.4
    assert float(training[-1]["train_probe_loss"]) == pytest.approx(0.3)
    assert float(training[-1]["val_loss"]) == pytest.approx(1.2)
    assert float(training[-1]["val_wdl_brier"]) == 0.4
    assert float(training[-1]["train_probe_policy_kl"]) == 0.08
    candidate = rows(out / "generations.csv")[0]
    assert (candidate["q_ratio"], candidate["train_batch_size"], candidate["train_device"]) == ("0.3", "32", "mps")
    assert "incomplete history" in (out / "diagnostics.json").read_text()
    assert "Fixed-probe and held-out policy loss" in (out / "index.html").read_text()
    assert "can also differ" in (out / "index.html").read_text()


def test_selfplay_snapshot_survives_pruning_and_partial_deletion_retry(tmp_path):
    directory = generation(tmp_path, 1, promoted=False)
    chunk(directory, 0)
    chunk(directory, 1)
    chunk(directory, 2, incomplete=True)
    (directory / "selfplay/chunk_000003.json.tmp").write_text("partial")
    archive_selfplay(directory, expected_games=4)
    snapshot = directory / "selfplay_summary.json"
    before = snapshot.read_bytes(), snapshot.stat().st_mtime_ns
    (directory / "selfplay/chunk_000000.bin").unlink()
    snapshot_run_selfplay(tmp_path)
    assert (snapshot.read_bytes(), snapshot.stat().st_mtime_ns) == before
    shutil.rmtree(directory / "selfplay")
    out = export_analysis(tmp_path)
    samples = rows(out / "selfplay.csv")
    assert len(samples) == 2
    assert {row["source"] for row in samples} == {"snapshot"}
    assert all(float(row["samples_per_game"]) == 10 for row in samples)
    summary = rows(out / "generations.csv")[0]
    assert summary["selfplay_games"] == "4"
    assert summary["selfplay_samples"] == "40"
    assert summary["selfplay_draw_rate"] == "0.5"
    assert summary["selfplay_false_positive_rate"] == "0.0"


def test_snapshot_requires_requested_complete_game_count(tmp_path):
    directory = generation(tmp_path, 1)
    chunk(directory, incomplete=True)
    with pytest.raises(ValueError, match="incomplete"):
        archive_selfplay(directory, expected_games=2)
    assert not (directory / "selfplay_summary.json").exists()


def test_optional_diagnostics_do_not_reject_accepted_legacy_chunks(tmp_path):
    directory = generation(tmp_path, 1)
    sidecar = chunk(directory)
    metadata = json.loads(sidecar.read_text())
    del metadata["results"]
    metadata["terminations"] = {"ply_cap": 2}
    write(sidecar, metadata)
    archive_selfplay(directory, expected_games=2)
    tables, warnings = collect(tmp_path)
    assert tables["generations"][0]["selfplay_samples"] == 20
    assert tables["generations"][0]["selfplay_draw_rate"] is None
    assert tables["selfplay"][0]["checkmate"] is None
    assert tables["selfplay"][0]["ply_cap"] == 2
    assert warnings


def test_numeric_generation_order_candidates_and_no_model_or_sample_reads(tmp_path, monkeypatch):
    write(tmp_path / "run.json", {"config": {"match": {"elo0": 10, "elo1": 60}}})
    for gen, promoted in ((10, False), (2, True), (0, True), (3, None)):
        directory = generation(tmp_path, gen, promoted=promoted)
        write(directory / "train.json", {"val": {"policy_loss": 0.4, "wdl_loss": 0.6, "policy_top1": 0.7}})
        chunk(directory)
    original = Path.open
    def protected(path, *args, **kwargs):
        if path.suffix in (".bin", ".pt", ".onnx"):
            raise AssertionError("analysis must not open sample/model contents")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", protected)
    tables, warnings = collect(tmp_path)
    assert [row["gen"] for row in tables["generations"]] == [0, 2, 3, 10]
    assert [row.get("ladder_elo") for row in tables["generations"]] == [0, 50, None, None]
    assert tables["generations"][-1]["promoted"] is False
    assert tables["generations"][-1]["val_loss"] == 1
    assert warnings == []


def test_cumulative_match_evidence_uses_generation_config(tmp_path):
    write(tmp_path / "run.json", {"config": {"match": {"elo0": 100, "elo1": 200}}})
    directory = generation(tmp_path, 1)
    write(directory / "generation.json", {"config": {"match": {"elo0": 0, "elo1": 40}}})
    for block, counts in enumerate(((2, 1, 1), (1, 2, 1))):
        w, d, l = counts
        write(directory / "match" / f"block_{block:02d}.json", {
            "format": "chessbot-match-v1", "a": "candidate.onnx", "b": "best.onnx", "games": 4,
            "a_wins": w, "draws": d, "b_wins": l,
            "results": [{"game": 0, "result": "1-0", "white": "a", "plies": 12}]})
    tables, _ = collect(tmp_path)
    last = tables["matches"][-1]
    assert last["games"] == 8
    assert last["score"] == elo.score(3, 3, 2)
    assert last["llr"] == elo.llr(3, 3, 2, 0, 40)
    assert last["settings_source"] == "generation.json"
    assert len(tables["match_games"]) == 2


def test_stage_totals_count_interruption_once_and_missing_resources(tmp_path):
    directory = generation(tmp_path, 1)
    ledger(directory, "stage", "first", "interrupted", seconds=7)
    ledger(directory, "worker", "w1", "interrupted", stage_attempt="first", peak_rss_bytes=120)
    ledger(directory, "stage", "second", seconds=13)
    ledger(directory, "worker", "w2", stage_attempt_id="second", peak_rss_bytes=150)
    ledger(directory, "process", "p1", stage_attempt="first", seconds=6, peak_rss_bytes=200)
    ledger(directory, "process", "p2", stage_attempt="second", seconds=12, peak_rss_bytes=None)
    tables, _ = collect(tmp_path)
    row = tables["generations"][0]
    assert row["selfplay_seconds"] == 20
    assert row["peak_rss_bytes"] == 200
    assert row["rss_complete"] is False
    assert tables["stage_summary"][0]["attempts"] == 2
    workers = {row["attempt_id"]: row for row in tables["processes"]}
    assert workers["w1"]["stage_attempt"] == "first"
    assert workers["w2"]["stage_attempt"] == "second"
    ledger(directory, "stage", "third", "running")
    assert collect(tmp_path)[0]["generations"][0]["selfplay_seconds"] is None


def test_export_idempotent_empty_tables_and_separate_output(tmp_path):
    generation(tmp_path, 1)
    out = export_analysis(tmp_path)
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in out.iterdir()}
    export_analysis(tmp_path)
    assert {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in out.iterdir()} == before
    assert rows(out / "training.csv") == []
    assert rows(out / "generations.csv")[0]["val_loss"] == "NA"
    assert "No measurements available yet" in (out / "index.html").read_text()
    assert not list(out.glob("*.tmp"))
    for target in (tmp_path, tmp_path / "gens/gen_0001", tmp_path / "supervised"):
        with pytest.raises(ValueError, match="separate"):
            export_analysis(tmp_path, target)


def test_cli_stdlib_help_and_export(tmp_path):
    generation(tmp_path, 1)
    script = Path(__file__).resolve().parents[1] / "analyze.py"
    result = subprocess.run([sys.executable, "-S", str(script), "--run", str(tmp_path)],
                            capture_output=True, text=True, check=True)
    assert "Saved analysis:" in result.stdout
    assert (tmp_path / "analysis/index.html").exists()


def test_historic_match_settings_are_auditable_or_explicitly_unknown(tmp_path):
    directory = generation(tmp_path, 1)
    write(directory / "match/block_00.json", {"format": "chessbot-match-v1", "games": 4,
          "a_wins": 2, "draws": 1, "b_wins": 1})
    tables, warnings = collect(tmp_path)
    assert tables["matches"][0]["llr"] is None
    assert "not captured" in " ".join(warnings)
    ledger(directory, "stage", "one", stage="match", config={"match": {"elo0": 0, "elo1": 40}})
    tables, _ = collect(tmp_path)
    assert tables["matches"][0]["llr"] == elo.llr(2, 1, 1, 0, 40)
    assert tables["matches"][0]["settings_source"] == "telemetry.match.config"
    assert json.loads(tables["stages"][0]["config_json"])["match"]["elo1"] == 40
    ledger(directory, "stage", "two", stage="match", config={"match": {"elo0": 0, "elo1": 80}})
    tables, warnings = collect(tmp_path)
    assert tables["matches"][0]["llr"] is None
    assert tables["matches"][0]["settings_source"] == "ambiguous"
    assert "configs differ" in " ".join(warnings)
