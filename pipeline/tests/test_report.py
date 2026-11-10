"""Phase 9.5 ladder and TSV contracts over small metadata-only fake runs."""

import csv
import json
from pathlib import Path
import shutil

import pytest

from report import build_ladder, write_report
from run_layout import generation_dir

HEADER = ("gen promoted reason games score elo elo_lo elo_hi ladder_elo vs_random "
          "vs_material train_loss val_top1 selfplay_samples selfplay_secs "
          "train_secs match_secs").split()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return path


def generation(run, gen, promoted=True, elo=50, interval=None):
    directory = generation_dir(run, gen)
    decision = ({"promoted": True, "reason": "bootstrap"} if gen == 0 else {
        "promoted": promoted, "reason": "sprt_accept" if promoted else "sprt_reject",
        "games": 40, "score": 0.6 if promoted else 0.4, "elo": elo,
        "elo_ci95": interval if interval is not None else [elo - 20, elo + 20],
        "llr": 3.0 if promoted else -3.0,
    })
    put(directory / "decision.json", decision)
    return directory


def chunk(directory, index, samples, **changes):
    binary = directory / "selfplay" / f"chunk_{index:06d}.bin"
    binary.parent.mkdir(parents=True, exist_ok=True)
    with binary.open("wb") as stream:
        stream.truncate(samples * 492)
    put(binary.with_suffix(".json"), {
        "format": "chessbot-chunk-v1", "kind": "selfplay", "samples": samples,
        **changes,
    })
    return binary


def rows(run):
    with (run / "report.tsv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        assert reader.fieldnames == HEADER
        result = list(reader)
    assert all(set(row) == set(HEADER) and None not in row.values() for row in result)
    return {int(row["gen"]): row for row in result}


def test_promotions_accumulate_and_rejections_remain_in_tsv(tmp_path):
    generation(tmp_path, 3, elo=30, interval=[10, 50])
    generation(tmp_path, 2, promoted=False, elo=-40)
    generation(tmp_path, 0)
    generation(tmp_path, 1, elo=50, interval=[20, 80])
    ladder = build_ladder(tmp_path)
    assert [entry["gen"] for entry in ladder] == [0, 1, 3]
    assert [entry["elo"] for entry in ladder] == [0, 50, 80]
    assert [entry["ci95"] for entry in ladder] == [[0, 0], [20, 80], [60, 100]]
    write_report(tmp_path)
    assert json.loads((tmp_path / "ladder.json").read_text()) == ladder
    report = rows(tmp_path)
    assert list(report) == [0, 1, 2, 3]
    assert [report[g]["promoted"].lower() in ("true", "1") for g in report] == [
        True, True, False, True,
    ]
    assert report[2]["reason"] == "sprt_reject"
    assert int(report[1]["games"]) == 40 and float(report[1]["score"]) == 0.6
    assert float(report[2]["elo"]) == -40 and report[2]["ladder_elo"] == "NA"
    assert [float(report[g]["ladder_elo"]) for g in (0, 1, 3)] == [0, 50, 80]
    assert float(report[3]["elo_lo"]) == 10 and float(report[3]["elo_hi"]) == 50


def test_minimal_bootstrap_and_missing_metrics_are_reportable(tmp_path):
    generation(tmp_path, 0)
    write_report(tmp_path)
    row = rows(tmp_path)[0]
    assert row["reason"] == "bootstrap" and float(row["ladder_elo"]) == 0
    for key in HEADER[3:8] + HEADER[9:]:
        assert row[key] == "NA", key
    assert build_ladder(tmp_path) == [{
        "gen": 0, "elo": 0, "ci95": [0, 0], "vs_random": None, "vs_material": None,
    }]


def test_report_reads_existing_training_and_anchor_metrics(tmp_path):
    generation(tmp_path, 0)
    directory = generation(tmp_path, 1)
    put(directory / "train.json", {"final_loss": 1.25, "seconds": 37.5,
                                     "val": {"policy_top1": 0.42}})
    put(directory / "anchors/vs_random.json", {"a_score": 0.975})
    put(directory / "anchors/vs_material.json", {"a_score": 0.75})
    write_report(tmp_path)
    row = rows(tmp_path)[1]
    for key, expected in {"train_loss": 1.25, "train_secs": 37.5, "val_top1": 0.42,
                          "vs_random": 0.975, "vs_material": 0.75}.items():
        assert float(row[key]) == expected
    assert row["selfplay_secs"] == row["match_secs"] == "NA"
    assert build_ladder(tmp_path)[1]["vs_random"] == 0.975
    assert build_ladder(tmp_path)[1]["vs_material"] == 0.75


def test_samples_count_only_complete_selfplay_chunks_without_reading_binaries(
        tmp_path, monkeypatch):
    generation(tmp_path, 0)
    directory = generation(tmp_path, 1)
    chunk(directory, 0, 3)
    chunk(directory, 1, 7)
    chunk(directory, 2, 100).with_suffix(".json").unlink()
    chunk(directory, 3, 100).write_bytes(b"incomplete")
    chunk(directory, 4, 100, kind="pgn")
    chunk(directory, 5, 100, format="wrong-format")
    (directory / "selfplay/chunk_000006.bin.tmp").write_bytes(b"pending")
    put(directory / "selfplay/chunk_000006.json.tmp", {"samples": 100})
    (directory / "model.pt").write_bytes(b"model")
    (directory / "model.onnx").write_bytes(b"model")
    original_open = Path.open

    def metadata_only(path, *args, **kwargs):
        if path.suffix in (".bin", ".pt", ".onnx"):
            pytest.fail(f"report opened binary data: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", metadata_only)
    write_report(tmp_path)
    assert int(rows(tmp_path)[1]["selfplay_samples"]) == 10


def test_rebuild_after_pruning_preserves_ladder_and_training_metrics(tmp_path):
    generation(tmp_path, 0)
    directory = generation(tmp_path, 1)
    put(directory / "train.json", {"final_loss": 0.25, "seconds": 15, "val": None})
    chunk(directory, 0, 20)
    for name in ("model.pt", "model.onnx"):
        (directory / name).write_bytes(b"model")
    write_report(tmp_path)
    before = build_ladder(tmp_path)
    assert int(rows(tmp_path)[1]["selfplay_samples"]) == 20
    shutil.rmtree(directory / "selfplay")
    (directory / "model.pt").unlink()
    (directory / "model.onnx").unlink()
    write_report(tmp_path)
    assert build_ladder(tmp_path) == before
    row = rows(tmp_path)[1]
    assert row["selfplay_samples"] == row["val_top1"] == "NA"
    assert float(row["train_loss"]) == 0.25 and float(row["train_secs"]) == 15


def test_generation_order_is_numeric_and_pending_decisions_are_ignored(tmp_path):
    generation(tmp_path, 0)
    generation(tmp_path, 10000, elo=30)
    generation(tmp_path, 9999, elo=50)
    pending = generation_dir(tmp_path, 2)
    put(pending / "decision.json.tmp", {"promoted": True, "elo": 1000})
    put(pending / "train.json", {"final_loss": 1})
    write_report(tmp_path)
    assert list(rows(tmp_path)) == [0, 9999, 10000]
    assert [entry["elo"] for entry in build_ladder(tmp_path)] == [0, 50, 80]


@pytest.mark.parametrize("value", [None, True, "0.8", float("nan"), float("inf")])
def test_unusable_optional_numeric_metrics_are_na(tmp_path, value):
    generation(tmp_path, 0)
    directory = generation(tmp_path, 1)
    put(directory / "anchors/vs_random.json", {"a_score": value})
    put(directory / "train.json", {"final_loss": value, "seconds": value,
                                     "val": {"policy_top1": value}})
    write_report(tmp_path)
    row = rows(tmp_path)[1]
    for key in ("vs_random", "train_loss", "train_secs", "val_top1"):
        assert row[key] == "NA", key
    assert build_ladder(tmp_path)[1]["vs_random"] is None
