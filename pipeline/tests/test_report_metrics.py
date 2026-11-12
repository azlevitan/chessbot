"""Reports use durable stage costs and retain archived selfplay counts."""

import shutil

from report import write_report
from telemetry import begin_attempt, finish_attempt
from test_report import chunk, generation, put, rows


def measured(directory, stage, seconds, status="complete", covered=True):
    path, record = begin_attempt(directory / "telemetry", kind="stage", stage=stage,
                                 coverage_complete=covered)
    finish_attempt(path, record, status, seconds, worker_peak_rss_bytes=100)
    child_path, child = begin_attempt(directory / "telemetry", kind="process", stage=stage,
                                      stage_attempt=record["attempt_id"], unit="workload")
    finish_attempt(child_path, child, status, seconds / 2, peak_rss_bytes=200)


def test_report_fills_existing_seventeen_columns_with_all_stage_attempts(tmp_path):
    generation(tmp_path, 0)
    directory = generation(tmp_path, 1)
    put(directory / "train.json", {"seconds": 5, "final_loss": 1.5,
                                     "val": {"policy_top1": 0.5}})
    measured(directory, "selfplay", 3, "interrupted")
    measured(directory, "selfplay", 7)
    measured(directory, "train", 12)
    measured(directory, "match", 4)
    write_report(tmp_path)
    row = rows(tmp_path)[1]
    assert len(row) == 17
    assert float(row["selfplay_secs"]) == 10
    assert float(row["train_secs"]) == 12
    assert float(row["match_secs"]) == 4
    assert float(row["train_loss"]) == 1.5
    before = (tmp_path / "report.tsv").stat().st_mtime_ns
    write_report(tmp_path)
    assert (tmp_path / "report.tsv").stat().st_mtime_ns == before


def test_legacy_and_incomplete_coverage_are_never_reported_as_zero(tmp_path):
    generation(tmp_path, 0)
    legacy = generation(tmp_path, 1)
    put(legacy / "train.json", {"seconds": 5})
    incomplete = generation(tmp_path, 2)
    measured(incomplete, "selfplay", 0.01, covered=False)
    begin_attempt(incomplete / "telemetry", kind="stage", stage="match")
    write_report(tmp_path)
    old, current = rows(tmp_path)[1], rows(tmp_path)[2]
    assert old["selfplay_secs"] == old["match_secs"] == "NA"
    assert float(old["train_secs"]) == 5
    assert current["selfplay_secs"] == current["match_secs"] == "NA"


def test_snapshot_counts_and_stage_costs_survive_selfplay_pruning(tmp_path):
    generation(tmp_path, 0)
    directory = generation(tmp_path, 1)
    chunk(directory, 0, 4)
    chunk(directory, 1, 6)
    put(directory / "selfplay_summary.json", {
        "format": "chessbot-selfplay-summary-v1",
        "chunks": [{"chunk": 0, "samples": 4, "games": 2},
                   {"chunk": 1, "samples": 6, "games": 2}],
    })
    measured(directory, "selfplay", 10)
    write_report(tmp_path)
    before = rows(tmp_path)[1]
    (directory / "selfplay/chunk_000001.bin").unlink()
    write_report(tmp_path)
    assert rows(tmp_path)[1] == before  # An interrupted prune retains the full snapshot.
    shutil.rmtree(directory / "selfplay")
    write_report(tmp_path)
    assert rows(tmp_path)[1] == before
    assert int(before["selfplay_samples"]) == 10
