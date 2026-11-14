"""Attempt histories retain retry costs and distinguish unavailable measurements."""

import json

from telemetry import begin_attempt, finish_attempt, read_attempts, stages_summary


def stage(directory, seconds=3, status="complete", **fields):
    path, record = begin_attempt(directory / "telemetry", kind="stage", stage="selfplay",
                                 coverage_complete=True)
    return finish_attempt(path, record, status, seconds, worker_peak_rss_bytes=100, **fields)


def process(directory, stage_attempt, peak=200, status="complete"):
    path, record = begin_attempt(directory / "telemetry", kind="process", stage="selfplay",
                                 stage_attempt=stage_attempt, unit="chunk_000000")
    if status == "running":
        return record
    return finish_attempt(path, record, status, 2, peak_rss_bytes=peak)


def test_retry_spans_add_once_and_nested_process_seconds_do_not(tmp_path):
    first = stage(tmp_path, 3, "interrupted")
    process(tmp_path, first["attempt_id"], 250, "interrupted")
    second = stage(tmp_path, 5)
    process(tmp_path, second["attempt_id"], 200)

    assert stages_summary(tmp_path) == {"selfplay": {
        "seconds": 8, "peak_rss_bytes": 250, "rss_complete": True, "attempts": 2}}
    assert len(read_attempts(tmp_path)) == 4
    assert len(list((tmp_path / "telemetry").glob("*.json"))) == 4
    assert not list(tmp_path.rglob("*.tmp"))


def test_legacy_generations_do_not_invent_zeroes(tmp_path):
    assert stages_summary(tmp_path) == {}


def test_unfinished_stage_and_legacy_coverage_make_time_unavailable(tmp_path):
    stage(tmp_path)
    begin_attempt(tmp_path / "telemetry", kind="stage", stage="selfplay")
    summary = stages_summary(tmp_path)["selfplay"]
    assert summary["seconds"] is None
    assert not summary["rss_complete"]


def test_incomplete_process_blocks_resource_acceptance(tmp_path):
    record = stage(tmp_path)
    process(tmp_path, record["attempt_id"], status="running")
    summary = stages_summary(tmp_path)["selfplay"]
    assert summary["seconds"] == 3
    assert not summary["rss_complete"]


def test_worker_measurements_can_be_separate_records(tmp_path):
    path, record = begin_attempt(tmp_path / "telemetry", kind="stage", stage="prune")
    finish_attempt(path, record, "complete", 1)
    worker_path, worker = begin_attempt(tmp_path / "telemetry", kind="worker", stage="prune",
                                        stage_attempt=record["attempt_id"])
    finish_attempt(worker_path, worker, "complete", 0.5, peak_rss_bytes=123)
    assert stages_summary(tmp_path)["prune"]["rss_complete"]


def test_missing_worker_and_unmeasured_legacy_work_are_explicit(tmp_path):
    path, record = begin_attempt(tmp_path / "telemetry", kind="stage", stage="train",
                                 coverage_complete=False)
    finish_attempt(path, record, "complete", 0.01, worker_peak_rss_bytes=100)
    summary = stages_summary(tmp_path)["train"]
    assert summary["seconds"] is None
    assert not summary["rss_complete"]


def test_malformed_ledger_blocks_acceptance_without_crashing_report(tmp_path):
    stage(tmp_path)
    (tmp_path / "telemetry/broken.json").write_text("{")
    assert stages_summary(tmp_path)["selfplay"]["seconds"] is None
    assert not stages_summary(tmp_path)["selfplay"]["rss_complete"]
    assert any(record["kind"] == "invalid" for record in read_attempts(tmp_path))


def test_spawn_failure_without_a_child_does_not_require_a_process_peak(tmp_path):
    record = stage(tmp_path, status="failed")
    path, child = begin_attempt(tmp_path / "telemetry", kind="process", stage="selfplay",
                                stage_attempt=record["attempt_id"], rss_required=False)
    finish_attempt(path, child, "failed", 0.01, peak_rss_bytes=None)
    assert stages_summary(tmp_path)["selfplay"]["rss_complete"]


def test_record_is_finalized_atomically_without_mutating_other_attempts(tmp_path):
    first_path, first = begin_attempt(tmp_path / "telemetry", kind="stage", stage="train")
    before = first_path.read_bytes()
    path, record = begin_attempt(tmp_path / "telemetry", kind="stage", stage="train")
    finish_attempt(path, record, "failed", 2, error_type="CalledProcessError")
    assert first_path.read_bytes() == before
    final = json.loads(path.read_text())
    assert final["error_type"] == "CalledProcessError"
    assert final["attempt_id"] == record["attempt_id"] != first["attempt_id"]
