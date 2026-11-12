"""Durable attempt ledgers; stage spans and process resource peaks stay separate."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import uuid

from run_layout import read_json, write_json_atomic

FORMAT = "chessbot-metrics-v1"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def begin_attempt(directory, *, kind, stage, **meta):
    """Publish a unique started record before work. Unfinished records stay visible."""
    if kind not in ("stage", "process", "worker") or not isinstance(stage, str) or not stage:
        raise ValueError("a telemetry attempt requires a valid kind and stage")
    identity = uuid.uuid4().hex
    path = Path(directory) / f"{kind}_{identity}.json"
    record = {**meta, "format": FORMAT, "attempt_id": identity, "kind": kind,
              "stage": stage, "status": "running", "started_at": _now()}
    write_json_atomic(path, record)
    return path, record


def finish_attempt(path, record, status, seconds, **fields):
    """Atomically finalize one invocation without changing previous attempt records."""
    if status not in ("complete", "interrupted", "failed") or not _number(seconds):
        raise ValueError("invalid attempt status or duration")
    result = {**record, **fields, "status": status, "seconds": seconds,
              "finished_at": _now()}
    write_json_atomic(path, result)
    return result


def read_attempts(generation_dir):
    """Return ledger records; malformed records remain explicit incomplete evidence."""
    result = []
    for path in sorted((Path(generation_dir) / "telemetry").glob("*.json")):
        try:
            value = read_json(path)
            if (value.get("format") != FORMAT or value.get("kind") not in
                    ("stage", "process", "worker") or not isinstance(value.get("stage"), str)):
                raise ValueError("invalid telemetry record")
            result.append(value)
        except (OSError, ValueError):
            result.append({"kind": "invalid", "path": str(path), "status": "incomplete"})
    return result


def stages_summary(generation_dir):
    """Sum stage attempts once and combine measured process peaks, never durations.

    A started or corrupt stage makes elapsed time unknown. RSS is complete only
    when each stage invocation has a worker peak and all spawned workloads have
    resource measurements; failed spawn attempts may set ``rss_required=False``.
    Legacy generations without a ledger return no invented zero-valued metrics.
    """
    records = read_attempts(generation_dir)
    corrupt = any(record["kind"] == "invalid" for record in records)
    result = {}
    for stage in sorted({record["stage"] for record in records if "stage" in record}):
        group = [record for record in records if record.get("stage") == stage]
        spans = [record for record in group if record["kind"] == "stage"]
        if not spans:
            continue
        finalized = lambda record: (record.get("status") in ("complete", "failed", "interrupted")
                                    and _number(record.get("seconds")))
        covered = all(record.get("coverage_complete", True) is True for record in spans)
        seconds = (sum(record["seconds"] for record in spans)
                   if not corrupt and covered and all(finalized(record) for record in spans) else None)
        peaks = [record[key] for record in group
                 for key in ("peak_rss_bytes", "worker_peak_rss_bytes")
                 if _number(record.get(key)) and record[key] > 0]
        measured_workers = {record.get("stage_attempt", record.get("stage_attempt_id"))
                            for record in group if record["kind"] == "worker"
                            and finalized(record) and _number(record.get("peak_rss_bytes"))
                            and record["peak_rss_bytes"] > 0}
        workers_complete = all(
            record.get("rss_required") is False
            or (_number(record.get("worker_peak_rss_bytes")) and record["worker_peak_rss_bytes"] > 0)
            or record.get("attempt_id") in measured_workers for record in spans)
        processes = [record for record in group if record["kind"] == "process"]
        resources_complete = all(
            record.get("rss_required") is False
            or (finalized(record) and _number(record.get("peak_rss_bytes"))
                and record["peak_rss_bytes"] > 0) for record in processes)
        result[stage] = {"seconds": seconds, "peak_rss_bytes": max(peaks) if peaks else None,
                         "rss_complete": (not corrupt and seconds is not None
                                          and workers_complete and resources_complete),
                         "attempts": len(spans)}
    return result
