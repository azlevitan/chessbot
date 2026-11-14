"""Stable rows for training curves, candidate quality and attempt telemetry."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re

from analysis_matches import match_rows, resolve_settings
from analysis_selfplay import _read, selfplay_rows
from telemetry import stages_summary

GEN_COLUMNS = tuple(("gen promoted reason games score elo elo_lo elo_hi ladder_elo "
    "vs_random vs_material train_steps samples_seen train_seconds train_attempt_seconds train_loss "
    "train_policy_loss train_wdl_loss train_probe_loss train_probe_policy_loss "
    "train_probe_wdl_loss train_probe_policy_top1 train_probe_wdl_accuracy train_probe_wdl_brier "
    "train_probe_policy_target_entropy train_probe_policy_kl val_loss val_policy_loss val_wdl_loss "
    "val_policy_top1 val_wdl_accuracy val_wdl_brier val_policy_target_entropy val_policy_kl "
    "selfplay_games selfplay_samples selfplay_draw_rate selfplay_false_positive_rate "
    "selfplay_seconds match_seconds peak_rss_bytes rss_complete recorded_rss_complete generation_complete "
    "attempt_id source_commit "
    "config_source q_ratio train_batch_size train_device sources_json").split())
TRAIN_COLUMNS = tuple(("gen attempt_id attempt_status published phase timestamp steps samples_seen "
    "seconds lr q_ratio train_loss train_policy_loss train_wdl_loss train_probe_loss "
    "train_probe_policy_loss train_probe_wdl_loss train_probe_policy_top1 train_probe_samples "
    "train_probe_wdl_accuracy train_probe_wdl_brier train_probe_policy_target_entropy train_probe_policy_kl "
    "val_loss val_policy_loss val_wdl_loss val_policy_top1 val_samples val_wdl_accuracy val_wdl_brier "
    "val_policy_target_entropy val_policy_kl").split())
STAGE_COLUMNS = tuple(("gen attempt_id stage status started_at finished_at seconds peak_rss_bytes "
                       "rss_complete unit stage_attempt coverage_complete rss_required exit_code "
                       "resource_method resource_platform config_json command_json").split())
SUMMARY_COLUMNS = tuple("gen stage seconds peak_rss_bytes rss_complete attempts".split())
ATTEMPT_COLUMNS = tuple(("gen attempt_id status published started_at finished_at seed device steps "
                         "batch_size q_ratio init_checkpoint metadata_json").split())


def generations(run):
    result = []
    for directory in (Path(run) / "gens").glob("gen_*"):
        match = re.fullmatch(r"gen_([0-9]{4,})", directory.name)
        if (match and directory.is_dir() and not directory.is_symlink()
                and directory.name == f"gen_{int(match[1]):04d}"):
            result.append((int(match[1]), directory))
    return sorted(result)


def _metric(value):
    if type(value) in (int, float):
        try:
            return value if math.isfinite(value) else None
        except OverflowError:
            pass
    return None


def _losses(row, prefix, metrics):
    metrics = metrics if isinstance(metrics, dict) else {}
    for key in ("loss", "policy_loss", "wdl_loss", "policy_top1", "samples", "wdl_accuracy",
                "wdl_brier", "policy_target_entropy", "policy_kl"):
        value = metrics.get(key)
        if key == "loss" and value is None:
            a, b = (_metric(metrics.get(field)) for field in ("policy_loss", "wdl_loss"))
            value = a + b if a is not None and b is not None else None
        row[f"{prefix}_{key}"] = _metric(value)


def _history(directory, gen, summary, warnings):
    published = summary.get("attempt_id")
    path = directory / "train_history.jsonl"
    if isinstance(summary.get("history_file"), str):
        custom = Path(summary["history_file"])
        if not custom.is_absolute():
            custom = directory / custom
        if custom.is_file():
            path = custom
    records = []
    try:
        with path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict) or record.get("format") != "chessbot-train-history-v1":
                        raise ValueError("expected a training history record")
                    records.append(record)
                except ValueError as error:
                    warnings.append(f"{path}:{index}: ignoring incomplete history record ({error})")
    except FileNotFoundError:
        return [], []
    statuses = {record.get("attempt_id"): record.get("event") for record in records}
    rows, attempts = [], []
    identities = list(dict.fromkeys(record.get("attempt_id") for record in records))
    for identity in identities:
        group = [record for record in records if record.get("attempt_id") == identity]
        start = next((record for record in group if record.get("event") == "start"), {})
        metadata = start.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        status = "complete" if identity == published else statuses[identity]
        if status not in ("complete", "interrupted", "failed"):
            status = "unfinished"
        attempt = {key: metadata.get(key) for key in ("seed", "device", "steps", "batch_size", "q_ratio", "init_checkpoint")}
        attempt.update(gen=gen, attempt_id=identity, status=status, published=identity == published,
                       started_at=start.get("timestamp"), metadata_json=json.dumps(metadata, sort_keys=True),
                       finished_at=group[-1].get("timestamp") if statuses[identity] in ("complete", "interrupted", "failed") else None)
        attempts.append(attempt)
    for record in records:
        if record.get("event") != "metrics":
            continue
        row = {key: record.get(key) for key in (
            "attempt_id", "phase", "timestamp", "steps", "samples_seen", "seconds", "lr", "q_ratio")}
        row.update(gen=gen, attempt_status=statuses.get(record.get("attempt_id")),
                   published=record.get("attempt_id") == published if published else False)
        if row["published"]:
            row["attempt_status"] = "complete"
        elif row["attempt_status"] not in ("complete", "failed", "interrupted"):
            row["attempt_status"] = "unfinished"
        for key in ("train", "train_probe", "val"):
            _losses(row, key, record.get(key))
        rows.append(row)
    return rows, attempts


def _telemetry(directory, gen, warnings):
    rows = {"stages": [], "processes": []}
    for path in sorted((directory / "telemetry").glob("*.json")):
        record = _read(path, warnings)
        kind = record.get("kind")
        if record.get("format") != "chessbot-metrics-v1" or kind not in ("stage", "process", "worker"):
            warnings.append(f"{path}: invalid telemetry record")
            continue
        row = {key: record.get(key) for key in STAGE_COLUMNS}
        row["stage_attempt"] = record.get("stage_attempt", record.get("stage_attempt_id"))
        row.update(gen=gen, command_json=json.dumps(record.get("command")) if record.get("command") else None,
                   config_json=json.dumps(record["config"], sort_keys=True) if "config" in record else None)
        rows["stages" if kind == "stage" else "processes"].append(row)
    return rows


def collect(run_dir):
    """Return plot-ready rows and diagnostics; tolerate absent/live optional data."""
    run = Path(run_dir).resolve(strict=True)
    if not run.is_dir():
        raise ValueError(f"{run}: expected a run directory")
    tables = {key: [] for key in (
        "generations", "training", "training_attempts", "matches", "match_games", "selfplay", "stages", "processes", "stage_summary")}
    warnings, best = [], 0.0
    state = _read(run / "run.json", warnings)
    for gen, directory in generations(run):
        manifest = _read(directory / "generation.json", warnings)
        cfg = manifest.get("config", state.get("config", {}))
        cfg = cfg if isinstance(cfg, dict) else {}
        cfg_source = "generation.json" if isinstance(manifest.get("config"), dict) else "run.json.config"
        train_cfg = cfg.get("train", {})
        train_cfg = train_cfg if isinstance(train_cfg, dict) else {}
        decision = _read(directory / "decision.json", warnings)
        train = _read(directory / "train.json", warnings)
        row = {key: decision.get(key) for key in ("promoted", "reason")}
        row.update({key: _metric(decision.get(key)) for key in ("games", "score", "elo")})
        interval = decision.get("elo_ci95")
        if isinstance(interval, list) and len(interval) == 2:
            row.update(elo_lo=_metric(interval[0]), elo_hi=_metric(interval[1]))
        if decision.get("promoted") is True:
            candidate = _metric(decision.get("elo"))
            if gen == 0:
                row["ladder_elo"] = best = 0.0
            elif candidate is not None:
                row["ladder_elo"] = best = best + candidate
        for opponent in ("random", "material"):
            anchor = _read(directory / "anchors" / f"vs_{opponent}.json", warnings)
            row[f"vs_{opponent}"] = _metric(anchor.get("a_score"))
        row.update(gen=gen, train_steps=train.get("steps"), samples_seen=train.get("samples_seen"),
                   train_seconds=_metric(train.get("seconds")), train_attempt_seconds=_metric(train.get("seconds")),
                   train_loss=_metric(train.get("final_loss")),
                   train_policy_loss=_metric(train.get("policy_loss")), train_wdl_loss=_metric(train.get("wdl_loss")),
                   attempt_id=train.get("attempt_id"), source_commit=manifest.get("source_commit"),
                   config_source=cfg_source, q_ratio=train_cfg.get("q_ratio"),
                   train_batch_size=train_cfg.get("batch_size"), train_device=train_cfg.get("device"),
                   sources_json=json.dumps(train.get("sources"), sort_keys=True) if train.get("sources") else None)
        for prefix in ("train_probe", "val"):
            _losses(row, prefix, train.get(prefix))
        history, attempts = _history(directory, gen, train, warnings)
        published = next((attempt for attempt in attempts if attempt["published"]), {})
        for column, key in (("q_ratio", "q_ratio"), ("train_batch_size", "batch_size"),
                            ("train_device", "device")):
            if published.get(key) is not None:
                row[column] = published[key]
        tables["training"].extend(history)
        tables["training_attempts"].extend(attempts)
        chunks = selfplay_rows(directory, gen, warnings)
        tables["selfplay"].extend(chunks)
        if chunks:
            n = sum(chunk["games"] for chunk in chunks)
            draws = [chunk.get("draws") for chunk in chunks]
            audits = [chunk.get(key) for chunk in chunks for key in ("playthrough_games", "false_positives")]
            played = sum(chunk["playthrough_games"] for chunk in chunks) if all(
                type(value) is int for value in audits) else None
            row.update(selfplay_games=n, selfplay_samples=sum(chunk["samples"] for chunk in chunks),
                       selfplay_draw_rate=sum(draws) / n if n and all(type(value) is int for value in draws) else None,
                       selfplay_false_positive_rate=sum(chunk["false_positives"] for chunk in chunks) / played
                       if played else None)
        match_cfg = cfg.get("match", {})
        ledger = _telemetry(directory, gen, warnings)
        match_cfg, match_source = resolve_settings(directory, match_cfg, cfg_source, ledger["stages"], warnings)
        matches, games = match_rows(directory, gen, match_cfg, match_source, warnings)
        tables["matches"].extend(matches)
        tables["match_games"].extend(games)
        for key, rows in ledger.items():
            tables[key].extend(rows)
        summaries = stages_summary(directory)
        tables["stage_summary"].extend(dict(values, gen=gen, stage=stage) for stage, values in summaries.items())
        for stage in ("selfplay", "train", "match"):
            if stage in summaries:
                row[f"{stage}_seconds"] = summaries[stage]["seconds"]
        current = state.get("current_gen")
        complete = (gen < current or (gen == current and state.get("stage") == "done")) if type(current) is int else None
        row["generation_complete"] = complete
        if summaries:
            row["recorded_rss_complete"] = all(values["rss_complete"] for values in summaries.values())
            expected = {"selfplay", "train", "export", "match", "decide", "anchors", "prune"}
            row["rss_complete"] = complete is True and expected <= summaries.keys() and row["recorded_rss_complete"]
            row["peak_rss_bytes"] = max((values["peak_rss_bytes"] for values in summaries.values()
                                         if values["peak_rss_bytes"] is not None), default=None)
        tables["generations"].append(row)
    return tables, warnings
