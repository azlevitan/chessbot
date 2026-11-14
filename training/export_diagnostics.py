"""Diagnose a stopped RL export without publishing a model or changing run state."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from config import normalize_config
from loop_lock import LoopLockError, run_lock
from run_layout import read_json, read_run_json, write_json_atomic
from set_inference_batch import _chunks, _fingerprint

PROFILE_NAMES = ("baseline", "single", "no_ort", "no_fold", "no_fold_no_ort")


def _run_profile(profile, run, evidence, timeout, failing_row):
    report = evidence / "profiles" / profile / "report.json"
    stdout = evidence / (profile + ".stdout.log")
    stderr = evidence / (profile + ".stderr.log")
    command = [sys.executable, str(ROOT / "training/export_probe_cases.py"),
               "--checkpoint", str(evidence / "backup/model.pt"), "--run", str(run),
               "--evidence", str(evidence), "--profile", profile,
               "--failing-row", str(failing_row), "--report", str(report)]
    print(f"Checking {profile} (limit {timeout:g}s)...", flush=True)
    try:
        with stdout.open("w") as out, stderr.open("w") as err:
            result = subprocess.run(command, stdout=out, stderr=err, timeout=timeout)
        if result.returncode:
            if report.exists():
                partial = read_json(report)
                partial.update(completed=False, gate_pass=False, report_path=str(report),
                               worker_exit_code=result.returncode, stdout=str(stdout), stderr=str(stderr))
                return partial
            raise ValueError(f"worker exited {result.returncode}; see {stderr}")
        record = read_json(report)
        if (record.get("profile") != profile or record.get("completed") is not True
                or type(record.get("gate_pass")) is not bool):
            raise ValueError(f"incomplete profile report: {report}")
        return record
    except subprocess.TimeoutExpired:
        return {"profile": profile, "completed": False, "gate_pass": False,
                "error": f"exceeded {timeout:g}s", "stdout": str(stdout), "stderr": str(stderr),
                "partial_cases": str(report.parent / "cases.jsonl")}
    except (OSError, ValueError) as error:
        return {"profile": profile, "completed": False, "gate_pass": False,
                "error": str(error), "stdout": str(stdout), "stderr": str(stderr)}


def _output_path(out_dir):
    if out_dir is not None:
        return Path(out_dir).expanduser().resolve()
    base = Path.home() / "Library/Logs/chessbot" if sys.platform == "darwin" else ROOT / "work"
    return base / datetime.now(timezone.utc).strftime("export-diagnostics-%Y%m%dT%H%M%S%fZ")


def _protect_validation(run, files):
    total = 0
    for binary in sorted((run / "supervised_val").glob("*.bin")):
        sidecar = binary.with_suffix(".json")
        metadata = read_json(sidecar)
        count = metadata.get("samples")
        if (metadata.get("format") != "chessbot-chunk-v1" or type(count) is not int
                or count < 0 or binary.stat().st_size != count * 492):
            raise ValueError(f"{binary}: invalid held-out chunk")
        total += count
        for path in (binary, sidecar):
            files[str(path.relative_to(run))] = _fingerprint(path)
    if total == 0:
        raise ValueError("held-out parity source must contain complete nonempty chunks")


def diagnose(run_dir, out_dir=None, timeout=60, failing_row=2):
    """Keep the run locked while recording evidence and running bounded workers."""
    if type(timeout) not in (int, float) or not 0 < timeout <= 300:
        raise ValueError("timeout must be in (0, 300] seconds")
    if type(failing_row) is not int or not 0 <= failing_row < 7:
        raise ValueError("failing row must be in 0..6")
    run = Path(run_dir).expanduser().resolve(strict=True)
    evidence = _output_path(out_dir)
    if evidence == run or run in evidence.parents:
        raise ValueError("diagnostic output must be outside the run")
    if evidence.exists():
        raise FileExistsError(f"{evidence}: choose a new diagnostic directory")
    with run_lock(run):
        state = read_run_json(run)
        generation = state.get("current_gen")
        if state.get("stage") != "export" or type(generation) is not int or generation < 1:
            raise ValueError("expected a stopped generation at stage=export")
        cfg = normalize_config(state.get("config"))
        candidate = run / "gens" / f"gen_{generation:04d}"
        summary = read_json(candidate / "train.json")
        if type(summary.get("steps")) is not int or summary["steps"] != cfg["train"]["steps"]:
            raise ValueError("training summary does not show all configured steps completed")
        files, incomplete = _chunks(run, generation)
        sp = cfg["selfplay"]
        expected_chunks = (sp["games"] + sp["games_per_chunk"] - 1) // sp["games_per_chunk"]
        if len(files) != 2 * expected_chunks or incomplete:
            raise ValueError("expected every configured self-play chunk to be complete")
        _protect_validation(run, files)
        required = [run / "run.json", candidate / "model.pt", candidate / "train.json"]
        optional = [candidate / name for name in ("train_history.jsonl", "generation.json",
                                                  "selfplay_summary.json", "model.onnx")]
        for path in required + [path for path in optional if path.exists()]:
            files[str(path.relative_to(run))] = _fingerprint(path)
        publication = candidate / "model.onnx"
        published_before = publication.exists()
        evidence.mkdir(parents=True, exist_ok=False)
        backup = evidence / "backup"
        backup.mkdir()
        for path in required + [path for path in optional if path.exists()]:
            destination = backup / path.name
            shutil.copy2(path, destination)
            if _fingerprint(destination)["sha256"] != files[str(path.relative_to(run))]["sha256"]:
                raise ValueError(f"backup verification failed: {path}")
        manifest_path = evidence / "preservation.json"
        write_json_atomic(manifest_path, {
            "format": "chessbot-export-preservation-v1", "run": str(run),
            "generation": generation, "complete_chunks": expected_chunks, "files": files,
        })
        print(f"Verified backup; preserved {expected_chunks} complete chunks.", flush=True)
        profiles = []
        try:
            for profile in PROFILE_NAMES:
                row = _run_profile(profile, run, evidence, timeout, failing_row)
                profiles.append(row)
                label = ("PASS" if row.get("gate_pass") is True else "PARITY FAIL")
                if row.get("completed") is not True:
                    label = "INCOMPLETE"
                print(f"  {profile}: {label}", flush=True)
        finally:
            errors = []
            for name, expected in files.items():
                try:
                    if _fingerprint(run / name) != expected:
                        errors.append(f"{name}: bytes, hash or modification time changed")
                except (OSError, ValueError) as error:
                    errors.append(str(error))
            if not published_before and publication.exists():
                errors.append("an official model.onnx appeared during diagnostics")
            baseline = next((row for row in profiles if row.get("profile") == "baseline"), {})
            report = {"format": "chessbot-export-diagnostics-v1", "run": str(run),
                      "generation": generation, "created_at": datetime.now(timezone.utc).isoformat(),
                      "checkpoint_sha256": files[str((candidate / "model.pt").relative_to(run))]["sha256"],
                      "preservation_manifest": str(manifest_path), "baseline_pass": baseline.get("gate_pass") is True,
                      "protected_unchanged": not errors, "preservation_errors": errors, "profiles": profiles}
            write_json_atomic(evidence / "report.json", report)
    print(f"Saved diagnostic report: {evidence / 'report.json'}", flush=True)
    print("Protected artifacts: " + ("UNCHANGED" if report["protected_unchanged"] else "PRESERVATION FAILED"), flush=True)
    print("Diagnostic models are unpublished; review the report before recovery.", flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--failing-row", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        report = diagnose(args.run, args.out, args.timeout, args.failing_row)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, TypeError, LoopLockError) as error:
        print(f"export diagnostics: {error}", file=sys.stderr)
        return 2
    if (not report["protected_unchanged"] or len(report["profiles"]) != len(PROFILE_NAMES)
            or any(row.get("completed") is not True for row in report["profiles"])):
        return 2
    return 0 if report["baseline_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
