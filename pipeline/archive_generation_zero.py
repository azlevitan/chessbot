"""Copy Generation 0 models/results and available logs into a verified local archive.

Run on the Mac that owns data_store/runs/main. No training or source modification.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from zoneinfo import ZoneInfo

CORE = ("model.pt", "model.onnx", "train.json", "decision.json",
        "anchors/vs_random.json", "anchors/vs_material.json")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def copy_checked(source: Path, target: Path) -> dict:
    before = source.stat()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_hash, target_hash = sha256(source), sha256(target)
    after = source.stat()
    if (source_hash != target_hash or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns):
        raise RuntimeError(f"copy verification failed or source changed: {source}")
    return {"bytes": after.st_size, "sha256": target_hash}


def archive(run: Path, archive_root: Path, log_directory: Path = Path("/tmp")) -> Path:
    run = run.resolve()
    gen = run / "gens/gen_0000"
    sources = {name: gen / name for name in CORE}
    sources["run.json"] = run / "run.json"
    for path in sources.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"required artifact missing or empty: {path}")
        if path.suffix == ".json":
            read_json(path)

    optional = {
        "logs/bootstrap.log": run.parent.parent / "bootstrap.log",
        "logs/export-retry.log": log_directory / "chessbot-export-retry.log",
        "logs/anchor-random.log": log_directory / "chessbot-anchor-random.log",
        "logs/anchor-material.log": log_directory / "chessbot-anchor-material.log",
        "supervised.complete.json": run / "supervised/.complete.json",
        "supervised_val.complete.json": run / "supervised_val/.complete.json",
        "owner-results-record.json": Path(__file__).resolve().parents[1]
        / "training/GENERATION_0_RESULTS.json",
    }
    archive_root = archive_root.expanduser().resolve()
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("America/Toronto")).strftime("%Y%m%dT%H%M%S%z")
    stage = Path(tempfile.mkdtemp(prefix=f".gen_0000-{stamp}-", suffix=".partial",
                                  dir=archive_root))
    files = {name: copy_checked(source, stage / name) for name, source in sources.items()}
    missing, errors = [], {}
    for name, source in optional.items():
        if not source.is_file():
            missing.append(name)
            continue
        try:
            files[name] = copy_checked(source, stage / name)
        except (OSError, RuntimeError) as error:
            errors[name] = str(error)
            (stage / name).unlink(missing_ok=True)

    rss = {}
    for label, name in (("training", "logs/bootstrap.log"),
                        ("random", "logs/anchor-random.log"),
                        ("material", "logs/anchor-material.log")):
        matches = []
        if name in files:
            matches = re.findall(r"^\s*(\d+)\s+maximum resident set size\s*$",
                                 (stage / name).read_text(errors="replace"), re.MULTILINE)
        rss[label] = max(map(int, matches)) if matches else None
    manifest = {
        "format": "chessbot-gen0-archive-v1",
        "created_at": datetime.now(ZoneInfo("America/Toronto")).isoformat(),
        "files": files, "missing_optional_files": missing, "optional_copy_errors": errors,
        "training": read_json(stage / "train.json"),
        "anchors": {name: read_json(stage / f"anchors/vs_{name}.json")
                    for name in ("random", "material")},
        "maximum_rss_bytes_from_available_logs": rss,
        "logging_note": "The original training tee log reported timeouts and may be incomplete. "
                        "Null RSS means it was not found in the copied log, not zero usage.",
    }
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"],
                                cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, check=True)
        manifest["source_commit"] = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["source_commit"] = None
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    final = stage.with_name(stage.name[1:-len(".partial")])
    stage.rename(final)
    print(f"Saved verified archive: {final}")
    for name in ("random", "material"):
        print(f"{name}: a_score={manifest['anchors'][name].get('a_score')}; "
              f"maximum RSS bytes={rss[name] if rss[name] is not None else 'not found in log'}")
    print(f"Core files verified: {len(sources)}; available optional files: {len(files) - len(sources)}")
    if missing or errors:
        print("Some optional logs/metadata were unavailable; see manifest.json for details.")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("data_store/runs/main"))
    parser.add_argument("--archive-root", type=Path, default=Path.home()
                        / "Library/Application Support/chessbot/archives")
    args = parser.parse_args()
    archive(args.run, args.archive_root)


if __name__ == "__main__":
    main()
