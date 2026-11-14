"""Change only a stopped run's inference batch, preserving resume evidence."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

from config import normalize_config
from loop_lock import LoopLockError, run_lock
from run_layout import REPO_ROOT, read_json, read_run_json, write_json_atomic, write_run_json

FORMAT = "chessbot-inference-change-v1"


def _fingerprint(path: Path) -> dict:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{path}: expected a regular file, not a symbolic link")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"{path}: changed while recording evidence")
    return {"bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": digest}


def _chunks(run: Path, generation: int) -> tuple[dict, list[str]]:
    directory = run / "gens" / f"gen_{generation:04d}" / "selfplay"
    if directory.is_symlink():
        raise ValueError(f"{directory}: refusing a symbolic-link sample directory")
    names = sorted({path.stem for path in directory.glob("chunk_*.*")
                    if re.fullmatch(r"chunk_[0-9]{6}\.(?:bin|json)", path.name)})
    files, skipped = {}, []
    for name in names:
        binary, sidecar = directory / f"{name}.bin", directory / f"{name}.json"
        if not binary.exists() or not sidecar.exists():
            skipped.append(name)
            continue
        metadata = read_json(sidecar)
        samples = metadata.get("samples")
        if (metadata.get("format") != "chessbot-chunk-v1"
                or metadata.get("kind") != "selfplay"
                or type(samples) is not int or samples < 0):
            raise ValueError(f"{sidecar}: invalid published selfplay metadata")
        if binary.stat().st_size != samples * 492:
            raise ValueError(f"{binary}: published chunk size does not match samples * 492")
        for path in (binary, sidecar):
            files[str(path.relative_to(run))] = _fingerprint(path)
    return files, skipped


def _default_out() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    root = Path.home() / "Library/Logs/chessbot" if sys.platform == "darwin" else REPO_ROOT / "work"
    return root / f"inference-change-{stamp}"


def update_run(run_dir, batch: int, out_dir=None) -> Path:
    """Back up exact state and complete chunk fingerprints before atomic update."""
    if type(batch) is not int or not 1 <= batch <= 64:
        raise ValueError("batch must be an integer in 1..64")
    run = Path(run_dir).expanduser().resolve(strict=True)
    out = (Path(out_dir).expanduser() if out_dir is not None else _default_out()).resolve()
    if out == run or run in out.parents:
        raise ValueError("backup output must be outside the run directory")
    if out.exists():
        raise FileExistsError(f"{out}: backup output already exists; choose a new directory")
    with run_lock(run):
        original = (run / "run.json").read_bytes()
        state = read_run_json(run)
        generation = state.get("current_gen")
        if type(generation) is not int or generation < 1:
            raise ValueError("current_gen must be a positive integer")
        cfg = deepcopy(state.get("config"))
        normalize_config(cfg)
        for section in ("selfplay", "match"):
            cfg.setdefault(section, {})["infer_batch"] = batch
        normalize_config(cfg)  # Validate without persisting merged defaults.
        files, skipped = _chunks(run, generation)
        if original != (run / "run.json").read_bytes():
            raise ValueError("run.json changed while recording evidence")
        out.mkdir(parents=True, exist_ok=False)
        with (out / "run.before.json").open("xb") as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
        write_json_atomic(out / "manifest.json", {
            "format": FORMAT, "run": str(run), "generation": generation,
            "saved_at": datetime.now(timezone.utc).isoformat(), "infer_batch": batch,
            "original_run_sha256": hashlib.sha256(original).hexdigest(),
            "complete_chunks": len(files) // 2, "files": files,
            "skipped_incomplete_chunks": skipped,
        })
        proposed = deepcopy(state)
        proposed["config"] = cfg
        write_run_json(run, proposed)
    print(f"selfplay.infer_batch = {batch}; match.infer_batch = {batch}")
    print(f"Saved backup and resume evidence: {out}")
    print(f"Complete selfplay chunks recorded: {len(files) // 2}; incomplete chunks skipped: {len(skipped)}")
    return out


def verify_saved(run_dir, saved_dir) -> int:
    """Compare saved file bytes and mtimes; no run artifacts are changed."""
    run = Path(run_dir).expanduser().resolve(strict=True)
    manifest = read_json(Path(saved_dir).expanduser() / "manifest.json")
    if manifest.get("format") != FORMAT or manifest.get("run") != str(run):
        raise ValueError("saved evidence does not describe this run directory")
    files = manifest.get("files")
    count = manifest.get("complete_chunks")
    if (not isinstance(files, dict) or type(count) is not int or count < 0
            or len(files) != count * 2):
        raise ValueError("invalid saved chunk manifest")
    with run_lock(run):
        for name, expected in files.items():
            relative = Path(name)
            path = run / relative
            if relative.is_absolute() or ".." in relative.parts or run not in path.resolve().parents:
                raise ValueError(f"{name}: saved file must remain inside the run")
            actual = _fingerprint(path)
            if not isinstance(expected, dict):
                raise ValueError(f"{name}: invalid saved fingerprint")
            for field in ("bytes", "mtime_ns", "sha256"):
                if actual[field] != expected.get(field):
                    raise ValueError(f"{path}: {field} differs from saved evidence")
    print(f"Verified unchanged complete selfplay chunks: {count}")
    print("This verifies saved chunks only; it does not establish all M2 acceptance criteria.")
    return count


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--batch", type=int, choices=range(1, 65), metavar="1..64")
    mode.add_argument("--verify-saved", type=Path, metavar="BACKUP_DIR")
    parser.add_argument("--out", type=Path, help="new backup directory, outside the run")
    args = parser.parse_args(argv)
    if args.verify_saved is not None and args.out is not None:
        parser.error("--out is only valid with --batch")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.verify_saved is not None:
            verify_saved(args.run, args.verify_saved)
        else:
            update_run(args.run, args.batch, args.out)
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, TypeError, LoopLockError) as error:
        print(f"inference batch: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
