"""Bootstrap a supervised generation 0, then leave the run ready for self-play.

Usage: python3 pipeline/bootstrap.py --pgn FILE.pgn[.zst] --run DIR --config CFG.json
The release engine must already be built. Training and export use this Python's
environment, in sequential subprocesses, to keep their memory out of this driver.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from run_layout import (
    ENGINE_BIN, REPO_ROOT, chunk_manifest, generation_dir, read_json,
    read_run_json, write_json_atomic, write_run_json,
)


def training_process(script: str, *arguments) -> None:
    subprocess.run([sys.executable, str(REPO_ROOT / "training" / script),
                    *map(str, arguments)], check=True)


def convert_pgn(pgn: Path, directory: Path, *limits: str) -> None:
    command = [str(ENGINE_BIN), "convert-pgn", "--out-dir", str(directory), *limits]
    if pgn.suffix != ".zst":
        subprocess.run([*command, "--input", str(pgn)], check=True)
        return
    with subprocess.Popen(["zstd", "-dc", str(pgn)], stdout=subprocess.PIPE) as decoder:
        try:
            # Retain this read end until conversion finishes. An early game/sample
            # limit then blocks the decoder instead of racing a broken-pipe error.
            subprocess.run([*command, "--input", "-"], stdin=decoder.stdout, check=True)
            status = decoder.poll()
            if status is None:
                decoder.terminate()
                status = decoder.wait()
                if status == -signal.SIGTERM:
                    status = 0  # Intentional early stop after successful conversion.
            if status:
                raise subprocess.CalledProcessError(status, decoder.args)
        finally:
            if decoder.poll() is None:
                decoder.terminate()
            decoder.wait()


def convert_stage(pgn: Path, directory: Path, *limits: str, invalidate=()) -> bool:
    """Return whether data changed. Publish a whole conversion, never a prefix."""
    temporary = directory.with_name(directory.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    for stale in directory.glob("*.tmp"):
        stale.unlink()
    identity = {"pgn": str(pgn), "size": pgn.stat().st_size,
                "mtime_ns": pgn.stat().st_mtime_ns, "limits": list(limits)}
    try:
        expected = {"input": identity, "chunks": chunk_manifest(directory)}
        if read_json(directory / ".complete.json") == expected:
            return False
    except (OSError, ValueError):
        pass
    # Invalidate dependents before publishing data: a crash immediately after
    # rename must never make an older checkpoint look current on the next run.
    remove_outputs(*invalidate)
    temporary.mkdir(parents=True)
    convert_pgn(pgn, temporary, *limits)
    write_json_atomic(temporary / ".complete.json",
                      {"input": identity, "chunks": chunk_manifest(temporary)})
    if directory.exists():
        shutil.rmtree(directory)
    temporary.replace(directory)
    return True


def valid_onnx(path: Path) -> bool:
    if not path.is_file():
        return False
    result = subprocess.run(
        [sys.executable, "-c", "import sys; from export_onnx import check_contract; "
         "check_contract(sys.argv[1])", str(path)], cwd=REPO_ROOT / "training",
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def remove_outputs(*paths: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
        path.with_name(path.name + ".tmp").unlink(missing_ok=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("pgn", "run", "config"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--val-games", type=int, default=2000)
    parser.add_argument("--max-samples", type=int, default=4194304)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args(argv)
    if min(args.val_games, args.max_samples, args.steps, args.batch_size) < 1:
        parser.error("val-games, max-samples, steps and batch-size must be positive")
    args.pgn, args.run, args.config = (path.resolve() for path in
                                       (args.pgn, args.run, args.config))
    if not args.pgn.is_file():
        parser.error(f"PGN not found: {args.pgn}")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    config = read_json(args.config)
    train_config = config.get("train", {})
    if not isinstance(train_config, dict):
        raise ValueError("config.train must be a JSON object")
    state_path = args.run / "run.json"
    state = read_run_json(args.run) if state_path.exists() else None
    if state is not None and state["config"] != config:
        raise ValueError("config differs from the existing run; use a new run directory")
    if state is not None and (state["current_gen"] != 1 or state["best_gen"] != 0
                              or state["stage"] != "selfplay"):
        print(f"Run already advanced beyond bootstrap: {args.run}")
        return
    if not ENGINE_BIN.is_file():
        raise FileNotFoundError(f"build the release engine first: {ENGINE_BIN}")
    validation, supervised = args.run / "supervised_val", args.run / "supervised"
    gen = generation_dir(args.run, 0)
    checkpoint, summary, onnx, decision = (gen / name for name in
                                          ("model.pt", "train.json", "model.onnx", "decision.json"))
    dependents = (checkpoint, summary, onnx, decision)
    changed = convert_stage(args.pgn, validation, "--max-games", str(args.val_games),
                            invalidate=dependents)
    changed |= convert_stage(args.pgn, supervised, "--skip-games", str(args.val_games),
                             "--max-samples", str(args.max_samples), invalidate=dependents)
    train_ready = checkpoint.is_file() and checkpoint.stat().st_size > 0
    try:
        read_json(summary)
    except (OSError, ValueError):
        train_ready = False
    train_ready &= not any(path.with_name(path.name + ".tmp").exists()
                           for path in (checkpoint, summary))
    if changed or not train_ready:
        remove_outputs(checkpoint, summary, onnx, decision)
        options = []
        for key in ("lr", "weight_decay", "warmup_steps", "steps_per_epoch", "val_samples"):
            if key in train_config:
                options.extend(("--" + key.replace("_", "-"), train_config[key]))
        training_process("train.py", "--source", supervised, "--val", validation,
                         "--q-ratio", "0.0", "--out", checkpoint, "--log-json", summary,
                         "--steps", args.steps, "--batch-size", args.batch_size,
                         "--device", args.device, "--seed", config.get("seed", 0), *options)
    temporary = onnx.with_name(onnx.name + ".tmp")
    if temporary.exists() or not valid_onnx(onnx):
        remove_outputs(onnx, decision)
        training_process("export_onnx.py", "--checkpoint", checkpoint, "--out", temporary)
        temporary.replace(onnx)
    bootstrap_decision = {"promoted": True, "reason": "bootstrap"}
    try:
        decision_ready = read_json(decision) == bootstrap_decision
    except (OSError, ValueError):
        decision_ready = False
    if not decision_ready:
        write_json_atomic(decision, bootstrap_decision)
    decision.with_name("decision.json.tmp").unlink(missing_ok=True)
    if state is None:
        write_run_json(args.run, {"config": config, "best_gen": 0, "current_gen": 1,
                                 "stage": "selfplay", "resign_threshold": -0.95})
    state_path.with_name("run.json.tmp").unlink(missing_ok=True)
    print(f"Generation 0 ready: {gen}")


if __name__ == "__main__":
    main()
