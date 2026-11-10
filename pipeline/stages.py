"""Idempotent RL stages; the future loop owns persistence and stage advancement.

Pass a validated config, optionally overlaid with a runtime-only ``engine`` path.
Only resignation/promotion mutate state. Training and inference run in separate,
sequential subprocesses; this module never imports training dependencies.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

from evaluate import _cleanup, _read_block
from prune import prune as prune_run
from run_layout import ENGINE_BIN, REPO_ROOT
from stage_support import (
    context, decision_ready, flags, generation_threshold, incumbent, invalidate,
    remove_outputs, selfplay_info, temporary, training_process, training_ready,
    update_threshold, valid_onnx,
)
from window import select_window, train_args


def _inference(cfg):
    return {key: cfg["match"].get(key, cfg["selfplay"][key]) for key in ("ep", "threads")}


def selfplay(run_dir, gen, cfg, state) -> None:
    run, candidate = context(run_dir, gen)
    model = incumbent(run, gen, state) / "model.onnx"
    sp = cfg["selfplay"]
    directory = candidate / "selfplay"
    count = (sp["games"] + sp["games_per_chunk"] - 1) // sp["games_per_chunk"]
    chunks = []
    for index in range(count):
        games = min(sp["games_per_chunk"], sp["games"] - index * sp["games_per_chunk"])
        seed = 1_000_000 * gen + 1000 * index + cfg["seed"]
        if seed + games - 1 > 2**64 - 1:
            raise ValueError("selfplay game seed overflows u64")
        binary = directory / f"chunk_{index:06d}.bin"
        expected = {"format": "chessbot-chunk-v1", "kind": "selfplay", "games": games,
                    "seed": seed, "model": str(model),
                    **{key: sp[key] for key in ("sims", "max_plies", "temp_plies")}}
        chunks.append((binary, expected, selfplay_info(binary, expected)))
    directory.mkdir(parents=True, exist_ok=True)
    threshold = generation_threshold(directory, model,
                                     [info for _, _, info in chunks if info is not None], state)
    completed = []
    for binary, expected, info in chunks:
        sidecar = binary.with_suffix(".json")
        if info is None:
            remove_outputs(binary, sidecar)
            try:
                subprocess.run([str(cfg.get("engine", ENGINE_BIN)), "selfplay", "--model", str(model),
                                "--games", str(expected["games"]), "--seed", str(expected["seed"]),
                                "--out", str(binary), "--resign-threshold", str(threshold),
                                *flags(sp, ("sims", "parallel", "leaf_batch", "max_plies", "temp_plies",
                                            "resign_consecutive", "resign_playthrough", "ep", "threads"))],
                               check=True)
                info = selfplay_info(binary, expected)
                if info is None:
                    raise ValueError(f"{binary}: selfplay did not publish a complete chunk")
            except BaseException:
                remove_outputs(binary, sidecar)
                raise
        temporary(binary).unlink(missing_ok=True)
        temporary(sidecar).unlink(missing_ok=True)
        completed.append(info)
    update_threshold(completed, threshold, state)


def train(run_dir, gen, cfg, state) -> None:
    run, candidate = context(run_dir, gen)
    if training_ready(candidate):
        return
    initial = incumbent(run, gen, state) / "model.pt"
    if not initial.is_file():
        raise FileNotFoundError(f"incumbent checkpoint not found: {initial}")
    sources = train_args(select_window(run, gen, cfg))
    checkpoint, summary = candidate / "model.pt", candidate / "train.json"
    invalidate(candidate, model=True)
    remove_outputs(checkpoint, summary)
    try:
        training_process("train.py", *sources, "--init", initial, "--out", checkpoint,
                         "--log-json", summary, "--val", run / "supervised_val",
                         "--seed", cfg["seed"], "--q-ratio", cfg["train"].get("q_ratio", 0.3),
                         *flags(cfg["train"], ("steps", "batch_size", "lr", "weight_decay",
                                              "warmup_steps", "steps_per_epoch", "device", "val_samples")))
        if not training_ready(candidate):
            raise ValueError(f"{candidate}: training did not publish checkpoint and summary")
    except BaseException:
        remove_outputs(checkpoint, summary)
        raise


def export(run_dir, gen, cfg, state) -> None:
    run, candidate = context(run_dir, gen)
    onnx = candidate / "model.onnx"
    pending = temporary(onnx)
    if not pending.exists() and valid_onnx(onnx):
        return
    if not training_ready(candidate):
        raise ValueError(f"{candidate}: export requires complete training outputs")
    invalidate(candidate)
    pending.unlink(missing_ok=True)
    try:
        training_process("export_onnx.py", "--checkpoint", candidate / "model.pt",
                         "--out", pending, "--parity-source", run / "supervised_val")
        pending.replace(onnx)
    finally:
        pending.unlink(missing_ok=True)


def match(run_dir, gen, cfg, state) -> None:
    run, candidate = context(run_dir, gen)
    best = incumbent(run, gen, state)
    # evaluate validates cached blocks before any engine work and preserves mtimes.
    subprocess.run([sys.executable, str(REPO_ROOT / "pipeline" / "evaluate.py"),
                    "--engine", str(cfg.get("engine", ENGINE_BIN)),
                    "--a", str(candidate / "model.onnx"), "--b", str(best / "model.onnx"),
                    "--out-dir", str(candidate / "match"), "--decision", str(candidate / "decision.json"),
                    "--seed-base", str(1_000_000 * gen),
                    *flags(cfg["match"], ("sims", "block_games", "max_games", "batch", "max_plies",
                                          "elo0", "elo1", "alpha", "beta")),
                    *flags(_inference(cfg), ("ep", "threads"))], check=True)


def decide(run_dir, gen, cfg, state) -> None:
    run, candidate = context(run_dir, gen)
    path = candidate / "decision.json"
    temporary(path).unlink(missing_ok=True)
    decision = decision_ready(path)
    if decision is None:
        match(run_dir, gen, cfg, state)
        decision = decision_ready(path)
        if decision is None:
            raise ValueError(f"{path}: evaluation did not publish a decision")
    if decision["promoted"]:
        state["best_gen"] = gen
    else:
        state["best_gen"] = int(incumbent(run, gen, state).name.removeprefix("gen_"))


def anchors(run_dir, gen, cfg, state) -> None:
    _, candidate = context(run_dir, gen)
    decision = decision_ready(candidate / "decision.json")
    if decision is None:
        raise ValueError(f"{candidate}: anchors require a promotion decision")
    if not decision["promoted"]:
        return
    directory = candidate / "anchors"
    directory.mkdir(parents=True, exist_ok=True)
    games = cfg["anchors"]["games"]
    for opponent in ("random", "material"):
        path = directory / f"vs_{opponent}.json"
        pending = temporary(path)
        _cleanup(pending)
        args = SimpleNamespace(a=str(candidate / "model.onnx"), b=opponent,
                               sims=cfg["match"]["sims"], seed_base=1_000_000 * gen,
                               max_plies=cfg["match"]["max_plies"])
        if path.exists():
            _read_block(path, args, 0, games)
            continue
        try:
            subprocess.run([str(cfg.get("engine", ENGINE_BIN)), "match", "--a", args.a,
                            "--b", opponent, "--games", str(games), "--seed", str(args.seed_base),
                            "--opening-offset", "0", "--out", str(pending),
                            *flags(cfg["match"], ("sims", "batch", "max_plies")),
                            *flags(_inference(cfg), ("ep", "threads"))], check=True)
            _read_block(pending, args, 0, games)
            pending.replace(path)
        finally:
            _cleanup(pending)


def prune(run_dir, gen, cfg, state) -> None:
    """Remove eligible historical artifacts before the loop checkpoints success."""
    if state["current_gen"] != gen:
        raise ValueError("prune generation must match current_gen")
    prune_run(run_dir, state, cfg)
