"""Validated, independent RL configuration defaults from SPEC §10 (stdlib only)."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

from run_layout import read_json


DEFAULTS = {
    "seed": 0, "disk_budget_gb": 20,
    "selfplay": {
        "games": 1200, "games_per_chunk": 50, "sims": 200, "parallel": 32,
        "leaf_batch": 1, "max_plies": 512, "temp_plies": 30,
        "resign_consecutive": 3, "resign_playthrough": 0.1,
        "ep": "auto", "threads": 1,
    },
    "train": {
        "steps": 4000, "batch_size": 512, "lr": 0.001, "weight_decay": 0.0001,
        "warmup_steps": 100, "steps_per_epoch": 500, "device": "auto",
        "q_ratio": 0.3,
    },
    "window": {
        "generations": 4, "max_samples": 1000000,
        "supervised_weight": [0.5, 0.3, 0.2, 0.1, 0.0],
    },
    "match": {
        "block_games": 40, "max_games": 400, "sims": 200, "batch": 8,
        "max_plies": 300, "elo0": 0, "elo1": 40, "alpha": 0.05, "beta": 0.05,
    },
    "anchors": {"games": 20},
}


def _merge(defaults: dict, overrides: dict, prefix: str = "") -> dict:
    result = deepcopy(defaults)
    for key, value in overrides.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(defaults.get(key), dict):
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a JSON object")
            result[key] = _merge(defaults[key], value, name)
        else:
            # Lists represent complete schedules, not element-wise patches.
            result[key] = deepcopy(value)
    return result


def _integer(value, name: str, minimum: int = 1, maximum: int | None = None) -> None:
    if (type(value) is not int or value < minimum
            or (maximum is not None and value > maximum)):
        limit = f"..{maximum}" if maximum is not None else " or greater"
        raise ValueError(f"{name} must be an integer {minimum}{limit}")


def _number(value, name: str, minimum=None, maximum=None) -> None:
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")


def _choice(value, name: str, choices: tuple[str, ...]) -> None:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}")


def _validate(cfg: dict) -> None:
    _integer(cfg["seed"], "seed", 0, 2**64 - 1)
    _number(cfg["disk_budget_gb"], "disk_budget_gb", 0)
    if cfg["disk_budget_gb"] == 0:
        raise ValueError("disk_budget_gb must be positive")
    sp, train, window, match = (cfg[key] for key in ("selfplay", "train", "window", "match"))
    for section, fields in (
        ("selfplay", ("games", "parallel", "leaf_batch", "threads")),
        ("train", ("steps", "batch_size", "steps_per_epoch")),
        ("window", ("generations", "max_samples")),
        ("match", ("batch",)),
    ):
        for field in fields:
            _integer(cfg[section][field], f"{section}.{field}")
    _integer(sp["games_per_chunk"], "selfplay.games_per_chunk", 1, 1000)
    for section in ("selfplay", "match"):
        _integer(cfg[section]["sims"], f"{section}.sims", 2, 2**32 - 1)
        _integer(cfg[section]["max_plies"], f"{section}.max_plies", 1, 2**32 - 1)
    _integer(sp["temp_plies"], "selfplay.temp_plies", 0, 2**32 - 1)
    _integer(sp["resign_consecutive"], "selfplay.resign_consecutive", 1, 2**32 - 1)
    _integer(train["warmup_steps"], "train.warmup_steps", 0)
    if "val_samples" in train:
        _integer(train["val_samples"], "train.val_samples")
    for section, field in (("match", "block_games"), ("match", "max_games"), ("anchors", "games")):
        value, name = cfg[section][field], f"{section}.{field}"
        _integer(value, name, 2)
        if value % 2:
            raise ValueError(f"{name} must be even")
    # Only an individual match block (or anchor match) reaches the Rust u32 CLI.
    _integer(min(match["block_games"], match["max_games"]), "match block size", 2, 2**32 - 1)
    _integer(cfg["anchors"]["games"], "anchors.games", 2, 2**32 - 1)
    _number(train["lr"], "train.lr", 0)
    if train["lr"] == 0:
        raise ValueError("train.lr must be positive")
    _number(train["weight_decay"], "train.weight_decay", 0)
    _number(train["q_ratio"], "train.q_ratio", 0, 1)
    _number(sp["resign_playthrough"], "selfplay.resign_playthrough", 0, 1)
    _choice(sp["ep"], "selfplay.ep", ("auto", "cpu", "coreml"))
    _choice(train["device"], "train.device", ("auto", "cpu", "mps"))
    schedule = window["supervised_weight"]
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("window.supervised_weight must be a nonempty list")
    for index, weight in enumerate(schedule):
        _number(weight, f"window.supervised_weight[{index}]", 0, 1)
    for field in ("elo0", "elo1", "alpha", "beta"):
        _number(match[field], f"match.{field}")
    if match["elo0"] >= match["elo1"]:
        raise ValueError("match.elo0 must be less than match.elo1")
    for field in ("alpha", "beta"):
        if not 0 < match[field] < 1:
            raise ValueError(f"match.{field} must be strictly between 0 and 1")


def load_config(path: str | Path) -> dict:
    """Deep-merge a JSON object over fresh defaults and validate known fields.

    Additional keys are preserved for callers (such as bootstrap's optional
    train.val_samples). Loading is platform-independent: the CPU CI host may
    inspect an M1/CoreML config without attempting to initialize that provider.
    """
    cfg = _merge(DEFAULTS, read_json(path))
    _validate(cfg)
    return cfg
