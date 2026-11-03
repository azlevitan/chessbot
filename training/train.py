"""Task 6.4 CLI: stream Rust chunks into the SE-ResNet with bounded memory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Subset

from checkpoint import load_checkpoint, save_checkpoint
from data.chunk_dataset import ChunkDataset, list_complete_chunks, read_sidecar
from data.replay_buffer import make_loader
from data.sampler import WeightedSourceSampler
from device import default_device
from models.net import ChessNet
from train_loop import train_steps


def source_paths(text: str) -> list[Path]:
    paths = []
    for item in text.split(","):
        if not item:
            raise ValueError("source paths must not be empty")
        path = Path(item)
        paths.extend(list_complete_chunks(path) if path.is_dir() else [path])
    return paths


def build_sources(specs: list[str]) -> tuple[ChunkDataset, list[tuple[int, int]], list[dict]]:
    paths, ranges, sources = [], [], []
    offset = 0
    for spec in specs:
        text, separator, suffix = spec.rpartition(":")
        try:
            weight = float(suffix) if separator else 1.0
        except ValueError:
            text, weight = spec, 1.0
        if not separator:
            text = spec
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("source weights must be finite and nonnegative")
        chunks = source_paths(text)
        count = sum(read_sidecar(path)["samples"] for path in chunks)
        ranges.append((offset, offset + count))
        sources.append({"paths": [str(path) for path in chunks], "weight": weight, "samples": count})
        paths.extend(chunks)
        offset += count
    return ChunkDataset(paths), ranges, sources


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--q-ratio", type=float, default=0.0)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--val", type=Path)
    parser.add_argument("--val-samples", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-json", type=Path)
    args = parser.parse_args(argv)
    if min(args.steps, args.batch_size, args.steps_per_epoch, args.val_samples) < 1:
        parser.error("steps, batch-size, steps-per-epoch and val-samples must be positive")
    if min(args.warmup_steps, args.seed) < 0:
        parser.error("warmup-steps and seed must be nonnegative")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight-decay must be finite and nonnegative")
    if not 0 <= args.q_ratio <= 1:
        parser.error("q-ratio must be in [0, 1]")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("mps is unavailable on this machine; use --device cpu or auto")
    args.log_json = args.log_json if args.log_json is not None else args.out.with_name("train.json")
    if args.log_json.resolve() == args.out.resolve():
        parser.error("out and log-json must be different paths")
    return args


def main(argv=None) -> dict:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    dataset, ranges, sources = build_sources(args.source)
    sampler = WeightedSourceSampler(ranges, [s["weight"] for s in sources],
                                    args.steps * args.batch_size, args.seed)
    loader = make_loader(dataset, args.batch_size, sampler=sampler)
    val_loader = None
    if args.val is not None:
        validation = ChunkDataset(source_paths(str(args.val)))
        if not len(validation):
            raise ValueError("validation requires at least one complete, nonempty chunk")
        subset = Subset(validation, range(min(args.val_samples, len(validation))))
        val_loader = make_loader(subset, args.batch_size, shuffle=False)
    model = load_checkpoint(args.init) if args.init is not None else ChessNet()
    device = default_device() if args.device == "auto" else torch.device(args.device)
    result = train_steps(model, loader, steps=args.steps, device=device, lr=args.lr,
                         weight_decay=args.weight_decay, warmup_steps=args.warmup_steps,
                         steps_per_epoch=args.steps_per_epoch, q_ratio=args.q_ratio,
                         val_loader=val_loader)
    result["sources"] = sources
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    save_checkpoint(model, temporary)
    temporary.replace(args.out)
    log_path = args.log_json
    log_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = log_path.with_name(log_path.name + ".tmp")
    temporary.write_text(json.dumps(result, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(log_path)
    return result


if __name__ == "__main__":
    main()
