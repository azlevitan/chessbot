"""Save / load `ChessNet` checkpoints (`.pt`)."""

from __future__ import annotations

from pathlib import Path

import torch

from models.net import ChessNet

ARCH = "se-resnet-6x64"


def save_checkpoint(model: ChessNet, path: str | Path) -> None:
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"arch": ARCH, "model_state": state}, Path(path))


def load_checkpoint(path: str | Path) -> ChessNet:
    """Load a checkpoint onto the CPU in eval mode."""
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    if ckpt.get("arch") != ARCH:
        raise ValueError(f"{path}: unexpected arch {ckpt.get('arch')!r}, want {ARCH!r}")
    model = ChessNet()
    model.load_state_dict(ckpt["model_state"])
    return model.eval()
