"""GATE 3: SE-ResNet-6x64 memorizes one synthetic batch, and ONNX export matches PyTorch.

The batch is random tensors (no chess encoding, CLAUDE.md Law 1) with one-hot
policy targets: soft targets could never reach loss < 0.05, since cross-entropy
is bounded below by the target's entropy.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from checkpoint import load_checkpoint, save_checkpoint
from device import default_device, empty_cache
from export_onnx import PARITY_ATOL, check_contract, export_onnx, verify_parity
from losses import ILLEGAL_LOGIT, masked_logits, policy_loss, total_loss
from models.net import NUM_BLOCKS, NUM_MOVES, ChessNet, ResBlock, count_parameters

BATCH = 16
STEPS = 80
LOSS_TARGET = 0.05
EXPECTED_PARAMS = 9_043_067  # derived by hand from the SPEC.md §4 layer sizes


def synthetic_batch(n: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """Random binary planes, random legal masks, one-hot policy, random WDL."""
    g = torch.Generator().manual_seed(seed)
    planes = (torch.rand(n, 18, 8, 8, generator=g) < 0.15).float()
    planes[:, 17] = torch.rand(n, 1, 1, generator=g)  # halfmove-style scalar plane
    mask = torch.zeros(n, NUM_MOVES, dtype=torch.bool)
    policy = torch.zeros(n, NUM_MOVES)
    for i in range(n):
        k = int(torch.randint(20, 41, (1,), generator=g))
        legal = torch.randperm(NUM_MOVES, generator=g)[:k]
        mask[i, legal] = True
        policy[i, legal[int(torch.randint(k, (1,), generator=g))]] = 1.0
    wdl = torch.randint(0, 3, (n,), generator=g)
    return {"planes": planes, "policy": policy, "legal_mask": mask, "wdl": wdl}


def test_architecture() -> None:
    model = ChessNet().eval()
    for b in (1, 5):
        policy, wdl = model(torch.zeros(b, 18, 8, 8))
        assert policy.shape == (b, NUM_MOVES)
        assert wdl.shape == (b, 3)
    blocks = [m for m in model.modules() if isinstance(m, ResBlock)]
    assert len(blocks) == NUM_BLOCKS == 6
    assert all(b.se.fc1.out_features == 16 for b in blocks)  # 64 / ratio 4
    stem_conv = model.stem[0][0]
    assert isinstance(stem_conv, nn.Conv2d) and stem_conv.bias is None
    assert (stem_conv.in_channels, stem_conv.out_channels) == (18, 64)
    assert count_parameters(model) == EXPECTED_PARAMS


def test_masked_policy_loss() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, NUM_MOVES)
    mask = torch.zeros(2, NUM_MOVES, dtype=torch.bool)
    mask[:, :10] = True
    target = torch.zeros(2, NUM_MOVES)
    target[:, 3] = 1.0
    base = policy_loss(logits, target, mask)
    assert torch.isfinite(base)
    # Changing illegal logits (even to huge values) must not change the loss.
    noisy = logits.clone()
    noisy[:, 10:] = 1e6
    assert torch.allclose(policy_loss(noisy, target, mask), base)
    assert torch.all(masked_logits(logits, mask)[:, 10:] == ILLEGAL_LOGIT)
    # Strongly favouring the one-hot target drives the loss to ~0.
    peaked = torch.zeros(2, NUM_MOVES)
    peaked[:, 3] = 50.0
    assert policy_loss(peaked, target, mask) < 1e-6


def test_overfit_then_onnx_parity(tmp_path: Path) -> None:
    torch.manual_seed(0)
    device = default_device()
    cpu_batch = synthetic_batch(BATCH)
    batch = {k: v.to(device) for k, v in cpu_batch.items()}
    model = ChessNet().to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    losses = []
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss, _, _ = total_loss(*model(batch["planes"]), batch)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    print(f"loss step0={losses[0]:.3f} step40={losses[40]:.4f} step79={losses[-1]:.5f}")
    # Sanity: an untrained net starts near ln(~30) + ln(3), so the drop is real.
    assert losses[0] > 3.0, f"initial loss {losses[0]:.3f} suspiciously low"

    with torch.no_grad():
        final, p_loss, v_loss = total_loss(*model(batch["planes"]), batch)
    final_loss = float(final)
    print(f"device={device} final_loss={final_loss:.4f} policy={float(p_loss):.4f} "
          f"wdl={float(v_loss):.4f}")
    assert final_loss < LOSS_TARGET, f"loss {final_loss:.4f} >= {LOSS_TARGET} after {STEPS} steps"

    ckpt = tmp_path / "overfit.pt"
    save_checkpoint(model, ckpt)
    empty_cache(device)
    restored = load_checkpoint(ckpt)
    onnx_path = export_onnx(restored, tmp_path / "overfit.onnx")
    check_contract(onnx_path)
    for x in (cpu_batch["planes"], cpu_batch["planes"][:1]):
        diff = verify_parity(restored, onnx_path, x)
        print(f"parity batch={x.shape[0]} max_abs_diff={diff:.2e}")
        assert diff <= PARITY_ATOL, f"ONNX parity {diff:.2e} > {PARITY_ATOL}"
