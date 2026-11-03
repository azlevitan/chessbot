"""Training losses: masked policy cross-entropy + WDL cross-entropy (SPEC.md §1, §4)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

ILLEGAL_LOGIT = -1e9


def masked_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Set illegal-move logits to -1e9 before softmax (SPEC.md §1)."""
    return logits.masked_fill(~legal_mask, ILLEGAL_LOGIT)


def policy_loss(
    logits: torch.Tensor, target: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    """Soft-target cross-entropy over legal moves, averaged over the batch.

    Illegal moves carry target 0, so their -1e9 log-probs contribute nothing.
    """
    log_probs = F.log_softmax(masked_logits(logits, legal_mask), dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


def wdl_loss(
    logits: torch.Tensor, labels: torch.Tensor,
    q_value: torch.Tensor | None = None, q_ratio: float = 0.0,
) -> torch.Tensor:
    """Blend outcome labels with search Q in `[Loss, Draw, Win]` order."""
    if not 0.0 <= q_ratio <= 1.0:
        raise ValueError("q_ratio must be in [0, 1]")
    if q_ratio == 0.0 or q_value is None:
        return F.cross_entropy(logits, labels)
    q = q_value.to(device=logits.device, dtype=logits.dtype).clamp(-1.0, 1.0)
    soft_q = torch.stack(((-q).clamp(0, 1), 1 - q.abs(), q.clamp(0, 1)), dim=-1)
    target = (1 - q_ratio) * F.one_hot(labels, 3).to(logits.dtype) + q_ratio * soft_q
    return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def total_loss(
    policy_logits: torch.Tensor,
    wdl_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    q_ratio: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(total, policy, wdl)` for a batch dict as produced by `ReplayBuffer`."""
    p = policy_loss(policy_logits, batch["policy"], batch["legal_mask"])
    v = wdl_loss(wdl_logits, batch["wdl"], batch.get("q_value", batch.get("q")), q_ratio)
    return p + v, p, v
