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


def wdl_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on `[Loss, Draw, Win]` class labels (0, 1, 2)."""
    return F.cross_entropy(logits, labels)


def total_loss(
    policy_logits: torch.Tensor,
    wdl_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(total, policy, wdl)` for a batch dict as produced by `ReplayBuffer`."""
    p = policy_loss(policy_logits, batch["policy"], batch["legal_mask"])
    v = wdl_loss(wdl_logits, batch["wdl"])
    return p + v, p, v
