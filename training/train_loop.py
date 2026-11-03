"""Streaming optimizer updates and bounded, hard-label validation (Task 6.4)."""

from __future__ import annotations

import json
import time

import torch

from device import empty_cache
from losses import masked_logits, total_loss
from schedule import lr_multiplier


def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    """Evaluate the supplied validation subset without changing training state."""
    was_training = model.training
    model.eval()
    samples, correct, policy_sum, wdl_sum = 0, 0, 0.0, 0.0
    try:
        with torch.no_grad():
            for cpu_batch in loader:
                batch = {key: value.to(device) for key, value in cpu_batch.items() if key != "q"}
                policy, wdl = model(batch["planes"])
                _, p_loss, v_loss = total_loss(policy, wdl, batch)
                n = len(batch["wdl"])
                predicted = masked_logits(policy, batch["legal_mask"]).argmax(dim=-1)
                correct += int((predicted == batch["policy"].argmax(dim=-1)).sum())
                samples += n
                policy_sum += float(p_loss) * n
                wdl_sum += float(v_loss) * n
    finally:
        model.train(was_training)
    if not samples:
        raise ValueError("validation requires at least one sample")
    return {"policy_top1": correct / samples, "policy_loss": policy_sum / samples,
            "wdl_loss": wdl_sum / samples}


def train_steps(
    model, loader, *, steps: int, device: torch.device, lr: float = 1e-3,
    weight_decay: float = 1e-4, warmup_steps: int = 100,
    steps_per_epoch: int = 500, q_ratio: float = 0.0, val_loader=None,
) -> dict:
    """Train on one continuous loader stream; log block means as JSON lines.

    Summary losses describe the final optimizer update. Validation always uses
    real outcomes so its WDL metric is comparable across training Q ratios.
    """
    if steps < 1 or steps_per_epoch < 1:
        raise ValueError("steps and steps_per_epoch must be positive")
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    batches = iter(loader)
    started = time.perf_counter()
    samples_seen, block_steps = 0, 0
    sums = [0.0, 0.0, 0.0]
    val = None
    for step in range(steps):
        try:
            cpu_batch = next(batches)
        except StopIteration as exc:
            raise ValueError("training loader exhausted before requested steps") from exc
        batch = {key: value.to(device) for key, value in cpu_batch.items() if key != "q"}
        current_lr = lr * lr_multiplier(step, steps, warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        loss, p_loss, v_loss = total_loss(*model(batch["planes"]), batch, q_ratio=q_ratio)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite loss at step {step + 1}")
        loss.backward()
        optimizer.step()
        losses = [float(value.detach()) for value in (loss, p_loss, v_loss)]
        sums = [a + b for a, b in zip(sums, losses)]
        samples_seen += len(batch["wdl"])
        block_steps += 1
        del cpu_batch, batch, loss, p_loss, v_loss
        if (step + 1) % steps_per_epoch == 0 or step + 1 == steps:
            val = evaluate(model, val_loader, device) if val_loader is not None else None
            empty_cache(device)
            record = dict(zip(("loss", "policy_loss", "wdl_loss"),
                              (value / block_steps for value in sums)))
            record.update(steps=step + 1, samples_seen=samples_seen,
                          seconds=time.perf_counter() - started, lr=current_lr, val=val)
            print(json.dumps(record, allow_nan=False), flush=True)
            sums, block_steps = [0.0, 0.0, 0.0], 0
    return {"steps": steps, "samples_seen": samples_seen,
            "seconds": time.perf_counter() - started, "final_loss": losses[0],
            "policy_loss": losses[1], "wdl_loss": losses[2], "val": val}
