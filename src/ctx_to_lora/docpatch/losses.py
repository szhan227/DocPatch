import torch
import torch.nn.functional as F
from torch import Tensor


def compute_task_loss(
    logits: Tensor,
    labels: Tensor,
    answer_mask: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    vocab_size = logits.shape[-1]
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    )
    mask = answer_mask.reshape(-1).to(loss.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (loss * mask).sum() / denom


def compute_source_selection_loss(
    source_logits: Tensor,
    target_source_ids: Tensor | None = None,
    soft_targets: Tensor | None = None,
) -> Tensor:
    if soft_targets is not None:
        log_probs = F.log_softmax(source_logits, dim=-1)
        return F.kl_div(log_probs, soft_targets, reduction="batchmean")
    if target_source_ids is None:
        raise ValueError("target_source_ids or soft_targets must be provided")
    return F.cross_entropy(source_logits, target_source_ids)


def compute_conflict_suppression_loss(
    chunk_gates: Tensor,
    conflict_pairs: list[list[tuple[int, int, int]]],
    margin: float = 0.3,
) -> Tensor:
    """Hinge loss for old/new conflicting chunk gates.

    conflict_pairs is batch-aligned. Each tuple is
    (old_chunk_index, new_chunk_index, winner_index), where winner_index is
    0 when the old chunk should win and 1 when the new chunk should win.
    """

    losses = []
    for batch_idx, pairs in enumerate(conflict_pairs):
        for old_idx, new_idx, winner_idx in pairs:
            old_gate = chunk_gates[batch_idx, old_idx]
            new_gate = chunk_gates[batch_idx, new_idx]
            if winner_idx == 1:
                losses.append(torch.clamp(old_gate - new_gate + margin, min=0))
            else:
                losses.append(torch.clamp(new_gate - old_gate + margin, min=0))
    if not losses:
        return chunk_gates.new_tensor(0.0)
    return torch.stack(losses).mean()


def compute_locality_loss(
    logits_with_lora: Tensor,
    logits_base: Tensor,
    answer_mask: Tensor,
) -> Tensor:
    log_probs_with_lora = F.log_softmax(logits_with_lora, dim=-1)
    probs_base = F.softmax(logits_base.detach(), dim=-1)
    kl = F.kl_div(log_probs_with_lora, probs_base, reduction="none").sum(dim=-1)
    mask = answer_mask.to(kl.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (kl * mask).sum() / denom


def compute_sparsity_loss(chunk_gates: Tensor, method: str = "l1") -> Tensor:
    if method == "l1":
        return chunk_gates.abs().mean()
    if method == "entropy":
        eps = 1e-8
        gates = chunk_gates.clamp(eps, 1 - eps)
        entropy = -(gates * torch.log(gates) + (1 - gates) * torch.log(1 - gates))
        return entropy.mean()
    raise ValueError(f"Unknown sparsity method: {method}")


def compute_total_loss(
    task_loss: Tensor,
    source_loss: Tensor,
    conflict_loss: Tensor,
    locality_loss: Tensor,
    sparsity_loss: Tensor,
    weights: dict[str, float] | None = None,
) -> Tensor:
    weights = {
        "task": 1.0,
        "source": 0.5,
        "conflict": 0.3,
        "locality": 0.2,
        "sparsity": 0.1,
        **(weights or {}),
    }
    return (
        weights["task"] * task_loss
        + weights["source"] * source_loss
        + weights["conflict"] * conflict_loss
        + weights["locality"] * locality_loss
        + weights["sparsity"] * sparsity_loss
    )
