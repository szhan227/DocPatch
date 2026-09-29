"""Section 3.3 -- query-dependent knowledge routing.

    s_i = R(e_q, z_i)                                          (Eq. 2)
    p_i = exp(s_i) / sum_{j in C_q} exp(s_j)                   (Eq. 3)
    y_i = 1/|G_q| for i in G_q, else 0                         (Eq. 4, multi-hot)
    L_chunk = -sum_{i in C_q} y_i log p_i                      (Eq. 5)
    p_d = sum_{i in C_q, c(i)=d} p_i                           (Eq. 6)
    L_source = -log p_{d+}                                     (Eq. 7)
    L_router = L_chunk + lambda_source * L_source              (Eq. 8)

``ctx_to_lora.docpatch.router.SourceAwareRouter`` already implements the
scorer R and a *softmax-normalized* p_i exactly as Eq. 2-3 (its
``chunk_gates`` output). This module reuses that scorer as-is and adds the
losses the paper actually specifies: a normalized multi-hot chunk objective
(Eq. 4-5, distinct from ``SourceAwareRouter``'s sibling training script
which uses single-label CE) and a source objective built by summing chunk
probabilities per source (Eq. 6-7) rather than max-pooling logits.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ctx_to_lora.docpatch.router import RouterOutput, SourceAwareRouter, topk_chunk_indices

DocPatchRouter = SourceAwareRouter  # R in the paper; identical architecture, paper-naming alias.

__all__ = [
    "DocPatchRouter",
    "RouterOutput",
    "topk_chunk_indices",
    "build_multi_hot_targets",
    "compute_chunk_selection_loss",
    "compute_source_probabilities",
    "compute_source_loss",
    "compute_router_loss",
]


def build_multi_hot_targets(
    ground_truth_indices: list[list[int]],
    num_candidates: int,
) -> Tensor:
    """Eq. 4: y_i = 1/|G_q| for every ground-truth chunk index, 0 elsewhere.

    ``ground_truth_indices[b]`` lists the positions (within the batch's
    candidate axis) of the supporting knowledge LoRAs G_q for example b.
    """

    batch_size = len(ground_truth_indices)
    targets = torch.zeros(batch_size, num_candidates, dtype=torch.float32)
    for b, indices in enumerate(ground_truth_indices):
        if not indices:
            continue
        weight = 1.0 / len(indices)
        for idx in indices:
            targets[b, idx] = weight
    return targets


def compute_chunk_selection_loss(
    chunk_gates: Tensor,
    multi_hot_targets: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Eq. 5: L_chunk = -sum_i y_i log p_i, averaged over the batch."""

    log_p = torch.log(chunk_gates.clamp_min(eps))
    per_example = -(multi_hot_targets.to(log_p.device, log_p.dtype) * log_p).sum(dim=-1)
    return per_example.mean()


def compute_source_probabilities(
    chunk_gates: Tensor,
    source_ids: Tensor,
    num_sources: int,
) -> Tensor:
    """Eq. 6: p_d = sum_{i: c(i)=d} p_i, via scatter-add of chunk probabilities."""

    batch_size = chunk_gates.shape[0]
    source_probs = torch.zeros(batch_size, num_sources, dtype=chunk_gates.dtype, device=chunk_gates.device)
    source_probs.scatter_add_(1, source_ids, chunk_gates)
    return source_probs


def compute_source_loss(
    source_probs: Tensor,
    target_source_ids: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Eq. 7: L_source = -log p_{d+}."""

    p_target = source_probs.gather(1, target_source_ids[:, None].to(source_probs.device)).squeeze(-1)
    return -torch.log(p_target.clamp_min(eps)).mean()


def compute_router_loss(
    router_output: RouterOutput,
    multi_hot_targets: Tensor,
    source_ids: Tensor,
    target_source_ids: Tensor,
    num_sources: int,
    source_loss_weight: float = 0.5,
) -> dict[str, Tensor]:
    """Eq. 8: L_router = L_chunk + lambda_source * L_source."""

    chunk_loss = compute_chunk_selection_loss(router_output.chunk_gates, multi_hot_targets)
    source_probs = compute_source_probabilities(router_output.chunk_gates, source_ids, num_sources)
    source_loss = compute_source_loss(source_probs, target_source_ids)
    total = chunk_loss + source_loss_weight * source_loss
    return {"total": total, "chunk": chunk_loss, "source": source_loss}
