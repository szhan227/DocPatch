"""Section 3.5 / Appendix A.1 -- exact multi-LoRA knowledge composition.

The rank-concatenation construction (Eq. 11-12, derived exactly in Appendix
A.1 Eq. 19-25) is already implemented faithfully in
``ctx_to_lora.docpatch.lora_composition.concatenate_lora_trees``: gate g_i is
folded into A_i only, then all active A_i/B_i are concatenated along the rank
axis so that B_q @ A_q == sum_i g_i * B_i @ A_i exactly, without introducing
the spurious cross-LoRA terms a naive independent sum of factors would create
(Eq. 21-22). This module just wires that primitive to
``docpatch.value_types.RetrievedKnowledge`` + conflict-resolution gates, and adds
the ablation composition modes used in Table 3 / Table 5
(uniform / router-weighted, Eq. 17-18).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from ctx_to_lora.docpatch.lora_composition import (
    concatenate_lora_trees,
    load_cached_lora,
    lora_weight_tree,
    sum_lora_trees,
)
from docpatch.conflict_resolution import active_indices
from docpatch.value_types import RetrievedKnowledge


def _load_tree(item: RetrievedKnowledge, map_location: str | torch.device) -> dict[str, dict[str, Tensor]]:
    if item.lora_tree is not None:
        return item.lora_tree
    return lora_weight_tree(load_cached_lora(item.lora_path, map_location=map_location))


def compose(
    retrieved: list[RetrievedKnowledge],
    gates: list[float],
    mode: str = "rank_concat",
    map_location: str | torch.device = "cpu",
) -> dict[str, dict[str, Tensor]] | None:
    """Build the query-specific composed update Delta W_know (Eq. 11).

    mode:
        "rank_concat"      -- Eq. 12/23-25, the DocPatch default. Preserves
                               every active LoRA's rank exactly.
        "weighted_sum"      -- naive sum of A_i @ B_i with weight g_i (introduces
                               the cross terms Appendix A.1 warns against; kept
                               only for ablations against the exact construction).
        "uniform"           -- Table 3 "Uniform Composition" baseline (Eq. 17):
                               ignores router weights entirely and rank-concats
                               with 1/|S_q| for every active LoRA.
        "router_weighted"   -- Table 3 "Router-Weighted Composition" baseline
                               (Eq. 18): rank-concats using renormalized router
                               probabilities without conflict-aware masking.
    """

    active = active_indices(gates)
    if not active:
        return None

    if mode == "uniform":
        weights = [1.0 / len(active)] * len(active)
        trees = [_load_tree(retrieved[i], map_location) for i in active]
        return concatenate_lora_trees(trees, weights)

    if mode == "router_weighted":
        alphas = [retrieved[i].alpha for i in active]
        denom = sum(alphas) or 1.0
        weights = [a / denom for a in alphas]
        trees = [_load_tree(retrieved[i], map_location) for i in active]
        return concatenate_lora_trees(trees, weights)

    weights = [gates[i] for i in active]
    trees = [_load_tree(retrieved[i], map_location) for i in active]
    if mode == "rank_concat":
        return concatenate_lora_trees(trees, weights)
    if mode == "weighted_sum":
        return sum_lora_trees(trees, weights)
    raise ValueError(f"Unknown composition mode: {mode}")


def compose_from_paths(
    lora_paths: list[str | Path],
    weights: list[float],
    mode: str = "rank_concat",
    map_location: str | torch.device = "cpu",
) -> dict[str, dict[str, Tensor]]:
    """Convenience path-based entry point (used by scripts/tests)."""

    trees = [lora_weight_tree(load_cached_lora(p, map_location=map_location)) for p in lora_paths]
    if mode == "rank_concat":
        return concatenate_lora_trees(trees, weights)
    if mode == "weighted_sum":
        return sum_lora_trees(trees, weights)
    raise ValueError(f"Unknown composition mode: {mode}")


def verify_exact_composition(
    lora_trees: list[dict[str, dict[str, Tensor]]],
    weights: list[float],
    atol: float = 1e-4,
) -> dict[str, float]:
    """Sanity check for Appendix A.1: rank-concat must equal the literal
    weighted sum sum_i g_i * B_i @ A_i for every adapted module/layer.

    Returns the max absolute difference per module (used by unit tests).
    """

    composed = concatenate_lora_trees(lora_trees, weights)
    diffs: dict[str, float] = {}
    for module, tree in composed.items():
        composed_delta = _delta_from_ab(tree["A"], tree["B"])
        expected = None
        for weight, src in zip(weights, lora_trees):
            a, b = src[module]["A"], src[module]["B"]
            term = float(weight) * _delta_from_ab(a, b)
            expected = term if expected is None else expected + term
        diff = (composed_delta - expected).abs().max().item()
        diffs[module] = diff
        if diff > atol:
            raise AssertionError(f"module {module}: exact composition mismatch {diff} > {atol}")
    return diffs


def _delta_from_ab(a: Tensor, b: Tensor) -> Tensor:
    """Delta W (shape [..., d_out, d_in], nn.Linear.weight convention).

    Repo convention: A has shape [..., r, d_in], B has shape [..., r, d_out]
    (see ``ctx_to_lora.modeling.lora_layer.lora_forward``, which computes
    y = B^T (A x) i.e. Delta W = B^T A when A, B are read as [r, dim]
    matrices), hence the explicit einsum rather than a plain matmul.
    """

    return torch.einsum("...ro,...ri->...oi", b, a)
