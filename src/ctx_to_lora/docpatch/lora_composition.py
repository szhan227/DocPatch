from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def _rank_gates_for_tensor(rank_gates: Tensor, tensor: Tensor) -> Tensor:
    """Align gates ending in rank with a LoRA tensor ending in [rank, dim]."""
    gates = rank_gates.to(device=tensor.device, dtype=tensor.dtype)
    if gates.shape[-1] != tensor.shape[-2]:
        raise ValueError(
            f"rank gate size {gates.shape[-1]} does not match rank {tensor.shape[-2]}"
        )
    if gates.ndim > tensor.ndim - 1:
        raise ValueError("rank gates have too many dimensions for the LoRA tensor")
    while gates.ndim < tensor.ndim - 1:
        gates = gates.unsqueeze(0)
    return gates.unsqueeze(-1)


def load_cached_lora(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def lora_weight_tree(cache_entry: dict[str, Any]) -> dict[str, dict[str, Tensor]]:
    if "lora_weights" not in cache_entry:
        raise ValueError("cache entry missing lora_weights")
    return cache_entry["lora_weights"]


def scale_lora_tree(
    lora_tree: dict[str, dict[str, Tensor]],
    scale: float | Tensor,
) -> dict[str, dict[str, Tensor]]:
    # Scaling A is equivalent to scaling the low-rank delta A@B by scale.
    return {
        module: {
            "A": weights["A"] * scale,
            "B": weights["B"],
        }
        for module, weights in lora_tree.items()
    }


def sum_lora_trees(
    lora_trees: list[dict[str, dict[str, Tensor]]],
    weights: list[float] | Tensor | None = None,
) -> dict[str, dict[str, Tensor]]:
    if not lora_trees:
        raise ValueError("at least one LoRA tree is required")
    if weights is None:
        weights_tensor = torch.ones(len(lora_trees), dtype=torch.float32)
    elif torch.is_tensor(weights):
        weights_tensor = weights.detach().cpu().to(torch.float32)
    else:
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
    if len(weights_tensor) != len(lora_trees):
        raise ValueError("weights length must match lora_trees length")

    modules = lora_trees[0].keys()
    out: dict[str, dict[str, Tensor]] = {}
    for module in modules:
        out[module] = {}
        for key in ("A", "B"):
            tensors = [
                tree[module][key].to(torch.float32) * float(weights_tensor[idx])
                for idx, tree in enumerate(lora_trees)
            ]
            out[module][key] = torch.stack(tensors, dim=0).sum(dim=0)
    return out


def concatenate_lora_trees(
    lora_trees: list[dict[str, dict[str, Tensor]]],
    weights: list[float] | Tensor | None = None,
) -> dict[str, dict[str, Tensor]]:
    """Compose LoRAs by rank concatenation.

    This preserves each selected chunk's rank instead of summing matrices with
    the same rank. The scale is applied to A so each delta is weighted.
    """

    if not lora_trees:
        raise ValueError("at least one LoRA tree is required")
    if weights is None:
        weights_tensor = torch.ones(len(lora_trees), dtype=torch.float32)
    elif torch.is_tensor(weights):
        weights_tensor = weights.detach().cpu().to(torch.float32)
    else:
        weights_tensor = torch.tensor(weights, dtype=torch.float32)

    out: dict[str, dict[str, Tensor]] = {}
    for module in lora_trees[0].keys():
        a_parts = []
        b_parts = []
        for idx, tree in enumerate(lora_trees):
            a_parts.append(tree[module]["A"].to(torch.float32) * float(weights_tensor[idx]))
            b_parts.append(tree[module]["B"].to(torch.float32))
        out[module] = {
            "A": torch.cat(a_parts, dim=-2),
            "B": torch.cat(b_parts, dim=-2),
        }
    return out


def concatenate_rank1_lora_trees(
    lora_trees: list[dict[str, dict[str, Tensor]]],
    rank_gates: list[Tensor] | Tensor,
) -> dict[str, dict[str, Tensor]]:
    """Exactly compose LoRAs with independent gates for each rank-1 atom.

    Each gate tensor ends in the adapter rank and may optionally include layer
    dimensions. A gate is applied to A only, ensuring each outer product is
    scaled exactly once before all atoms are concatenated along the rank axis.
    """

    if not lora_trees:
        raise ValueError("at least one LoRA tree is required")
    if torch.is_tensor(rank_gates):
        if rank_gates.shape[0] != len(lora_trees):
            raise ValueError("rank_gates first dimension must match lora_trees")
        gates_by_tree = list(rank_gates.unbind(0))
    else:
        gates_by_tree = rank_gates
    if len(gates_by_tree) != len(lora_trees):
        raise ValueError("rank_gates length must match lora_trees")

    modules = lora_trees[0].keys()
    if any(tree.keys() != modules for tree in lora_trees[1:]):
        raise ValueError("all LoRA trees must contain the same modules")

    out: dict[str, dict[str, Tensor]] = {}
    for module in modules:
        a_parts = []
        b_parts = []
        for tree, gates in zip(lora_trees, gates_by_tree):
            a = tree[module]["A"]
            b = tree[module]["B"]
            if a.shape[:-1] != b.shape[:-1]:
                raise ValueError(f"A/B rank axes differ for module {module}")
            a_parts.append(a * _rank_gates_for_tensor(gates, a))
            b_parts.append(b)
        out[module] = {
            "A": torch.cat(a_parts, dim=-2),
            "B": torch.cat(b_parts, dim=-2),
        }
    return out


def apply_rank_gates(
    lora_tree: dict[str, dict[str, Tensor]],
    rank_gates: Tensor,
) -> dict[str, dict[str, Tensor]]:
    """Apply fine-grained gates over LoRA rank dimensions.

    rank_gates can be shape [rank] or broadcastable to each module's A matrix
    rank axis. Gates are applied to A so each rank contribution is scaled once.
    """

    out: dict[str, dict[str, Tensor]] = {}
    for module, weights in lora_tree.items():
        out[module] = {
            "A": weights["A"]
            * _rank_gates_for_tensor(rank_gates, weights["A"]),
            "B": weights["B"],
        }
    return out


def rank1_self_activation_scores(
    hidden_states: Tensor,
    a_keys: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Compute MoRAM intrinsic relevance for rank-1 atoms.

    hidden_states has shape [..., sequence, d_in] and a_keys has shape
    [..., rank, d_in]. Leading dimensions must be broadcastable. The returned
    scores have shape [..., sequence, rank] and are L2-normalized over atoms.
    """

    if hidden_states.shape[-1] != a_keys.shape[-1]:
        raise ValueError("hidden states and A keys must share d_in")
    activations = torch.matmul(hidden_states, a_keys.transpose(-1, -2))
    denominator = activations.square().sum(dim=-1, keepdim=True).sqrt()
    return activations / denominator.clamp_min(eps)


def sparse_rank1_weights(
    scores: Tensor,
    top_k: int,
    temperature: float = 1.0,
    threshold: float | None = None,
) -> Tensor:
    """Apply MoRAM top-k masking, temperature softmax, and optional pruning."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    k = min(top_k, scores.shape[-1])
    top_values, top_indices = scores.topk(k, dim=-1)
    masked = torch.full_like(scores, torch.finfo(scores.dtype).min)
    masked.scatter_(-1, top_indices, top_values)
    weights = torch.softmax(masked / temperature, dim=-1)
    if threshold is not None:
        weights = weights * (scores >= threshold).to(weights.dtype)
    return weights


def moram_rank1_delta(
    hidden_states: Tensor,
    a_keys: Tensor,
    b_values: Tensor,
    top_k: int,
    temperature: float = 1.0,
    threshold: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply a token-dependent sparse mixture of rank-1 LoRA atoms.

    b_values uses the repository convention [..., rank, d_out]. Returns the
    dynamic LoRA output and the sparse atom weights.
    """

    if a_keys.shape[-2] != b_values.shape[-2]:
        raise ValueError("A keys and B values must have the same rank")
    scores = rank1_self_activation_scores(hidden_states, a_keys)
    weights = sparse_rank1_weights(scores, top_k, temperature, threshold)
    atom_activations = torch.matmul(hidden_states, a_keys.transpose(-1, -2))
    delta = torch.matmul(atom_activations * weights, b_values)
    return delta, weights


def compose_cached_loras(
    lora_paths: list[str | Path],
    weights: list[float] | Tensor | None = None,
    mode: str = "rank_concat",
    map_location: str | torch.device = "cpu",
) -> dict[str, dict[str, Tensor]]:
    trees = [
        lora_weight_tree(load_cached_lora(path, map_location=map_location))
        for path in lora_paths
    ]
    if mode == "rank_concat":
        return concatenate_lora_trees(trees, weights)
    if mode == "weighted_sum":
        return sum_lora_trees(trees, weights)
    raise ValueError(f"Unknown composition mode: {mode}")


def compose_cached_loras_rank1(
    lora_paths: list[str | Path],
    rank_gates: list[Tensor] | Tensor,
    map_location: str | torch.device = "cpu",
) -> dict[str, dict[str, Tensor]]:
    """Load cached LoRAs and compose them with per-atom rank gates."""

    trees = [
        lora_weight_tree(load_cached_lora(path, map_location=map_location))
        for path in lora_paths
    ]
    return concatenate_rank1_lora_trees(trees, rank_gates)


def select_topk_lora_paths(
    manifest_records,
    chunk_indices: Tensor,
    chunk_weights: Tensor,
    top_k: int,
) -> tuple[list[str], list[float]]:
    selected_paths = []
    selected_weights = []
    k = min(top_k, chunk_indices.numel())
    for idx in range(k):
        record = manifest_records[int(chunk_indices[idx])]
        selected_paths.append(record.lora_path)
        selected_weights.append(float(chunk_weights[idx]))
    return selected_paths, selected_weights
