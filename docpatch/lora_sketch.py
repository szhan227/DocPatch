"""Appendix A.3 -- LoRA-to-feature mapping.

For an adapted layer l, Delta W_i^(l) = B_i^(l) A_i^(l). Fixed projection
matrices P_in in R^{d_in x p}, P_out in R^{d_out x p} (p < d_in, d_out) give a
low-dimensional sketch

    S_i^(l) = P_out^T B_i^(l) A_i^(l) P_in = (P_out^T B_i^(l)) (A_i^(l) P_in)  in R^{p x p}

which is evaluated via the right-hand factored form so the dense
Delta W_i^(l) is never materialized. Layer-wise sketches are flattened,
pooled over layers, and mapped into the shared routing space by a learned
projector P_LoRA to produce z_i.

This is a faithful implementation of the paper's exact construction. The
existing ``ctx_to_lora.docpatch.lora_features`` module instead uses
hand-crafted tensor statistics (mean/std/norm/...) as a simpler stand-in --
that remains available for quick experimentation, but this module is what
the paper actually specifies and is what ``KnowledgeBank`` uses by default.
"""

from __future__ import annotations

import zlib

import torch
from torch import Tensor, nn


class FixedRandomProjections(nn.Module):
    """Deterministic, frozen P_in^(l)/P_out^(l) per adapted module.

    Projections are seeded so the same module/dim pair always yields the same
    matrices without needing to persist them separately from a run's seed --
    convenient since the knowledge bank is meant to be append-only and stable
    across process restarts.
    """

    def __init__(self, sketch_dim: int, seed: int = 0):
        super().__init__()
        self.sketch_dim = sketch_dim
        self.seed = seed
        self._cache: dict[tuple[str, int, int], tuple[Tensor, Tensor]] = {}

    def get(self, module: str, d_in: int, d_out: int, device, dtype) -> tuple[Tensor, Tensor]:
        key = (module, d_in, d_out)
        if key not in self._cache:
            # zlib.crc32, not hash(): str hashes are salted per process, which would
            # give different projections (and different features) on every run.
            key_bytes = f"{self.seed}|{module}|{d_in}|{d_out}".encode()
            gen = torch.Generator(device="cpu").manual_seed(zlib.crc32(key_bytes))
            p_in = torch.randn(d_in, self.sketch_dim, generator=gen) / (d_in**0.5)
            p_out = torch.randn(d_out, self.sketch_dim, generator=gen) / (d_out**0.5)
            self._cache[key] = (p_in, p_out)
        p_in, p_out = self._cache[key]
        return p_in.to(device=device, dtype=dtype), p_out.to(device=device, dtype=dtype)

    def sketch_layer(self, a: Tensor, b: Tensor, module: str) -> Tensor:
        """S_i^(l) = P_out^T B_paper A P_in, contracted over rank r.

        Repo convention stores both factors as [..., r, dim] (A: [..., r,
        d_in], B: [..., r, d_out] == B_paper^T), so the projected factors
        a_proj = A @ P_in and b_proj = B @ P_out (both [..., r, p]) combine
        via a contraction over r to give the same [p, p] sketch the paper's
        [d_out, r] x [r, p] x ... form would.
        """

        d_in = a.shape[-1]
        d_out = b.shape[-1]
        p_in, p_out = self.get(module, d_in, d_out, a.device, a.dtype)
        # a: [..., r, d_in] @ p_in [d_in, p] -> [..., r, p]
        a_proj = torch.matmul(a, p_in)
        # b: [..., r, d_out] @ p_out [d_out, p] -> [..., r, p]; then combine ranks.
        b_proj = torch.matmul(b, p_out)
        # S = (P_out^T B)(A P_in) = b_proj^T @ a_proj, contracted over rank r.
        sketch = torch.einsum("...rp,...rq->...pq", b_proj, a_proj)
        return sketch


def compute_lora_sketch(
    lora_tree: dict[str, dict[str, Tensor]],
    projections: FixedRandomProjections,
    pool: str = "mean",
) -> Tensor:
    """Flatten + pool per-layer sketches across modules into one feature vector."""

    parts = []
    for module in sorted(lora_tree):
        a, b = lora_tree[module]["A"], lora_tree[module]["B"]
        sketch = projections.sketch_layer(a, b, module)  # [..., n_layers, p, p] or [n_layers, p, p]
        flat = sketch.reshape(*sketch.shape[:-2], -1)  # [..., n_layers, p*p]
        if flat.ndim >= 2:
            pooled = flat.mean(dim=-2) if pool == "mean" else flat.amax(dim=-2)
        else:
            pooled = flat
        parts.append(pooled.reshape(-1))
    return torch.cat(parts, dim=0)


class LoRAFeatureProjector(nn.Module):
    """Learned P_LoRA: maps the pooled sketch vector into the routing space."""

    def __init__(self, sketch_feature_dim: int, routing_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or max(routing_dim, sketch_feature_dim // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(sketch_feature_dim),
            nn.Linear(sketch_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, routing_dim),
        )

    def forward(self, sketch_features: Tensor) -> Tensor:
        return self.net(sketch_features)
