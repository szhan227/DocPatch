from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


def _tensor_stats(tensor: Tensor) -> Tensor:
    x = tensor.detach().float().reshape(-1)
    if x.numel() == 0:
        return torch.zeros(7, dtype=torch.float32)
    return torch.tensor(
        [
            float(x.mean()),
            float(x.std(unbiased=False)),
            float(x.abs().mean()),
            float(x.norm(p=2)),
            float(x.abs().max()),
            float((x * x).mean()),
            float(x.numel()),
        ],
        dtype=torch.float32,
    )


def lora_stat_features(
    lora_tree: dict[str, dict[str, Tensor]],
    include_module_order: bool = True,
) -> Tensor:
    """Extract compact non-text features from LoRA weights.

    The output is deterministic and contains only statistics of the adapter
    tensors, not source text.
    """

    parts = []
    module_names = sorted(lora_tree) if include_module_order else list(lora_tree)
    for module in module_names:
        weights = lora_tree[module]
        a = weights["A"]
        b = weights["B"]
        parts.extend(
            [
                _tensor_stats(a),
                _tensor_stats(b),
                torch.tensor(
                    [
                        float(a.shape[-2] if a.ndim >= 2 else 0),
                        float(a.shape[-1] if a.ndim >= 1 else 0),
                        float(b.shape[-2] if b.ndim >= 2 else 0),
                        float(b.shape[-1] if b.ndim >= 1 else 0),
                    ],
                    dtype=torch.float32,
                ),
            ]
        )
    return torch.cat(parts, dim=0) if parts else torch.empty(0, dtype=torch.float32)


def rank1_lora_stat_features(
    lora_tree: dict[str, dict[str, Tensor]],
) -> Tensor:
    """Extract one deterministic feature vector per aligned rank-1 atom."""

    if not lora_tree:
        return torch.empty(0, 0, dtype=torch.float32)
    first = next(iter(lora_tree.values()))["A"]
    rank = first.shape[-2]
    rows = []
    for rank_idx in range(rank):
        parts = []
        for module in sorted(lora_tree):
            weights = lora_tree[module]
            if weights["A"].shape[-2] != rank or weights["B"].shape[-2] != rank:
                raise ValueError("all LoRA modules must use the same rank")
            parts.extend(
                [
                    _tensor_stats(weights["A"].select(-2, rank_idx)),
                    _tensor_stats(weights["B"].select(-2, rank_idx)),
                ]
            )
        rows.append(torch.cat(parts))
    return torch.stack(rows)


def load_lora_feature(
    lora_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> Tensor:
    payload = torch.load(lora_path, map_location=map_location, weights_only=False)
    if "lora_feature" in payload:
        return payload["lora_feature"].float()
    return lora_stat_features(payload["lora_weights"])


class LoraFeatureProjector(nn.Module):
    """Project LoRA-stat vectors into the router hidden size."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        projector_hidden_size: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        projector_hidden_size = projector_hidden_size or max(hidden_size, input_size)
        self.net = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, projector_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projector_hidden_size, hidden_size),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class LoraFeatureStore:
    def __init__(
        self,
        records: list[dict[str, Any]],
        features: Tensor,
        rank1_features: Tensor | None = None,
    ):
        self.records = records
        self.features = features.float()
        self.rank1_features = (
            rank1_features.float() if rank1_features is not None else None
        )
        self.by_cache_key = {
            record["cache_key"]: idx for idx, record in enumerate(records)
        }

    @classmethod
    def load(cls, path: str | Path) -> "LoraFeatureStore":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            payload["records"],
            payload["features"],
            payload.get("rank1_features"),
        )

    def get(self, cache_key: str) -> Tensor:
        return self.features[self.by_cache_key[cache_key]]

    def get_rank1(self, cache_key: str) -> Tensor:
        if self.rank1_features is None:
            raise ValueError("feature store does not contain rank-1 features")
        return self.rank1_features[self.by_cache_key[cache_key]]
