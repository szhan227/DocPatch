"""Shared value types threaded through the DocPatch pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from docpatch.metadata import KnowledgeMetadata


@dataclass
class RetrievedKnowledge:
    """One candidate knowledge LoRA L_i retrieved for a query, i in C_q / S_q.

    ``alpha`` is the router's normalized weight (Eq. 2-3, alpha_i =
    Softmax(s_i)); ``lora_path`` points at the cached ``.pt`` produced by the
    offline knowledge bank (Section 3.2); ``metadata`` is mu_i.
    """

    cache_key: str
    lora_path: str
    metadata: KnowledgeMetadata
    router_score: float  # s_i, raw router logit (Eq. 2)
    alpha: float  # softmax-normalized routing weight over the candidate set (Eq. 3)
    lora_tree: dict[str, dict[str, Any]] | None = None  # optionally pre-loaded {A, B} tensors
