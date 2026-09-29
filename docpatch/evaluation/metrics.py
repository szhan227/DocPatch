"""Metrics for Tables 1-6 of the paper.

- QA: EM / token-level F1 (Table 1, Section 4.1: "we report EM and
  token-level F1").
- Routing: Recall / F1 / full-coverage Accuracy over the top-k retrieved set
  (Eq. 16, Table 2).
- Continual updating: per-category accuracy over
  {Unchanged, Updated, Old-Only, Override, Unrelated} (Table 4).
"""

from __future__ import annotations

import re
import string
from collections import Counter

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# QA: EM / F1 (Table 1)
# ---------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def qa_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must be the same length")
    n = max(len(predictions), 1)
    em = sum(exact_match(p, r) for p, r in zip(predictions, references)) / n
    f1 = sum(token_f1(p, r) for p, r in zip(predictions, references)) / n
    return {"em": em, "f1": f1}


# ---------------------------------------------------------------------------
# Routing: Recall / F1 / Accuracy (Eq. 16, Table 2)
# ---------------------------------------------------------------------------


def routing_metrics(chunk_gates: Tensor, ground_truth_indices: set[int], top_k: int) -> dict[str, float]:
    """chunk_gates: [num_candidates] router probabilities p_i for one query."""

    k = min(top_k, chunk_gates.numel())
    if k == 0 or not ground_truth_indices:
        return {"recall": 0.0, "f1": 0.0, "accuracy": 0.0}
    top_indices = set(torch.topk(chunk_gates, k).indices.tolist())
    intersection = top_indices & ground_truth_indices
    recall = len(intersection) / len(ground_truth_indices)
    precision = len(intersection) / len(top_indices) if top_indices else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = float(ground_truth_indices.issubset(top_indices))
    return {"recall": recall, "f1": f1, "accuracy": accuracy}


def aggregate_routing_metrics(per_query: list[dict[str, float]]) -> dict[str, float]:
    n = max(len(per_query), 1)
    return {
        key: sum(m[key] for m in per_query) / n
        for key in ("recall", "f1", "accuracy")
    }


# ---------------------------------------------------------------------------
# Continual updating categories (Table 4)
# ---------------------------------------------------------------------------

CONTINUAL_CATEGORIES = ("unchanged", "updated", "old_only", "override", "unrelated")


def continual_update_accuracy(
    predictions: list[str],
    references: list[str],
    categories: list[str],
) -> dict[str, float]:
    """EM accuracy broken down by continual-update test category.

    ``categories[i]`` must be one of ``CONTINUAL_CATEGORIES`` for example i
    (Unchanged/Updated/Old-Only/Override/Unrelated, Section 4.5).
    """

    buckets: dict[str, list[float]] = {c: [] for c in CONTINUAL_CATEGORIES}
    for pred, ref, category in zip(predictions, references, categories):
        if category not in buckets:
            raise ValueError(f"unknown continual-update category: {category}")
        buckets[category].append(exact_match(pred, ref))
    return {
        category: (sum(scores) / len(scores) if scores else float("nan"))
        for category, scores in buckets.items()
    }
