"""Appendix A.6 -- counterfactual training pairs.

"We construct counterfactual training pairs in which the question remains
fixed while the activated knowledge memory and target answer change. For
example, an earlier-version LoRA may support y_old, whereas a later-version
LoRA for the same question supports y_new. Because the question itself
provides no information about which target is correct, successful
prediction requires conditioning on the active parametric memory."

Built from the ``multi_source_conflict_synthetic`` category of the mixed
DocPatch dataset (``data_preprocess/build_synthetic_conflicts.py``), whose
``conflict_info`` already records ``old_answer``/``new_answer`` and the
``winner_source`` alongside the conflicting chunk ids per document version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ctx_to_lora.docpatch.data import DocPatchDataset, SourceChunkRef


@dataclass
class CounterfactualPair:
    example_id: str
    query: str
    active_chunks: list[SourceChunkRef]
    target_answer: str
    is_winner: bool


def _chunk_refs(example: dict[str, Any], source_key: str, chunk_ids: list[int]) -> list[SourceChunkRef]:
    doc = example["documents"][source_key]
    by_id = {int(c["chunk_id"]): c for c in doc["chunks"]}
    refs = []
    for chunk_id in chunk_ids:
        chunk = by_id.get(int(chunk_id))
        if chunk is None:
            continue
        refs.append(
            SourceChunkRef(source_key=source_key, source_id=doc["source_id"], chunk_id=int(chunk_id), text=chunk["text"])
        )
    return refs


def build_counterfactual_pairs(
    dataset: DocPatchDataset,
    category: str = "multi_source_conflict_synthetic",
) -> list[CounterfactualPair]:
    pairs: list[CounterfactualPair] = []
    for example in dataset.by_category.get(category, []):
        conflict_info = example.get("conflict_info", {})
        if not conflict_info.get("has_conflict"):
            continue
        conflicting_chunks: dict[str, list[int]] = conflict_info.get("conflicting_chunks", {})
        winner_source = conflict_info.get("winner_source")
        old_answer = conflict_info.get("old_answer")
        new_answer = conflict_info.get("new_answer", example["ground_truth"].get("answer"))
        if old_answer is None or new_answer is None:
            continue
        for source_key, chunk_ids in conflicting_chunks.items():
            refs = _chunk_refs(example, source_key, chunk_ids)
            if not refs:
                continue
            is_winner = source_key == winner_source
            pairs.append(
                CounterfactualPair(
                    example_id=example["id"],
                    query=example["query"],
                    active_chunks=refs,
                    target_answer=new_answer if is_winner else old_answer,
                    is_winner=is_winner,
                )
            )
    return pairs
