"""Knowledge-LoRA metadata: source, chunk id, version, conflict-group id.

Paper section 3.2: "Each knowledge LoRA is stored together with metadata
mu_i, including its source, chunk identifier, document version, and conflict
information." Section 3.4 additionally assumes conflict-group identifiers are
available at knowledge-bank construction time: "incompatible alternatives of
the same underlying information share a conflict-group identifier, while
version metadata specifies their precedence" (also Appendix A.5).

``ctx_to_lora.docpatch.cache.ChunkLoraRecord`` already carries source/chunk
id/version; this module adds the conflict-group id on top and derives it
automatically for the synthetic-conflict DocPatch dataset format used in
``data_preprocess/outputs/mixed/*.jsonl`` (each chunk there tags the
underlying fact with a ``fact_slot`` index -- chunks across document versions
that share an entity + fact_slot are alternatives of the same fact and hence
belong to the same conflict group).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ctx_to_lora.docpatch.data import SourceChunkRef


@dataclass(frozen=True)
class KnowledgeMetadata:
    """mu_i in the paper: metadata attached to one knowledge LoRA L_i."""

    source_id: str
    chunk_id: int
    version: int = 0
    conflict_group: str | None = None

    @property
    def precedence_key(self) -> tuple[int, str]:
        # Ties within a version are broken deterministically by source_id so
        # resolution is reproducible.
        return (self.version, self.source_id)


def infer_conflict_group(
    example: dict[str, Any],
    ref: SourceChunkRef,
) -> str | None:
    """Derive a conflict-group id for one chunk of a DocPatch training example.

    Uses the ``fact_slot`` annotation produced by
    ``data_preprocess/build_synthetic_conflicts.py`` when present: chunks
    across the example's document versions that encode the same fact slot are
    alternatives of one another. Falls back to explicit
    ``conflict_info.conflicting_chunks`` membership, then to ``None`` (no
    group -> the chunk never competes with anything, matching Section 3.4's
    m_i = 1 default for standalone knowledge).
    """

    doc = example["documents"].get(ref.source_key, {})
    chunk_lookup = {int(c["chunk_id"]): c for c in doc.get("chunks", [])}
    chunk = chunk_lookup.get(ref.chunk_id, {})
    fact_slot = chunk.get("fact_slot")
    if fact_slot is not None:
        return f"{example['id']}::slot_{fact_slot}"

    conflict_info = example.get("conflict_info", {})
    conflicting = conflict_info.get("conflicting_chunks", {})
    if ref.source_key in conflicting and ref.chunk_id in conflicting[ref.source_key]:
        return f"{example['id']}::conflict"
    return None


def build_metadata_for_ref(
    example: dict[str, Any],
    ref: SourceChunkRef,
) -> KnowledgeMetadata:
    doc = example["documents"][ref.source_key]
    version = doc.get("source_version")
    return KnowledgeMetadata(
        source_id=ref.source_id,
        chunk_id=ref.chunk_id,
        version=int(version) if version is not None else 0,
        conflict_group=infer_conflict_group(example, ref),
    )
