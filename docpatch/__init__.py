"""DocPatch: Continual Knowledge Internalization with Conflict-Aware LoRA Memories.

Implementation of the ICLR 2027 submission (``iclr2027_docpatch.pdf``) as a
standalone package. See ``docpatch/MAPPING.md`` for a section/equation ->
module index, and ``docpatch/README.md`` for usage and how this relates to
the pre-existing ``ctx_to_lora.docpatch`` scaffolding it builds on.
"""

from docpatch.config import (
    CompositionConfig,
    ConflictResolutionConfig,
    DocPatchConfig,
    KnowledgeBankConfig,
    QueryEncoderConfig,
    ReasoningAdapterConfig,
    RouterConfig,
)
from docpatch.conflict_resolution import compute_applicability_mask, resolve_conflicts
from docpatch.knowledge_bank import BankRecord, KnowledgeBank
from docpatch.metadata import KnowledgeMetadata
from docpatch.pipeline import DocPatchAnswer, DocPatchPipeline
from docpatch.reasoning_adapter import ReasoningAdapter
from docpatch.router import DocPatchRouter
from docpatch.value_types import RetrievedKnowledge

__all__ = [
    "BankRecord",
    "CompositionConfig",
    "ConflictResolutionConfig",
    "DocPatchAnswer",
    "DocPatchConfig",
    "DocPatchPipeline",
    "DocPatchRouter",
    "KnowledgeBank",
    "KnowledgeBankConfig",
    "KnowledgeMetadata",
    "QueryEncoderConfig",
    "ReasoningAdapter",
    "ReasoningAdapterConfig",
    "RetrievedKnowledge",
    "RouterConfig",
    "compute_applicability_mask",
    "resolve_conflicts",
]
