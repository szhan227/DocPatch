"""Configuration dataclasses for the DocPatch framework.

Field names follow the notation used in the paper (arXiv/ICLR 2027,
"DocPatch: Continual Knowledge Internalization with Conflict-Aware LoRA
Memories") wherever practical, so config -> equation lookups stay obvious.
See ``docpatch/MAPPING.md`` for the full paper-section-to-code index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class KnowledgeBankConfig:
    """Section 3.2 -- offline knowledge bank construction."""

    d2l_checkpoint: str = "trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
    output_dir: str = "cached_loras/docpatch_bank"
    manifest_name: str = "manifest.jsonl"
    chunk_size: int = 512
    chunk_overlap: int = 128
    sketch_dim: int = 64  # p in Appendix A.3 (P_in/P_out project to R^p)
    feature_dim: int = 256  # output dim of z_i (router representation space)
    use_flash_attn: bool = False


@dataclass
class QueryEncoderConfig:
    """Appendix A.2 -- query representation extraction."""

    pooling: str = "mean"  # "mean" or "last" over non-padding token positions
    feature_dim: int = 256  # output dim of e_q, must match KnowledgeBankConfig.feature_dim
    max_length: int = 64


@dataclass
class RouterConfig:
    """Section 3.3 -- query-dependent knowledge routing."""

    hidden_size: int = 256  # shared routing space dim (e_q and z_i live here)
    router_hidden_size: int = 512
    max_sources: int = 64
    dropout: float = 0.1
    top_k: int = 5
    source_loss_weight: float = 0.5  # lambda_source in Eq. 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01


@dataclass
class ConflictResolutionConfig:
    """Section 3.4 -- conflict-aware knowledge resolution."""

    # When True, a missing conflict_group on a retrieved chunk means it never
    # competes with anything else (m_i = 1 unconditionally).
    treat_missing_group_as_unique: bool = True


@dataclass
class CompositionConfig:
    """Section 3.5 -- multi-LoRA knowledge composition."""

    mode: str = "rank_concat"  # "rank_concat" (Eq. 11-12/23-25) or "weighted_sum"/"uniform" for ablations


@dataclass
class ReasoningAdapterConfig:
    """Section 3.6 -- global reasoning adapter and training."""

    rank: int = 8
    target_modules: tuple[str, ...] = ("down_proj",)
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    num_steps: int = 40_000
    curriculum_stage_fractions: tuple[float, float, float, float] = (
        0.30,  # ground-truth single-LoRA activation
        0.25,  # counterfactual pairs (Appendix A.6)
        0.25,  # multi-LoRA / multi-hop reasoning
        0.20,  # versioned conflicts after metadata-based resolution
    )


@dataclass
class DocPatchConfig:
    """Top-level configuration bundling every DocPatch component."""

    backbone_name_or_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    knowledge_bank: KnowledgeBankConfig = field(default_factory=KnowledgeBankConfig)
    query_encoder: QueryEncoderConfig = field(default_factory=QueryEncoderConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    conflict_resolution: ConflictResolutionConfig = field(default_factory=ConflictResolutionConfig)
    composition: CompositionConfig = field(default_factory=CompositionConfig)
    reasoning_adapter: ReasoningAdapterConfig = field(default_factory=ReasoningAdapterConfig)
    seed: int = 0

    @property
    def bank_dir(self) -> Path:
        return Path(self.knowledge_bank.output_dir)
