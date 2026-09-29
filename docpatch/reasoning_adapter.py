"""Section 3.6 -- global reasoning adapter.

    h' = W_0 h + Delta W_know h + Delta W_reason h              (Eq. 13)
       = (W_0 + Delta W_know + Delta W_reason) h                (Eq. 14)

Delta W_reason is a single LoRA shared across all documents and queries
(unlike the document-specific, frozen knowledge LoRAs): it is not meant to
store factual knowledge, only to teach the frozen base LLM "general
behaviors for using the parametric knowledge supplied by the activated
memories" (Section 3.6, third paragraph).

Because Eq. 14 is just two additive low-rank deltas summed into one layer
computation, ``ReasoningAdapter`` reuses the exact rank-concatenation
composition primitive from ``docpatch/composition.py``: the trainable
reasoning A/B tensors are concatenated onto the (frozen, cached) composed
knowledge tree with weight 1.0, which reproduces Eq. 14 exactly by the same
Appendix A.1 argument that justifies Eq. 12 for knowledge LoRAs -- summing
low-rank deltas of the same shape is exact rank concatenation, not a
lower-rank approximation.

Training (Section 3.6, last two paragraphs + Appendix A.6):
    L_answer = -sum_t log p_theta_reason(y_t | q, y_<t; K)       (Eq. 15)
gradients update only the reasoning adapter; the base model,
document-specific knowledge LoRAs, query encoder, and router all stay
frozen. See ``docpatch/data/curriculum.py`` for the staged training order
and ``docpatch/data/counterfactual.py`` for the counterfactual pairs that
keep the adapter from memorizing a fixed question -> answer mapping.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ctx_to_lora.modeling.lora_layer import apply_lora_to_layers
from docpatch.composition import compose


class ReasoningAdapter(nn.Module):
    """theta_reason: one trainable (A, B) pair per target module, shared globally."""

    def __init__(
        self,
        layer_indices: list[int],
        dims: dict[str, tuple[int, int]],
        rank: int = 8,
    ):
        super().__init__()
        self.layer_indices = list(layer_indices)
        self.rank = rank
        self.dims = dict(dims)
        n_layers = len(self.layer_indices)
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        for module, (d_in, d_out) in dims.items():
            # Standard LoRA init: A ~ Kaiming-uniform (nonzero), B = 0, so the
            # adapter starts as an exact no-op (Delta W_reason = B^T A = 0).
            a = torch.empty(n_layers, rank, d_in)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.A[module] = nn.Parameter(a)
            self.B[module] = nn.Parameter(torch.zeros(n_layers, rank, d_out))

    def lora_tree(self) -> dict[str, dict[str, Tensor]]:
        """Delta W_reason as a live (grad-tracking) LoRA tree, batch dim = 1."""

        return {
            module: {"A": self.A[module].unsqueeze(0), "B": self.B[module].unsqueeze(0)}
            for module in self.dims
        }

    def compose_with_knowledge(
        self, knowledge_tree: dict[str, dict[str, Tensor]] | None
    ) -> dict[str, dict[str, Tensor]]:
        """Eq. 14: Delta W_know + Delta W_reason, via exact rank concatenation."""

        reasoning_tree = self.lora_tree()
        if knowledge_tree is None:
            return reasoning_tree
        # Cached knowledge LoRAs are loaded on CPU; the adapter may live on GPU.
        device = next(self.parameters()).device
        knowledge_tree = {
            module: {key: t.to(device) for key, t in tree.items()}
            for module, tree in knowledge_tree.items()
        }
        return compose(
            _as_retrieved_pair(knowledge_tree, reasoning_tree),
            gates=[1.0, 1.0],
            mode="rank_concat",
        )


def _as_retrieved_pair(knowledge_tree, reasoning_tree):
    # Local import to avoid a module-level cycle (composition <-> types).
    from docpatch.metadata import KnowledgeMetadata
    from docpatch.value_types import RetrievedKnowledge

    dummy_meta = KnowledgeMetadata(source_id="__reasoning__", chunk_id=-1)
    return [
        RetrievedKnowledge("know", "", dummy_meta, 0.0, 1.0, lora_tree=knowledge_tree),
        RetrievedKnowledge("reason", "", dummy_meta, 0.0, 1.0, lora_tree=reasoning_tree),
    ]


def apply_reasoning_and_knowledge(
    model,
    reasoning_adapter: ReasoningAdapter,
    knowledge_tree: dict[str, dict[str, Tensor]] | None,
) -> None:
    """Monkey-patch the base model's target linear layers for one forward/generate call.

    Mirrors how the D2L hypernetwork itself activates generated LoRAs
    (``ModulatedPretrainedModel`` patches ``nn.Linear.forward`` via
    ``apply_lora_to_layers``); this lets ``Delta W_know + Delta W_reason``
    (Eq. 14) be applied with the exact same mechanism the base model already
    uses for its own knowledge LoRAs, batch size 1 (single query at a time).
    """

    composed = reasoning_adapter.compose_with_knowledge(knowledge_tree)
    n_qs = torch.ones(1, dtype=torch.long, device=next(reasoning_adapter.parameters()).device)
    apply_lora_to_layers(model, reasoning_adapter.layer_indices, composed, n_qs)


def reasoning_adapter_from_hypernet(model, rank: int = 8, target_modules: tuple[str, ...] | None = None) -> ReasoningAdapter:
    """Build a ``ReasoningAdapter`` sized to match a loaded D2L hypernetwork's
    target layers/dims, so it can be composed with that model's knowledge
    LoRAs (same layer_indices, same d_in/d_out per module)."""

    hypernet = model.hypernet
    modules = target_modules or hypernet.target_modules
    dims = {module: (hypernet.d_in[module], hypernet.d_out[module]) for module in modules}
    return ReasoningAdapter(layer_indices=list(hypernet.layer_indices), dims=dims, rank=rank)
