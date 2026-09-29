"""Unit tests for Appendix A.1 (docpatch/composition.py): exact rank-concat composition."""

import torch

from docpatch.composition import _delta_from_ab, compose, verify_exact_composition
from docpatch.metadata import KnowledgeMetadata
from docpatch.reasoning_adapter import ReasoningAdapter
from docpatch.value_types import RetrievedKnowledge

MODULE = "down_proj"
N_LAYERS, RANK, D_IN, D_OUT = 3, 4, 8, 6


def random_tree(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        MODULE: {
            "A": torch.randn(1, N_LAYERS, RANK, D_IN, generator=g),
            "B": torch.randn(1, N_LAYERS, RANK, D_OUT, generator=g),
        }
    }


def test_rank_concat_matches_literal_weighted_sum_eq25():
    trees = [random_tree(0), random_tree(1), random_tree(2)]
    weights = [0.5, 0.3, 0.2]
    verify_exact_composition(trees, weights, atol=1e-4)  # raises on mismatch


def test_delta_from_ab_matches_naive_double_loop():
    a = torch.randn(RANK, D_IN)
    b = torch.randn(RANK, D_OUT)
    delta = _delta_from_ab(a, b)
    assert delta.shape == (D_OUT, D_IN)
    expected = torch.zeros(D_OUT, D_IN)
    for r in range(RANK):
        expected += torch.outer(b[r], a[r])
    assert torch.allclose(delta, expected, atol=1e-5)


def test_compose_via_retrieved_knowledge_respects_gates():
    dummy_meta = KnowledgeMetadata(source_id="s", chunk_id=0)
    tree_a, tree_b = random_tree(10), random_tree(11)
    retrieved = [
        RetrievedKnowledge("a", "", dummy_meta, 0.0, 0.6, lora_tree=tree_a),
        RetrievedKnowledge("b", "", dummy_meta, 0.0, 0.4, lora_tree=tree_b),
    ]
    gates = [0.6, 0.0]  # b fully suppressed by conflict resolution
    composed = compose(retrieved, gates, mode="rank_concat")
    expected = _delta_from_ab(tree_a[MODULE]["A"], tree_a[MODULE]["B"]) * 0.6
    got = _delta_from_ab(composed[MODULE]["A"], composed[MODULE]["B"])
    assert torch.allclose(got, expected, atol=1e-4)


def test_compose_returns_none_when_all_gates_zero():
    dummy_meta = KnowledgeMetadata(source_id="s", chunk_id=0)
    retrieved = [RetrievedKnowledge("a", "", dummy_meta, 0.0, 1.0, lora_tree=random_tree(0))]
    assert compose(retrieved, [0.0], mode="rank_concat") is None


def test_reasoning_adapter_starts_as_no_op():
    adapter = ReasoningAdapter(layer_indices=list(range(N_LAYERS)), dims={MODULE: (D_IN, D_OUT)}, rank=RANK)
    tree = adapter.lora_tree()
    delta = _delta_from_ab(tree[MODULE]["A"], tree[MODULE]["B"])
    assert torch.allclose(delta, torch.zeros_like(delta))


def test_reasoning_adapter_composes_additively_with_knowledge_eq14():
    adapter = ReasoningAdapter(layer_indices=list(range(N_LAYERS)), dims={MODULE: (D_IN, D_OUT)}, rank=RANK)
    with torch.no_grad():
        adapter.B[MODULE].add_(torch.randn_like(adapter.B[MODULE]))  # make it a non-trivial delta
    knowledge_tree = random_tree(42)
    composed = adapter.compose_with_knowledge(knowledge_tree)

    delta_know = _delta_from_ab(knowledge_tree[MODULE]["A"], knowledge_tree[MODULE]["B"])
    delta_reason = _delta_from_ab(adapter.A[MODULE].unsqueeze(0), adapter.B[MODULE].unsqueeze(0))
    delta_composed = _delta_from_ab(composed[MODULE]["A"], composed[MODULE]["B"])
    assert torch.allclose(delta_composed, delta_know + delta_reason, atol=1e-4)


def test_reasoning_adapter_alone_when_no_knowledge_retrieved():
    adapter = ReasoningAdapter(layer_indices=list(range(N_LAYERS)), dims={MODULE: (D_IN, D_OUT)}, rank=RANK)
    composed = adapter.compose_with_knowledge(None)
    assert composed[MODULE]["A"].shape == adapter.A[MODULE].unsqueeze(0).shape
