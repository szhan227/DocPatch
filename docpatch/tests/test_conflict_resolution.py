"""Unit tests for Section 3.4 / Eq. 9-10 (docpatch/conflict_resolution.py)."""

from docpatch.conflict_resolution import active_indices, compute_applicability_mask, resolve_conflicts
from docpatch.metadata import KnowledgeMetadata
from docpatch.value_types import RetrievedKnowledge


def make(source_id, chunk_id, version, group, alpha, score=0.0):
    return RetrievedKnowledge(
        cache_key=f"{source_id}_{chunk_id}",
        lora_path="",
        metadata=KnowledgeMetadata(source_id=source_id, chunk_id=chunk_id, version=version, conflict_group=group),
        router_score=score,
        alpha=alpha,
    )


def test_no_conflict_groups_pass_through_normalized():
    retrieved = [make("A", 0, 0, None, 0.6), make("B", 0, 0, None, 0.4)]
    gates = resolve_conflicts(retrieved)
    assert gates == [0.6, 0.4]


def test_newer_version_wins_within_conflict_group():
    old = make("A", 0, version=1, group="fact_x", alpha=0.5)
    new = make("B", 0, version=2, group="fact_x", alpha=0.3)
    gates = resolve_conflicts([old, new])
    # Old is suppressed (m_i = 0); new gets the entire renormalized weight.
    assert gates[0] == 0.0
    assert gates[1] == 1.0


def test_mask_matches_paper_eq9():
    old = make("A", 0, version=1, group="fact_x", alpha=0.5)
    new = make("B", 0, version=2, group="fact_x", alpha=0.3)
    unrelated = make("C", 0, version=0, group=None, alpha=0.2)
    mask = compute_applicability_mask([old, new, unrelated])
    assert mask == [0, 1, 1]


def test_compatible_groups_both_contribute():
    a = make("A", 0, version=1, group="fact_x", alpha=0.5)
    b = make("B", 0, version=1, group="fact_y", alpha=0.3)
    gates = resolve_conflicts([a, b])
    total = sum(gates)
    assert abs(total - 1.0) < 1e-9
    # Both are highest precedence within their own distinct groups.
    assert all(g > 0 for g in gates)


def test_ties_broken_deterministically_by_source_id():
    a = make("A", 0, version=1, group="fact_x", alpha=0.5)
    b = make("Z", 0, version=1, group="fact_x", alpha=0.5)
    gates = resolve_conflicts([a, b])
    # precedence_key = (version, source_id); "Z" > "A" lexicographically.
    assert gates == [0.0, 1.0]


def test_lone_group_member_is_never_suppressed_even_with_zero_alpha():
    only = make("A", 0, version=1, group="fact_x", alpha=0.0)
    gates = resolve_conflicts([only])
    assert gates == [1.0]


def test_active_indices():
    gates = [0.0, 0.7, 0.3, 0.0]
    assert active_indices(gates) == [1, 2]
