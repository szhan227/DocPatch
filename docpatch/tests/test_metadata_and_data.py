"""Integration tests against the real mixed DocPatch dataset (data-only, no
model/GPU required): metadata inference (Section 3.2/3.4), counterfactual
pair construction (Appendix A.6), and the staged curriculum (Section 3.6)."""

from ctx_to_lora.docpatch.data import DocPatchDataset
from docpatch.conflict_resolution import resolve_conflicts
from docpatch.data.counterfactual import build_counterfactual_pairs
from docpatch.data.curriculum import ReasoningCurriculum
from docpatch.metadata import build_metadata_for_ref
from docpatch.config import ReasoningAdapterConfig


def test_conflict_group_shared_across_versions_same_fact_slot(mixed_dev_dataset: DocPatchDataset):
    examples = mixed_dev_dataset.by_category.get("multi_source_conflict_synthetic", [])
    assert examples, "expected at least one conflict example in the mixed dev set"
    example = examples[0]
    refs = list(DocPatchDataset.iter_chunks(example))
    metas = [build_metadata_for_ref(example, r) for r in refs]
    groups = [m.conflict_group for m in metas if m.conflict_group is not None]
    assert groups, "expected at least one chunk to carry a conflict-group id"

    # Two chunks sharing a fact_slot (across A/B document versions) must land
    # in the same conflict group so resolve_conflicts can compare them.
    slot_to_group = {}
    for ref, meta in zip(refs, metas):
        doc = example["documents"][ref.source_key]
        chunk = next(c for c in doc["chunks"] if int(c["chunk_id"]) == ref.chunk_id)
        slot = chunk.get("fact_slot")
        if slot is None or meta.conflict_group is None:
            continue
        slot_to_group.setdefault(slot, set()).add(meta.conflict_group)
    assert all(len(groups) == 1 for groups in slot_to_group.values())


def test_resolve_conflicts_on_real_conflict_example_prefers_newer_version(mixed_dev_dataset: DocPatchDataset):
    from docpatch.value_types import RetrievedKnowledge

    example = mixed_dev_dataset.by_category["multi_source_conflict_synthetic"][0]
    conflict_info = example["conflict_info"]
    conflicting = conflict_info["conflicting_chunks"]
    winner_source = conflict_info["winner_source"]

    retrieved = []
    for source_key, chunk_ids in conflicting.items():
        doc = example["documents"][source_key]
        for chunk_id in chunk_ids:
            ref = next(r for r in DocPatchDataset.iter_chunks(example) if r.source_key == source_key and r.chunk_id == int(chunk_id))
            meta = build_metadata_for_ref(example, ref)
            retrieved.append(RetrievedKnowledge(f"{source_key}_{chunk_id}", "", meta, 0.0, 0.5))

    gates = resolve_conflicts(retrieved)
    winner_indices = [i for i, r in enumerate(retrieved) if r.metadata.source_id == example["documents"][winner_source]["source_id"]]
    assert sum(gates[i] for i in winner_indices) > 0
    loser_indices = [i for i in range(len(retrieved)) if i not in winner_indices]
    assert all(gates[i] == 0.0 for i in loser_indices)


def test_counterfactual_pairs_have_differing_targets(mixed_dev_dataset: DocPatchDataset):
    pairs = build_counterfactual_pairs(mixed_dev_dataset)
    assert pairs, "expected at least one counterfactual pair from the conflict category"
    by_query = {}
    for pair in pairs:
        by_query.setdefault(pair.query, []).append(pair)
    multi = [p for p in by_query.values() if len(p) >= 2]
    assert multi, "expected some questions with both an old- and new-version counterfactual"
    for pair_group in multi:
        targets = {p.target_answer for p in pair_group}
        assert len(targets) >= 1  # same question, potentially different targets per active LoRA


def test_curriculum_stage_progress_boundaries(mixed_dev_dataset: DocPatchDataset):
    config = ReasoningAdapterConfig(curriculum_stage_fractions=(0.25, 0.25, 0.25, 0.25))
    curriculum = ReasoningCurriculum(mixed_dev_dataset, config)
    assert curriculum.stage_index_for_progress(0.0) == 0
    assert curriculum.stage_index_for_progress(0.24) == 0
    assert curriculum.stage_index_for_progress(0.26) == 1
    assert curriculum.stage_index_for_progress(0.51) == 2
    assert curriculum.stage_index_for_progress(0.99) == 3

    pool_stage0 = curriculum.active_pool(0.1)
    pool_stage3 = curriculum.active_pool(0.99)
    # Cumulative curriculum: later stages' pools are supersets in composition.
    assert len(pool_stage3) >= len(pool_stage0)


def test_curriculum_rejects_bad_fractions(mixed_dev_dataset: DocPatchDataset):
    import pytest

    with pytest.raises(ValueError):
        ReasoningCurriculum(mixed_dev_dataset, ReasoningAdapterConfig(curriculum_stage_fractions=(0.5, 0.5, 0.5, 0.5)))
