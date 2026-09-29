"""Section 3.4 -- Conflict-aware knowledge resolution.

Implements Eq. 9-10 exactly:

    m_i = 0  if a higher-precedence retrieved LoRA belongs to the same
             conflict group,
          1  otherwise.

    g_i = alpha_i * m_i / sum_{j in S_q} alpha_j * m_j

Only the retrieved set S_q is touched -- nothing is deleted from the
knowledge bank (Section 3.4, first paragraph: "Conflict resolution is applied
only to the retrieved set S_q; all LoRAs remain stored in the knowledge
bank"). Precedence is version-then-source-id (see
``docpatch.metadata.KnowledgeMetadata.precedence_key``); higher version wins,
ties broken deterministically. LoRAs with no conflict-group id never compete
with anything (m_i = 1 unconditionally), matching non-conflicting knowledge
staying available across versions (paper section 2, "new knowledge is added
rather than overwriting existing knowledge").
"""

from __future__ import annotations

from docpatch.config import ConflictResolutionConfig
from docpatch.value_types import RetrievedKnowledge


def compute_applicability_mask(
    retrieved: list[RetrievedKnowledge],
    config: ConflictResolutionConfig | None = None,
) -> list[int]:
    """Return m_i for each retrieved item (Eq. 9)."""

    config = config or ConflictResolutionConfig()
    # Highest precedence per conflict group among the retrieved set.
    best_precedence: dict[str, tuple[int, str]] = {}
    for item in retrieved:
        group = item.metadata.conflict_group
        if group is None:
            continue
        key = item.metadata.precedence_key
        if group not in best_precedence or key > best_precedence[group]:
            best_precedence[group] = key

    mask = []
    for item in retrieved:
        group = item.metadata.conflict_group
        if group is None:
            mask.append(1)
            continue
        is_highest = item.metadata.precedence_key == best_precedence[group]
        mask.append(1 if is_highest else 0)
    return mask


def resolve_conflicts(
    retrieved: list[RetrievedKnowledge],
    config: ConflictResolutionConfig | None = None,
) -> list[float]:
    """Return the final composition weight g_i for each retrieved item (Eq. 10)."""

    if not retrieved:
        return []
    mask = compute_applicability_mask(retrieved, config)
    weighted = [item.alpha * m for item, m in zip(retrieved, mask)]
    denom = sum(weighted)
    if denom <= 0.0:
        # Degenerate case: everything got suppressed (e.g. all-zero router
        # weights). Fall back to a uniform split over applicable items so the
        # pipeline still produces a well-defined composition.
        applicable = [i for i, m in enumerate(mask) if m == 1]
        if not applicable:
            return [0.0] * len(retrieved)
        share = 1.0 / len(applicable)
        return [share if m == 1 else 0.0 for m in mask]
    return [w / denom for w in weighted]


def active_indices(gates: list[float]) -> list[int]:
    """Indices with strictly positive composition weight (S_q^+ in Appendix A.1)."""

    return [idx for idx, g in enumerate(gates) if g > 0.0]
