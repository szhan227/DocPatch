"""Section 3.6, last paragraph -- staged curriculum for reasoning-adapter training.

    q -> knowledge routing -> conflict resolution -> LoRA composition
      -> frozen LLM + reasoning adapter -> answer

"Training follows a staged curriculum. We first activate ground-truth
knowledge LoRAs, [...]. The training set also includes counterfactual pairs
[...]. We then train with multiple complementary ground-truth LoRAs to
improve multi-memory reasoning. Finally, we introduce examples with
versioned conflicts after metadata-based conflict resolution [...]."

Each stage adds a new example pool; earlier pools stay in the mix
(standard curriculum-learning practice) so later stages don't regress the
skills the earlier ones taught. Stage boundaries are controlled by
``ReasoningAdapterConfig.curriculum_stage_fractions``
(``docpatch/config.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ctx_to_lora.docpatch.data import DocPatchDataset

from docpatch.config import ReasoningAdapterConfig
from docpatch.data.counterfactual import CounterfactualPair, build_counterfactual_pairs

STAGE_NAMES = ("ground_truth_single", "counterfactual", "multi_source", "conflict")


@dataclass
class CurriculumStage:
    name: str
    examples: list[dict[str, Any] | CounterfactualPair]


class ReasoningCurriculum:
    """Drives which example pool(s) are sampled at a given training step."""

    def __init__(self, dataset: DocPatchDataset, config: ReasoningAdapterConfig | None = None):
        self.dataset = dataset
        self.config = config or ReasoningAdapterConfig()
        fractions = self.config.curriculum_stage_fractions
        if len(fractions) != 4 or abs(sum(fractions) - 1.0) > 1e-6:
            raise ValueError("curriculum_stage_fractions must be 4 values summing to 1.0")
        self.stages = [
            CurriculumStage("ground_truth_single", dataset.by_group.get("single_source", [])),
            CurriculumStage("counterfactual", build_counterfactual_pairs(dataset)),
            CurriculumStage("multi_source", dataset.by_group.get("multi_source", [])),
            CurriculumStage("conflict", dataset.by_group.get("conflict", [])),
        ]
        self._boundaries = []
        cumulative = 0.0
        for frac in fractions:
            cumulative += frac
            self._boundaries.append(cumulative)

    def stage_index_for_progress(self, progress: float) -> int:
        """progress in [0, 1) -> which curriculum stage has "unlocked" so far."""

        for idx, boundary in enumerate(self._boundaries):
            if progress < boundary:
                return idx
        return len(self._boundaries) - 1

    def active_pool(self, progress: float) -> list[dict[str, Any] | CounterfactualPair]:
        """Union of every stage unlocked so far (cumulative curriculum)."""

        unlocked = self.stage_index_for_progress(progress)
        pool: list[Any] = []
        for stage in self.stages[: unlocked + 1]:
            pool.extend(stage.examples)
        return pool

    def stage_counts(self) -> dict[str, int]:
        return {stage.name: len(stage.examples) for stage in self.stages}
