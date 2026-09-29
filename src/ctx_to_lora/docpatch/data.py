import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


CATEGORY_GROUPS = {
    "single_source": {"single_source_hotpot"},
    "multi_source": {
        "multi_source_consistent_hotpot",
        "multi_source_consistent_synthetic",
        "a_exclusive_synthetic",
        "b_new_synthetic",
    },
    "conflict": {"multi_source_conflict_synthetic"},
    "locality": {"unrelated_locality"},
}

DEFAULT_BATCH_MIX = {
    "single_source": 0.40,
    "multi_source": 0.25,
    "conflict": 0.25,
    "locality": 0.10,
}


@dataclass(frozen=True)
class SourceChunkRef:
    source_key: str
    source_id: str
    chunk_id: int
    text: str


class DocPatchDataset:
    """JSONL-backed dataset for DocPatch training examples."""

    def __init__(self, path: str | Path, validate: bool = True):
        self.path = Path(path)
        self.examples = self._load_jsonl(self.path)
        if validate:
            for idx, example in enumerate(self.examples):
                self.validate_example(example, idx)
        self.by_category = self._group_by_category(self.examples)
        self.by_group = self._group_by_training_group(self.by_category)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}") from exc
        return rows

    @staticmethod
    def _group_by_category(
        examples: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for example in examples:
            grouped[example["category"]].append(example)
        return dict(grouped)

    @staticmethod
    def _group_by_training_group(
        by_category: dict[str, list[dict[str, Any]]],
    ) -> dict[str, list[dict[str, Any]]]:
        by_group: dict[str, list[dict[str, Any]]] = {}
        for group, categories in CATEGORY_GROUPS.items():
            rows = []
            for category in categories:
                rows.extend(by_category.get(category, []))
            by_group[group] = rows
        return by_group

    @staticmethod
    def validate_example(example: dict[str, Any], idx: int | None = None) -> None:
        prefix = f"example {idx}: " if idx is not None else ""
        required = {
            "id",
            "category",
            "query",
            "documents",
            "ground_truth",
            "conflict_info",
            "relevance_labels",
        }
        missing = required - set(example)
        if missing:
            raise ValueError(f"{prefix}missing required fields: {sorted(missing)}")
        if not isinstance(example["documents"], dict) or not example["documents"]:
            raise ValueError(f"{prefix}documents must be a non-empty dict")
        for source_key, doc in example["documents"].items():
            for field in ("source_id", "chunks", "full_text"):
                if field not in doc:
                    raise ValueError(f"{prefix}document {source_key} missing {field}")
            if not isinstance(doc["chunks"], list) or not doc["chunks"]:
                raise ValueError(f"{prefix}document {source_key} has no chunks")
            for chunk in doc["chunks"]:
                for field in ("source_id", "chunk_id", "text"):
                    if field not in chunk:
                        raise ValueError(
                            f"{prefix}chunk in {source_key} missing {field}"
                        )
        gt = example["ground_truth"]
        for field in ("answer", "relevant_chunks"):
            if field not in gt:
                raise ValueError(f"{prefix}ground_truth missing {field}")

    def category_counts(self) -> dict[str, int]:
        return {
            category: len(rows)
            for category, rows in sorted(self.by_category.items())
        }

    def group_counts(self) -> dict[str, int]:
        return {
            group: len(rows)
            for group, rows in sorted(self.by_group.items())
        }

    @staticmethod
    def iter_chunks(example: dict[str, Any]) -> Iterator[SourceChunkRef]:
        for source_key, doc in example["documents"].items():
            source_id = doc["source_id"]
            for chunk in doc["chunks"]:
                yield SourceChunkRef(
                    source_key=source_key,
                    source_id=source_id,
                    chunk_id=int(chunk["chunk_id"]),
                    text=chunk["text"],
                )

    @staticmethod
    def relevant_chunk_refs(example: dict[str, Any]) -> list[SourceChunkRef]:
        relevant = example["ground_truth"].get("relevant_chunks", {})
        refs = []
        for source_key, chunk_ids in relevant.items():
            if source_key not in example["documents"]:
                continue
            doc = example["documents"][source_key]
            chunks_by_id = {int(chunk["chunk_id"]): chunk for chunk in doc["chunks"]}
            for chunk_id in chunk_ids:
                chunk = chunks_by_id.get(int(chunk_id))
                if chunk is None:
                    continue
                refs.append(
                    SourceChunkRef(
                        source_key=source_key,
                        source_id=doc["source_id"],
                        chunk_id=int(chunk_id),
                        text=chunk["text"],
                    )
                )
        return refs


class BalancedDocPatchSampler:
    """Sample balanced batches by DocPatch training group."""

    def __init__(
        self,
        dataset: DocPatchDataset,
        batch_size: int,
        mix: dict[str, float] | None = None,
        seed: int = 42,
        drop_empty_groups: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.mix = dict(DEFAULT_BATCH_MIX if mix is None else mix)
        self.rng = random.Random(seed)
        if drop_empty_groups:
            self.mix = {
                group: weight
                for group, weight in self.mix.items()
                if dataset.by_group.get(group)
            }
        missing = [group for group in self.mix if group not in dataset.by_group]
        if missing:
            raise ValueError(f"unknown training groups: {missing}")
        empty = [group for group in self.mix if not dataset.by_group.get(group)]
        if empty:
            raise ValueError(f"cannot sample empty training groups: {empty}")

    def sample_batch(self) -> list[dict[str, Any]]:
        counts = self._counts_for_batch()
        batch = []
        for group, count in counts.items():
            if count <= 0:
                continue
            pool = self.dataset.by_group[group]
            batch.extend(self.rng.choice(pool) for _ in range(count))
        self.rng.shuffle(batch)
        return batch

    def iter_batches(self, num_batches: int | None = None) -> Iterator[list[dict[str, Any]]]:
        n = 0
        while num_batches is None or n < num_batches:
            yield self.sample_batch()
            n += 1

    def _counts_for_batch(self) -> dict[str, int]:
        total_weight = sum(max(weight, 0.0) for weight in self.mix.values())
        if total_weight <= 0:
            raise ValueError("batch mix must have positive total weight")

        raw_counts = {
            group: self.batch_size * max(weight, 0.0) / total_weight
            for group, weight in self.mix.items()
        }
        counts = {group: int(value) for group, value in raw_counts.items()}
        remainder = self.batch_size - sum(counts.values())
        order = sorted(
            raw_counts,
            key=lambda group: raw_counts[group] - counts[group],
            reverse=True,
        )
        for group in order[:remainder]:
            counts[group] += 1
        return counts


def batch_summary(batch: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for example in batch:
        counts[example["category"]] += 1
    return dict(sorted(counts.items()))
