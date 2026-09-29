from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class ConflictAnnotation:
    source_a: str
    chunk_a: int
    source_b: str
    chunk_b: int
    conflict_type: str
    winner_source: str | None = None
    score: float | None = None
    description: str | None = None


FACT_PATTERN = re.compile(
    r"(?:the\s+)?(?P<label>[a-zA-Z_ ]+?)\s+of\s+(?P<entity>.+?)\s+is\s+(?P<value>.+?)[\.;]?$",
    re.IGNORECASE,
)


def parse_simple_fact(text: str) -> tuple[str, str, str] | None:
    text = text.strip()
    text = re.sub(r"^in version [a-z],\s+", "", text, flags=re.IGNORECASE)
    match = FACT_PATTERN.search(text)
    if not match:
        return None
    label = re.sub(r"\s+", " ", match.group("label").strip().lower())
    entity = re.sub(r"\s+", " ", match.group("entity").strip().lower())
    value = re.sub(r"\s+", " ", match.group("value").strip())
    return entity, label, value


def detect_pair_conflict(
    chunk_a: dict[str, Any],
    chunk_b: dict[str, Any],
) -> ConflictAnnotation | None:
    fact_a = parse_simple_fact(chunk_a["text"])
    fact_b = parse_simple_fact(chunk_b["text"])
    if fact_a is None or fact_b is None:
        return None
    entity_a, label_a, value_a = fact_a
    entity_b, label_b, value_b = fact_b
    if entity_a == entity_b and label_a == label_b and value_a != value_b:
        source_a = chunk_a["source_id"]
        source_b = chunk_b["source_id"]
        version_a = chunk_a.get("source_version")
        version_b = chunk_b.get("source_version")
        winner_source = None
        if version_a is not None and version_b is not None:
            winner_source = source_b if version_b >= version_a else source_a
        return ConflictAnnotation(
            source_a=source_a,
            chunk_a=int(chunk_a["chunk_id"]),
            source_b=source_b,
            chunk_b=int(chunk_b["chunk_id"]),
            conflict_type="entity_attribute_update",
            winner_source=winner_source,
            score=1.0,
            description=f"{entity_a}/{label_a}: {value_a} -> {value_b}",
        )
    return None


def detect_conflicts_between_docs(
    doc_a: dict[str, Any],
    doc_b: dict[str, Any],
) -> list[ConflictAnnotation]:
    annotations = []
    chunks_a = [
        {**chunk, "source_version": doc_a.get("source_version")}
        for chunk in doc_a.get("chunks", [])
    ]
    chunks_b = [
        {**chunk, "source_version": doc_b.get("source_version")}
        for chunk in doc_b.get("chunks", [])
    ]
    for chunk_a in chunks_a:
        for chunk_b in chunks_b:
            annotation = detect_pair_conflict(chunk_a, chunk_b)
            if annotation is not None:
                annotations.append(annotation)
    return annotations


def annotations_from_example(example: dict[str, Any]) -> list[ConflictAnnotation]:
    conflict_info = example.get("conflict_info", {})
    if conflict_info.get("has_conflict") and "conflicting_chunks" in conflict_info:
        chunks = conflict_info["conflicting_chunks"]
        source_keys = sorted(chunks)
        if len(source_keys) >= 2:
            left, right = source_keys[:2]
            return [
                ConflictAnnotation(
                    source_a=left,
                    chunk_a=int(chunk_a),
                    source_b=right,
                    chunk_b=int(chunk_b),
                    conflict_type=conflict_info.get("conflict_type", "unknown"),
                    winner_source=conflict_info.get("winner_source"),
                    description=conflict_info.get("conflict_description"),
                )
                for chunk_a in chunks[left]
                for chunk_b in chunks[right]
            ]

    docs = example.get("documents", {})
    if "A" in docs and "B" in docs:
        return detect_conflicts_between_docs(docs["A"], docs["B"])
    return []


def mark_conflicts(example: dict[str, Any]) -> dict[str, Any]:
    annotations = annotations_from_example(example)
    if not annotations:
        example["conflict_info"] = {"has_conflict": False}
        return example
    first = annotations[0]
    example["conflict_info"] = {
        "has_conflict": True,
        "conflicting_chunks": {
            first.source_a: [ann.chunk_a for ann in annotations],
            first.source_b: [ann.chunk_b for ann in annotations],
        },
        "conflict_type": first.conflict_type,
        "winner_source": first.winner_source,
        "annotations": [ann.__dict__ for ann in annotations],
    }
    return example
