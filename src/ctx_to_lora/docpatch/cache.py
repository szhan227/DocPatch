import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ctx_to_lora.docpatch.data import DocPatchDataset, SourceChunkRef


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def chunk_cache_key(ref: SourceChunkRef) -> str:
    return f"{ref.source_id}__chunk_{ref.chunk_id}__{text_hash(ref.text)[:16]}"


@dataclass(frozen=True)
class ChunkLoraRecord:
    cache_key: str
    source_key: str
    source_id: str
    chunk_id: int
    text_hash: str
    lora_path: str
    example_ids: list[str]
    source_version: int | None = None
    feature_path: str | None = None
    text: str | None = None


def collect_unique_chunks(dataset: DocPatchDataset) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    for example in dataset.examples:
        for ref in dataset.iter_chunks(example):
            key = chunk_cache_key(ref)
            source_doc = example["documents"][ref.source_key]
            if key not in chunks:
                chunks[key] = {
                    "ref": ref,
                    "source_version": source_doc.get("source_version"),
                    "example_ids": [],
                }
            chunks[key]["example_ids"].append(example["id"])
    return chunks


def tensor_tree_to_cpu(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: tensor_tree_to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [tensor_tree_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(tensor_tree_to_cpu(value) for value in obj)
    return obj


def save_manifest(path: Path, records: list[ChunkLoraRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> list[ChunkLoraRecord]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                payload = json.loads(line)
                records.append(ChunkLoraRecord(**payload))
    return records
