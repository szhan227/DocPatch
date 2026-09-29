from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ctx_to_lora.docpatch.cache import ChunkLoraRecord, load_manifest
from ctx_to_lora.docpatch.data import SourceChunkRef


@dataclass
class SourceRecord:
    source_key: str
    source_id: str
    chunks: dict[int, dict[str, Any]] = field(default_factory=dict)
    source_version: int | None = None
    full_text: str | None = None


class ChunkStore:
    """In-memory organization for multi-source DocPatch chunks."""

    def __init__(self):
        self.sources: dict[str, SourceRecord] = {}
        self.cache_records: dict[str, ChunkLoraRecord] = {}
        self.cache_by_source_chunk: dict[tuple[str, int], ChunkLoraRecord] = {}

    @classmethod
    def from_examples(cls, examples: list[dict[str, Any]]) -> "ChunkStore":
        store = cls()
        for example in examples:
            store.add_example(example)
        return store

    @classmethod
    def from_manifest(cls, manifest_path: str | Path) -> "ChunkStore":
        store = cls()
        for record in load_manifest(Path(manifest_path)):
            store.add_cache_record(record)
        return store

    def add_example(self, example: dict[str, Any]) -> None:
        for source_key, doc in example["documents"].items():
            source = self.sources.setdefault(
                source_key,
                SourceRecord(
                    source_key=source_key,
                    source_id=doc["source_id"],
                    source_version=doc.get("source_version"),
                    full_text=doc.get("full_text"),
                ),
            )
            for chunk in doc["chunks"]:
                source.chunks[int(chunk["chunk_id"])] = chunk

    def add_cache_record(self, record: ChunkLoraRecord) -> None:
        self.cache_records[record.cache_key] = record
        self.cache_by_source_chunk[(record.source_id, record.chunk_id)] = record
        source = self.sources.setdefault(
            record.source_key,
            SourceRecord(
                source_key=record.source_key,
                source_id=record.source_id,
                source_version=record.source_version,
            ),
        )
        source.chunks[record.chunk_id] = {
            "source_id": record.source_id,
            "chunk_id": record.chunk_id,
            "text": record.text,
            "text_hash": record.text_hash,
            "lora_path": record.lora_path,
            "feature_path": record.feature_path,
        }

    def get_source_chunks(self, source_key: str) -> list[dict[str, Any]]:
        source = self.sources[source_key]
        return [source.chunks[idx] for idx in sorted(source.chunks)]

    def iter_chunk_refs(self) -> list[SourceChunkRef]:
        refs = []
        for source_key, source in self.sources.items():
            for chunk_id, chunk in sorted(source.chunks.items()):
                refs.append(
                    SourceChunkRef(
                        source_key=source_key,
                        source_id=source.source_id,
                        chunk_id=chunk_id,
                        text=chunk.get("text") or "",
                    )
                )
        return refs

    def source_counts(self) -> dict[str, int]:
        return {
            source_key: len(source.chunks)
            for source_key, source in sorted(self.sources.items())
        }

    def cache_paths_by_source(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for record in self.cache_records.values():
            out[record.source_id].append(record.lora_path)
        return {key: sorted(paths) for key, paths in out.items()}
