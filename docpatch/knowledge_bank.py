"""Section 3.2 -- offline knowledge bank.

    L_i = H(c_i) = {(A_i^(l), B_i^(l))}_{l in T}          (Eq. 1)
    M = {(L_i, z_i, mu_i)}_{i=1}^N                          (knowledge bank)

H is the frozen, pretrained Doc-to-LoRA hypernetwork
(``ctx_to_lora.modeling.hypernet.ModulatedPretrainedModel``). z_i is the
Appendix A.3 sketch feature (pre-projector; the learned P_LoRA that finishes
mapping it into the routing space is trained jointly with the router, see
``docpatch/router.py`` and ``docpatch/training/train_router.py``). mu_i is
source/chunk id/version/conflict-group metadata (``docpatch/metadata.py``).

"Existing LoRAs are retained rather than overwritten. [...] DocPatch
maintains explicit parametric snapshots of knowledge across document
versions" (Section 3.2, last paragraph) -- ``KnowledgeBank.add_document``
therefore only ever appends new records; nothing already on disk is deleted
or mutated when a new document version arrives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from ctx_to_lora.docpatch.cache import chunk_cache_key, collect_unique_chunks, tensor_tree_to_cpu, text_hash
from ctx_to_lora.docpatch.chunking import chunk_document
from ctx_to_lora.docpatch.data import DocPatchDataset, SourceChunkRef
from docpatch.lora_sketch import FixedRandomProjections, compute_lora_sketch
from docpatch.metadata import KnowledgeMetadata, build_metadata_for_ref


@dataclass
class BankRecord:
    """One knowledge LoRA in a bank: (L_i, z_i, mu_i).

    A bank is just a ``list[BankRecord]``. The LoRA is either held in memory
    (``lora_tree``, e.g. from ``text_to_lora_record``) or cached on disk
    (``lora_path``, e.g. from ``KnowledgeBank``). ``sketch`` is the Appendix A.3
    feature z_i the router scores against.
    """

    cache_key: str
    source_id: str
    chunk_id: int
    version: int
    text: str
    conflict_group: str | None
    lora_path: str
    text_hash: str
    example_ids: list[str] | None = None
    lora_tree: dict | None = field(default=None, repr=False)
    sketch: Tensor | None = field(default=None, repr=False)


    @property
    def metadata(self) -> KnowledgeMetadata:
        return KnowledgeMetadata(
            source_id=self.source_id,
            chunk_id=self.chunk_id,
            version=self.version,
            conflict_group=self.conflict_group,
        )


def load_d2l_model(checkpoint_path: str | Path, use_flash_attn: bool = False):
    """Load the frozen, pretrained Doc-to-LoRA hypernetwork H (Eq. 1)."""

    from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict, train=False, use_sequence_packing=False, use_flash_attn=use_flash_attn
    )
    model.eval()
    model.reset()
    return model


def chunk_text_to_ids(text: str, tokenizer, device: torch.device) -> Tensor:
    tokenized = tokenizer.apply_chat_template(
        [[{"role": "user", "content": text.strip()}]],
        tokenize=True,
        add_generation_prompt=True,
        return_attention_mask=False,
        padding=False,
        truncation=False,
        add_special_tokens=False,
        return_dict=True,
    )
    return torch.tensor(tokenized["input_ids"], device=device)


def text_to_lora(model, tokenizer, text: str) -> dict:
    """Map one text to a knowledge LoRA with the hypernetwork (Eq. 1).

    Returns {module: {"A": [1, n_layers, r, d_in], "B": [1, n_layers, r, d_out]}}
    as ordinary CPU tensors.
    """

    with torch.inference_mode():
        ctx_ids = chunk_text_to_ids(text, tokenizer, model.device)
        lora_weights, _layernorm_weights = model.generate_weights(ctx_ids, torch.ones_like(ctx_ids))
    tree = tensor_tree_to_cpu(lora_weights)
    return {m: {k: t.clone() for k, t in weights.items()} for m, weights in tree.items()}  # non-inference tensors


def text_to_lora_record(
    model,
    tokenizer,
    text: str,
    source_id: str,
    version: int = 0,
    conflict_group: str | None = None,
    chunk_id: int = 0,
    sketch_dim: int = 64,
    seed: int = 0,
) -> BankRecord:
    """text -> LoRA (hypernetwork) -> ``BankRecord`` ready to append to a bank list.

    ``sketch_dim``/``seed`` must match ``config.knowledge_bank.sketch_dim`` (the
    router's input size is derived from it).
    """

    ref = SourceChunkRef(source_key=source_id, source_id=source_id, chunk_id=chunk_id, text=text)
    lora_tree = text_to_lora(model, tokenizer, text)
    sketch = compute_lora_sketch(lora_tree, FixedRandomProjections(sketch_dim=sketch_dim, seed=seed)).float()
    return BankRecord(
        cache_key=chunk_cache_key(ref),
        source_id=source_id,
        chunk_id=chunk_id,
        version=version,
        text=text,
        conflict_group=conflict_group,
        lora_path="",
        text_hash=text_hash(text),
        lora_tree=lora_tree,
        sketch=sketch,
    )


class KnowledgeBank:
    """Persistent, append-only on-disk bank (used by the build/train/eval scripts).

    For a quick in-memory bank, use a plain list of ``text_to_lora_record`` results.
    Its ``records`` list can be passed straight to ``DocPatchPipeline.answer``.
    """

    def __init__(
        self,
        output_dir: str | Path,
        sketch_dim: int = 64,
        seed: int = 0,
    ):
        self.output_dir = Path(output_dir)
        self.lora_dir = self.output_dir / "loras"
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "manifest.jsonl"
        self.features_path = self.output_dir / "sketch_features.pt"
        self.projections = FixedRandomProjections(sketch_dim=sketch_dim, seed=seed)
        self.records: list[BankRecord] = []
        self._features: list[Tensor] = []
        if self.manifest_path.exists():
            self._load_manifest()

    def __len__(self) -> int:
        return len(self.records)

    @property
    def feature_dim(self) -> int:
        if not self._features:
            raise ValueError("knowledge bank is empty; add something before reading feature_dim")
        return self._features[0].shape[-1]

    def _load_manifest(self) -> None:
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(BankRecord(**json.loads(line)))
        if self.features_path.exists():
            payload = torch.load(self.features_path, map_location="cpu", weights_only=False)
            self._features = list(payload["features"].unbind(0))
            for record, feature in zip(self.records, self._features):
                record.sketch = feature  # so bank.records can be passed straight to the pipeline

    def _persist(self) -> None:
        with self.manifest_path.open("w", encoding="utf-8") as f:
            for record in self.records:
                row = {
                    fl.name: getattr(record, fl.name)
                    for fl in fields(record)
                    if fl.name not in ("lora_tree", "sketch")
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self._features:
            torch.save({"features": torch.stack(self._features, dim=0)}, self.features_path)

    def by_cache_key(self) -> dict[str, BankRecord]:
        return {r.cache_key: r for r in self.records}

    def sketch_feature(self, cache_key: str) -> Tensor:
        index = {r.cache_key: i for i, r in enumerate(self.records)}[cache_key]
        return self._features[index]

    def stacked_features(self, records: list[BankRecord]) -> Tensor:
        """[len(records), feature_dim] sketch features, in the given record order."""

        index = {r.cache_key: i for i, r in enumerate(self.records)}
        return torch.stack([self._features[index[r.cache_key]] for r in records], dim=0)

    def records_for_source(self, source_id: str) -> list[BankRecord]:
        return [r for r in self.records if r.source_id == source_id]

    def _add_record(
        self,
        model,
        tokenizer,
        source_key: str,
        source_id: str,
        chunk_id: int,
        text: str,
        version: int,
        conflict_group: str | None,
        example_ids: list[str] | None,
        overwrite: bool,
    ) -> BankRecord:
        ref = SourceChunkRef(source_key=source_key, source_id=source_id, chunk_id=chunk_id, text=text)
        cache_key = chunk_cache_key(ref)
        lora_path = self.lora_dir / f"{cache_key}.pt"
        existing = self.by_cache_key().get(cache_key)
        if existing is not None and not overwrite:
            return existing

        lora_weights_cpu = text_to_lora(model, tokenizer, text)
        sketch = compute_lora_sketch(lora_weights_cpu, self.projections).float()
        torch.save(
            {"cache_key": cache_key, "source_id": source_id, "chunk_id": chunk_id, "lora_weights": lora_weights_cpu},
            lora_path,
        )
        record = BankRecord(
            cache_key=cache_key,
            source_id=source_id,
            chunk_id=chunk_id,
            version=version,
            conflict_group=conflict_group,
            lora_path=str(lora_path),
            text_hash=text_hash(text),
            example_ids=example_ids,
            sketch=sketch,
        )
        if existing is None:
            self.records.append(record)
            self._features.append(sketch)
        else:
            idx = self.records.index(existing)
            self.records[idx] = record
            self._features[idx] = sketch
        return record

    def add_document(
        self,
        model,
        tokenizer,
        source_id: str,
        text: str,
        version: int = 0,
        chunk_size: int = 512,
        overlap: int = 128,
        conflict_group_fn: Callable[[str], str | None] | None = None,
        overwrite: bool = False,
    ) -> list[BankRecord]:
        """Add (or append a new version of) a document to the bank.

        This is the general-purpose entry point for continual updating
        (Section 4.5): calling it again with a higher ``version`` for the
        same conceptual document appends new knowledge LoRAs without
        touching previously stored ones. ``conflict_group_fn`` maps a chunk's
        text to a conflict-group id shared by earlier alternatives of the
        same fact (Section 3.4 / Appendix A.5); leave it None for chunks that
        never conflict with anything.
        """

        chunks = chunk_document(text, chunk_size=chunk_size, overlap=overlap)
        records = []
        for chunk in chunks:
            group = conflict_group_fn(chunk.text) if conflict_group_fn else None
            record = self._add_record(
                model,
                tokenizer,
                source_key=source_id,
                source_id=source_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                version=version,
                conflict_group=group,
                example_ids=None,
                overwrite=overwrite,
            )
            records.append(record)
        self._persist()
        return records

    def build_from_dataset(
        self,
        model,
        tokenizer,
        dataset: DocPatchDataset,
        max_chunks: int = 0,
        overwrite: bool = False,
    ) -> list[BankRecord]:
        """Batch-build the bank from a ``DocPatchDataset`` (Section 4.1 setup:
        HotpotQA / Constitution-QA-derived + synthetic-conflict chunks)."""

        unique = collect_unique_chunks(dataset)
        items = list(unique.items())
        if max_chunks > 0:
            items = items[:max_chunks]
        records = []
        for cache_key, payload in items:
            ref: SourceChunkRef = payload["ref"]
            example = next(ex for ex in dataset.examples if ex["id"] == payload["example_ids"][0])
            metadata = build_metadata_for_ref(example, ref)
            record = self._add_record(
                model,
                tokenizer,
                source_key=ref.source_key,
                source_id=ref.source_id,
                chunk_id=ref.chunk_id,
                text=ref.text,
                version=metadata.version,
                conflict_group=metadata.conflict_group,
                example_ids=payload["example_ids"],
                overwrite=overwrite,
            )
            records.append(record)
        self._persist()
        return records
