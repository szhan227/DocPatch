from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from ctx_to_lora.docpatch.cache import ChunkLoraRecord, load_manifest
from ctx_to_lora.docpatch.cache import text_hash
from ctx_to_lora.docpatch.conflict_detection import annotations_from_example
from ctx_to_lora.docpatch.data import (
    BalancedDocPatchSampler,
    DocPatchDataset,
)
from ctx_to_lora.docpatch.losses import (
    compute_conflict_suppression_loss,
    compute_source_selection_loss,
    compute_sparsity_loss,
    compute_total_loss,
)
from ctx_to_lora.docpatch.lora_features import LoraFeatureProjector, LoraFeatureStore
from ctx_to_lora.docpatch.router import SourceAwareRouter


class TextEmbedder(Protocol):
    def encode(self, texts: list[str], max_length: int = 512) -> Tensor:
        ...


@dataclass
class DocPatchTrainingConfig:
    train_path: Path = Path("data_preprocess/outputs/mixed/docpatch_train.jsonl")
    dev_path: Path = Path("data_preprocess/outputs/mixed/docpatch_dev.jsonl")
    lora_manifest_path: Path | None = None
    lora_feature_path: Path | None = None
    allow_text_candidate_baseline: bool = False
    batch_size: int = 32
    hidden_size: int = 768
    router_hidden_size: int = 512
    max_sources: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "task": 0.0,
            "source": 0.5,
            "conflict": 0.3,
            "locality": 0.0,
            "sparsity": 0.1,
        }
    )


@dataclass
class RouterBatch:
    examples: list[dict[str, Any]]
    queries: list[str]
    candidate_texts: list[list[str]]
    candidate_cache_keys: list[list[str]]
    source_keys: list[list[str]]
    source_ids: Tensor
    chunk_mask: Tensor
    target_source_ids: Tensor
    source_loss_mask: Tensor
    conflict_pairs: list[list[tuple[int, int, int]]]


class HashTextEmbedder(nn.Module):
    """Deterministic query embedder and explicit text-candidate baseline.

    Use this for query embeddings or controlled text-router baselines. The
    privacy-preserving DocPatch path should use LoRA features for candidates.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def encode(self, texts: list[str], max_length: int = 512) -> Tensor:
        vectors = []
        for text in texts:
            vec = torch.zeros(self.hidden_size, dtype=torch.float32)
            for token in text.lower().split()[:max_length]:
                idx = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % self.hidden_size
                vec[idx] += 1.0
            norm = vec.norm().clamp_min(1.0)
            vectors.append(vec / norm)
        return torch.stack(vectors, dim=0)


class DocPatchTrainer:
    def __init__(
        self,
        config: DocPatchTrainingConfig,
        query_embedder: TextEmbedder | None = None,
        candidate_embedder: TextEmbedder | None = None,
        router: SourceAwareRouter | None = None,
    ):
        self.config = config
        self.train_dataset = DocPatchDataset(config.train_path)
        self.dev_dataset = DocPatchDataset(config.dev_path)
        self.sampler = BalancedDocPatchSampler(
            self.train_dataset,
            batch_size=config.batch_size,
        )
        self.query_embedder = query_embedder or HashTextEmbedder(config.hidden_size)
        self.candidate_embedder = candidate_embedder
        self.lora_feature_store = (
            LoraFeatureStore.load(config.lora_feature_path)
            if config.lora_feature_path is not None
            else None
        )
        self.lora_feature_projector = (
            LoraFeatureProjector(
                input_size=self.lora_feature_store.features.shape[-1],
                hidden_size=config.hidden_size,
            )
            if self.lora_feature_store is not None
            else None
        )
        if (
            self.lora_feature_store is None
            and not config.allow_text_candidate_baseline
        ):
            raise ValueError(
                "DocPatchTrainer now expects --lora_feature_path for LoRA-feature "
                "routing. Set allow_text_candidate_baseline=True only for ablations."
            )
        if self.lora_feature_store is None and self.candidate_embedder is None:
            self.candidate_embedder = self.query_embedder
        self.router = router or SourceAwareRouter(
            hidden_size=config.hidden_size,
            router_hidden_size=config.router_hidden_size,
            num_sources=config.max_sources,
        )
        self.lora_records = (
            load_manifest(config.lora_manifest_path)
            if config.lora_manifest_path is not None
            else []
        )

    def build_optimizer(self) -> torch.optim.Optimizer:
        modules = [self.router]
        if self.lora_feature_projector is not None:
            modules.append(self.lora_feature_projector)
        params = [param for module in modules for param in module.parameters()]
        return torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def make_router_batch(self, examples: list[dict[str, Any]]) -> RouterBatch:
        max_chunks = max(
            sum(len(doc["chunks"]) for doc in ex["documents"].values())
            for ex in examples
        )
        source_vocab = self._batch_source_vocab(examples)
        source_ids = torch.zeros(len(examples), max_chunks, dtype=torch.long)
        chunk_mask = torch.zeros(len(examples), max_chunks, dtype=torch.bool)
        target_source_ids = torch.zeros(len(examples), dtype=torch.long)
        source_loss_mask = torch.ones(len(examples), dtype=torch.bool)
        queries = []
        candidate_texts: list[list[str]] = []
        candidate_cache_keys: list[list[str]] = []
        source_keys_by_example: list[list[str]] = []
        conflict_pairs: list[list[tuple[int, int, int]]] = []

        for batch_idx, ex in enumerate(examples):
            queries.append(ex["query"])
            flat_candidate_texts = []
            flat_cache_keys = []
            flat_source_keys = []
            source_chunk_to_flat: dict[tuple[str, int], int] = {}
            cursor = 0
            for source_key, doc in ex["documents"].items():
                for chunk in doc["chunks"]:
                    flat_candidate_texts.append(chunk["text"])
                    flat_cache_keys.append(self._cache_key_for_chunk(doc, chunk))
                    flat_source_keys.append(source_key)
                    source_ids[batch_idx, cursor] = source_vocab[source_key]
                    chunk_mask[batch_idx, cursor] = True
                    source_chunk_to_flat[(source_key, int(chunk["chunk_id"]))] = cursor
                    cursor += 1
            candidate_texts.append(flat_candidate_texts)
            candidate_cache_keys.append(flat_cache_keys)
            source_keys_by_example.append(flat_source_keys)
            target_id = self._target_source_id(ex, source_vocab)
            if target_id is None:
                source_loss_mask[batch_idx] = False
                target_id = 0
            target_source_ids[batch_idx] = target_id
            conflict_pairs.append(
                self._conflict_pairs_for_example(ex, source_chunk_to_flat)
            )

        return RouterBatch(
            examples=examples,
            queries=queries,
            candidate_texts=candidate_texts,
            candidate_cache_keys=candidate_cache_keys,
            source_keys=source_keys_by_example,
            source_ids=source_ids,
            chunk_mask=chunk_mask,
            target_source_ids=target_source_ids,
            source_loss_mask=source_loss_mask,
            conflict_pairs=conflict_pairs,
        )

    def router_forward(self, batch: RouterBatch):
        query_embeds = self.query_embedder.encode(batch.queries)
        if self.lora_feature_store is not None:
            flat_features = [
                self.lora_feature_store.get(cache_key)
                for per_example_keys in batch.candidate_cache_keys
                for cache_key in per_example_keys
            ]
            flat_candidate_embeds = torch.stack(flat_features, dim=0)
            flat_candidate_embeds = self.lora_feature_projector(flat_candidate_embeds)
        else:
            flat_candidate_texts = [
                text
                for per_example_texts in batch.candidate_texts
                for text in per_example_texts
            ]
            flat_candidate_embeds = self.candidate_embedder.encode(flat_candidate_texts)
        candidate_embeds = torch.zeros(
            batch.source_ids.shape[0],
            batch.source_ids.shape[1],
            flat_candidate_embeds.shape[-1],
            dtype=flat_candidate_embeds.dtype,
            device=flat_candidate_embeds.device,
        )
        cursor = 0
        for batch_idx, per_example_keys in enumerate(batch.candidate_cache_keys):
            n = len(per_example_keys)
            candidate_embeds[batch_idx, :n] = flat_candidate_embeds[cursor : cursor + n]
            cursor += n
        return self.router(
            query_embeds.to(candidate_embeds.device),
            candidate_embeds,
            batch.source_ids.to(candidate_embeds.device),
            batch.chunk_mask.to(candidate_embeds.device),
        )

    def router_loss_step(self, examples: list[dict[str, Any]]) -> dict[str, Tensor]:
        batch = self.make_router_batch(examples)
        output = self.router_forward(batch)
        source_loss_mask = batch.source_loss_mask.to(output.source_logits.device)
        if source_loss_mask.any():
            source_loss = compute_source_selection_loss(
                output.source_logits[source_loss_mask],
                batch.target_source_ids.to(output.source_logits.device)[source_loss_mask],
            )
        else:
            source_loss = output.source_logits.new_tensor(0.0)
        conflict_loss = compute_conflict_suppression_loss(
            output.chunk_gates,
            batch.conflict_pairs,
        )
        sparsity_loss = compute_sparsity_loss(output.chunk_gates)
        zero = source_loss.new_tensor(0.0)
        total_loss = compute_total_loss(
            task_loss=zero,
            source_loss=source_loss,
            conflict_loss=conflict_loss,
            locality_loss=zero,
            sparsity_loss=sparsity_loss,
            weights=self.config.loss_weights,
        )
        return {
            "total": total_loss,
            "source": source_loss,
            "conflict": conflict_loss,
            "sparsity": sparsity_loss,
        }

    def train_router_step(self, optimizer: torch.optim.Optimizer) -> dict[str, float]:
        self.router.train()
        batch = self.sampler.sample_batch()
        losses = self.router_loss_step(batch)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        return {key: float(value.detach().cpu()) for key, value in losses.items()}

    def query_only_prompt(self, query: str) -> str:
        return query

    def full_lm_training_step(self, examples: list[dict[str, Any]]):
        raise NotImplementedError(
            "Full LM training requires cached LoRA precompute and adapter application. "
            "Use router_loss_step for current Phase 3 router/loss smoke training."
        )

    @staticmethod
    def _batch_source_vocab(examples: list[dict[str, Any]]) -> dict[str, int]:
        keys = []
        for ex in examples:
            for source_key in ex["documents"]:
                if source_key not in keys:
                    keys.append(source_key)
        return {source_key: idx for idx, source_key in enumerate(keys)}

    @staticmethod
    def _target_source_id(example: dict[str, Any], source_vocab: dict[str, int]) -> int | None:
        source = example["ground_truth"].get("source")
        if isinstance(source, list):
            source = source[0] if source else None
        if source is None:
            return None
        return source_vocab.get(source, 0)

    @staticmethod
    def _cache_key_for_chunk(doc: dict[str, Any], chunk: dict[str, Any]) -> str:
        return (
            f"{doc['source_id']}__chunk_{int(chunk['chunk_id'])}"
            f"__{text_hash(chunk['text'])[:16]}"
        )

    @staticmethod
    def _conflict_pairs_for_example(
        example: dict[str, Any],
        source_chunk_to_flat: dict[tuple[str, int], int],
    ) -> list[tuple[int, int, int]]:
        pairs = []
        for ann in annotations_from_example(example):
            left = source_chunk_to_flat.get((ann.source_a, ann.chunk_a))
            right = source_chunk_to_flat.get((ann.source_b, ann.chunk_b))
            if left is None or right is None:
                continue
            winner_idx = 1 if ann.winner_source in {ann.source_b, "B"} else 0
            pairs.append((left, right, winner_idx))
        return pairs
