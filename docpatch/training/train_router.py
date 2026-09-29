#!/usr/bin/env python
"""CLI: train the DocPatch query-dependent router (Section 3.3).

Loss is the paper's exact Eq. 8: L_router = L_chunk + lambda_source * L_source,
with L_chunk the normalized multi-hot objective (Eq. 4-5) and L_source built
from summed chunk probabilities per source (Eq. 6-7) -- see
``docpatch/router.py`` for why this differs from
``scripts/train_constitution_router.py``'s single-label CE.

Candidates for each training example are that example's own document chunks
(so this works directly against the existing
``data_preprocess/outputs/mixed/docpatch_{train,dev}.jsonl``), looked up by
cache key in a knowledge bank built beforehand with
``docpatch/scripts/build_knowledge_bank.py --dataset ...``.

Usage (from repo root, after building a bank for the same dataset):

    PYTHONPATH=src:. python docpatch/training/train_router.py \\
        --train data_preprocess/outputs/mixed/docpatch_train.jsonl \\
        --dev data_preprocess/outputs/mixed/docpatch_dev.jsonl \\
        --bank-dir cached_loras/docpatch_bank \\
        --output-dir checkpoints/docpatch_router
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ctx_to_lora.docpatch.cache import chunk_cache_key
from ctx_to_lora.docpatch.data import BalancedDocPatchSampler, DocPatchDataset, SourceChunkRef
from docpatch.config import RouterConfig
from docpatch.evaluation.metrics import routing_metrics
from docpatch.knowledge_bank import KnowledgeBank
from docpatch.lora_sketch import LoRAFeatureProjector
from docpatch.query_encoder import QueryEncoder, QueryProjector
from docpatch.router import DocPatchRouter, build_multi_hot_targets, compute_router_loss


def batch_source_vocab(examples: list[dict]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for ex in examples:
        for doc in ex["documents"].values():
            source_id = doc["source_id"]
            if source_id not in vocab:
                vocab[source_id] = len(vocab)
    return vocab


def build_candidates(example: dict, bank: KnowledgeBank):
    """Every chunk in the example's own documents, looked up in the bank."""

    by_key = bank.by_cache_key()
    candidates = []
    gt_refs = {(r.source_id, r.chunk_id) for r in DocPatchDataset.relevant_chunk_refs(example)}
    gt_indices = []
    for source_key, doc in example["documents"].items():
        for chunk in doc["chunks"]:
            ref = SourceChunkRef(source_key=source_key, source_id=doc["source_id"], chunk_id=int(chunk["chunk_id"]), text=chunk["text"])
            key = chunk_cache_key(ref)
            record = by_key.get(key)
            if record is None:
                continue
            if (ref.source_id, ref.chunk_id) in gt_refs:
                gt_indices.append(len(candidates))
            candidates.append(record)
    return candidates, gt_indices


def make_batch(examples: list[dict], bank: KnowledgeBank, query_encoder: QueryEncoder, lora_projector: LoRAFeatureProjector):
    source_vocab = batch_source_vocab(examples)
    per_example_candidates = []
    per_example_gt = []
    target_source_ids = []
    for ex in examples:
        candidates, gt_indices = build_candidates(ex, bank)
        per_example_candidates.append(candidates)
        per_example_gt.append(gt_indices)
        gt_source = ex["ground_truth"].get("source")
        if isinstance(gt_source, list):
            gt_source = gt_source[0] if gt_source else None
        target_source_ids.append(source_vocab.get(gt_source, 0) if gt_source else 0)

    max_candidates = max((len(c) for c in per_example_candidates), default=0)
    if max_candidates == 0:
        return None

    device = next(lora_projector.parameters()).device
    z_dim = bank.feature_dim if bank._features else 0
    z = torch.zeros(len(examples), max_candidates, z_dim)
    source_ids = torch.zeros(len(examples), max_candidates, dtype=torch.long)
    chunk_mask = torch.zeros(len(examples), max_candidates, dtype=torch.bool)
    for b, candidates in enumerate(per_example_candidates):
        for i, record in enumerate(candidates):
            z[b, i] = bank.sketch_feature(record.cache_key)
            source_ids[b, i] = source_vocab[record.source_id]
            chunk_mask[b, i] = True

    e_q = query_encoder([ex["query"] for ex in examples]).to(device)
    z_proj = lora_projector(z.to(device))
    multi_hot = build_multi_hot_targets(per_example_gt, max_candidates).to(device)
    return {
        "e_q": e_q,
        "z": z_proj,
        "source_ids": source_ids.to(device),
        "chunk_mask": chunk_mask.to(device),
        "multi_hot": multi_hot,
        "target_source_ids": torch.tensor(target_source_ids, dtype=torch.long, device=device),
        "num_sources": len(source_vocab),
    }


def evaluate(dataset: DocPatchDataset, bank: KnowledgeBank, query_encoder, lora_projector, router, top_k: int) -> dict:
    all_recall, all_f1, all_acc = [], [], []
    for example in dataset.examples:
        candidates, gt_indices = build_candidates(example, bank)
        if not candidates or not gt_indices:
            continue
        device = next(router.parameters()).device
        z = torch.stack([bank.sketch_feature(c.cache_key) for c in candidates]).to(device)
        z_proj = lora_projector(z).unsqueeze(0)
        source_vocab = {c.source_id: i for i, c in enumerate({c.source_id: None for c in candidates})}
        source_ids = torch.tensor([[source_vocab[c.source_id] for c in candidates]], dtype=torch.long, device=device)
        e_q = query_encoder([example["query"]]).to(device)
        output = router(e_q, z_proj, source_ids)
        m = routing_metrics(output.chunk_gates[0], set(gt_indices), top_k)
        all_recall.append(m["recall"])
        all_f1.append(m["f1"])
        all_acc.append(m["accuracy"])
    n = max(len(all_recall), 1)
    return {
        "recall": sum(all_recall) / n,
        "f1": sum(all_f1) / n,
        "accuracy": sum(all_acc) / n,
        "n_eval": len(all_recall),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, default=Path("trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/docpatch_router"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true", help="Build one batch and print shapes, no training")
    args = parser.parse_args()

    config = RouterConfig()
    train_dataset = DocPatchDataset(args.train)
    dev_dataset = DocPatchDataset(args.dev)
    bank = KnowledgeBank(args.bank_dir)
    if len(bank) == 0:
        raise SystemExit(f"Knowledge bank at {args.bank_dir} is empty; build it first.")

    from docpatch.knowledge_bank import load_d2l_model
    from ctx_to_lora.model_loading import get_tokenizer

    d2l_model = load_d2l_model(args.backbone_checkpoint)
    tokenizer = get_tokenizer(d2l_model.ctx_encoder.base_model.name_or_path)
    d2l_model.base_model.to(args.device)

    query_projector = QueryProjector(d2l_model.base_model.config.hidden_size, config.hidden_size).to(args.device)
    query_encoder = QueryEncoder(d2l_model.base_model, tokenizer, query_projector, max_length=64)
    lora_projector = LoRAFeatureProjector(bank.feature_dim, config.hidden_size).to(args.device)
    router = DocPatchRouter(
        hidden_size=config.hidden_size,
        router_hidden_size=config.router_hidden_size,
        num_sources=config.max_sources,
        dropout=config.dropout,
    ).to(args.device)

    sampler = BalancedDocPatchSampler(train_dataset, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(
        list(router.parameters()) + list(query_projector.parameters()) + list(lora_projector.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    if args.dry_run:
        batch = make_batch(sampler.sample_batch(), bank, query_encoder, lora_projector)
        print({k: (v.shape if torch.is_tensor(v) else v) for k, v in batch.items()})
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_recall = -1.0
    for step in range(1, args.steps + 1):
        examples = sampler.sample_batch()
        batch = make_batch(examples, bank, query_encoder, lora_projector)
        if batch is None:
            continue
        output = router(batch["e_q"], batch["z"], batch["source_ids"], batch["chunk_mask"])
        losses = compute_router_loss(
            output,
            batch["multi_hot"],
            batch["source_ids"],
            batch["target_source_ids"],
            num_sources=max(batch["num_sources"], 1),
            source_loss_weight=config.source_loss_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step {step}: total={losses['total'].item():.4f} chunk={losses['chunk'].item():.4f} source={losses['source'].item():.4f}")

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(dev_dataset, bank, query_encoder, lora_projector, router, config.top_k)
            print(f"[eval @ {step}] {metrics}")
            if metrics["recall"] > best_recall:
                best_recall = metrics["recall"]
                torch.save(
                    {
                        "router": router.state_dict(),
                        "query_projector": query_projector.state_dict(),
                        "lora_projector": lora_projector.state_dict(),
                        "metrics": metrics,
                        "step": step,
                    },
                    args.output_dir / "best.pt",
                )
    with (args.output_dir / "run_config.json").open("w") as f:
        json.dump(vars(args) | {"best_recall": best_recall}, f, default=str, indent=2)


if __name__ == "__main__":
    main()
