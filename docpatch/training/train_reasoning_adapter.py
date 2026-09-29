#!/usr/bin/env python
"""CLI: train the global reasoning adapter (Section 3.6, Eq. 15, Appendix A.6).

    L_answer = -sum_t log p_theta_reason(y_t | q, y_<t; K)

Everything except the reasoning adapter stays frozen: the base LLM, the
document-specific knowledge LoRAs (activated here from ground-truth chunks,
"allowing the reasoning adapter to learn how to use document-derived
knowledge independently of retrieval errors" -- Section 3.6), the query
encoder, and the router.

Follows the staged curriculum from ``docpatch/data/curriculum.py``:
ground-truth single-LoRA -> counterfactual pairs -> multi-LoRA -> versioned
conflicts (conflict-resolved via ``docpatch/conflict_resolution.py``).

Processes one example per step (variable active-knowledge rank makes naive
batching require rank-padding across a batch; kept simple and correct here,
with ``--grad-accum-steps`` for effective larger batches). Usage:

    PYTHONPATH=src:. python docpatch/training/train_reasoning_adapter.py \\
        --train data_preprocess/outputs/mixed/docpatch_train.jsonl \\
        --bank-dir cached_loras/docpatch_bank \\
        --output-dir checkpoints/docpatch_reasoning_adapter \\
        --steps 40000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from ctx_to_lora.docpatch.cache import chunk_cache_key
from ctx_to_lora.docpatch.data import DocPatchDataset
from ctx_to_lora.docpatch.losses import compute_task_loss
from docpatch.composition import compose_from_paths
from docpatch.config import ReasoningAdapterConfig
from docpatch.conflict_resolution import compute_applicability_mask
from docpatch.data.counterfactual import CounterfactualPair
from docpatch.data.curriculum import ReasoningCurriculum
from docpatch.knowledge_bank import KnowledgeBank, load_d2l_model
from docpatch.metadata import build_metadata_for_ref
from docpatch.reasoning_adapter import apply_reasoning_and_knowledge, reasoning_adapter_from_hypernet
from docpatch.value_types import RetrievedKnowledge


def _bank_lookup(bank: KnowledgeBank):
    by_key = bank.by_cache_key()
    return by_key


def ground_truth_knowledge_tree(example: dict, dataset: DocPatchDataset, bank_lookup: dict, apply_conflict_resolution: bool):
    """Activate the ground-truth supporting chunk LoRAs uniformly-weighted
    (Table 3's "Ground-truth Knowledge LoRA" setting: bypass routing, use the
    exact supporting adapters). When ``apply_conflict_resolution`` is set
    (curriculum stage 4), versioned duplicates within the ground-truth set
    are resolved via Eq. 9-10 before composition."""

    refs = DocPatchDataset.relevant_chunk_refs(example)
    retrieved = []
    for ref in refs:
        key = chunk_cache_key(ref)
        record = bank_lookup.get(key)
        if record is None:
            continue
        metadata = build_metadata_for_ref(example, ref)
        retrieved.append(RetrievedKnowledge(key, record.lora_path, metadata, 0.0, 1.0))
    if not retrieved:
        return None
    if apply_conflict_resolution:
        mask = compute_applicability_mask(retrieved)
        active = [r for r, m in zip(retrieved, mask) if m == 1]
    else:
        active = retrieved
    if not active:
        return None
    weight = 1.0 / len(active)
    return compose_from_paths([r.lora_path for r in active], [weight] * len(active))


def counterfactual_knowledge_tree(pair: CounterfactualPair, bank_lookup: dict):
    paths = []
    for ref in pair.active_chunks:
        key = chunk_cache_key(ref)
        record = bank_lookup.get(key)
        if record is not None:
            paths.append(record.lora_path)
    if not paths:
        return None
    weight = 1.0 / len(paths)
    return compose_from_paths(paths, [weight] * len(paths))


def build_example_batch(item, dataset: DocPatchDataset, bank_lookup: dict, stage_name: str):
    """Returns (query, target_answer, knowledge_tree | None)."""

    if isinstance(item, CounterfactualPair):
        return item.query, item.target_answer, counterfactual_knowledge_tree(item, bank_lookup)
    example = item
    apply_conflict = stage_name == "conflict"
    tree = ground_truth_knowledge_tree(example, dataset, bank_lookup, apply_conflict)
    answer = example["ground_truth"]["answer"]
    if isinstance(answer, list):
        answer = answer[0]
    return example["query"], answer, tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, default=Path("trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/docpatch_reasoning_adapter"))
    parser.add_argument("--steps", type=int, default=40_000)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Build a few batches without any model forward/backward")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset = DocPatchDataset(args.train)
    bank = KnowledgeBank(args.bank_dir)
    if len(bank) == 0:
        raise SystemExit(f"Knowledge bank at {args.bank_dir} is empty; build it first.")
    bank_lookup = _bank_lookup(bank)

    config = ReasoningAdapterConfig(rank=args.rank, num_steps=args.steps)
    curriculum = ReasoningCurriculum(dataset, config)
    print("Curriculum stage sizes:", curriculum.stage_counts())

    if args.dry_run:
        for step in (0, args.steps // 4, args.steps // 2, args.steps - 1):
            progress = step / max(args.steps - 1, 1)
            pool = curriculum.active_pool(progress)
            item = rng.choice(pool)
            query, target, tree = build_example_batch(item, dataset, bank_lookup, curriculum.stages[curriculum.stage_index_for_progress(progress)].name)
            print(f"step={step} progress={progress:.2f} stage_idx={curriculum.stage_index_for_progress(progress)} "
                  f"query={query[:60]!r} target={target[:40]!r} n_modules={0 if tree is None else len(tree)}")
        return

    model = load_d2l_model(args.backbone_checkpoint)
    from ctx_to_lora.model_loading import get_tokenizer

    tokenizer = get_tokenizer(model.ctx_encoder.base_model.name_or_path)
    model.base_model.to(args.device)
    model.patch_lora_forward()
    for p in model.base_model.parameters():
        p.requires_grad_(False)

    reasoning_adapter = reasoning_adapter_from_hypernet(model, rank=args.rank).to(args.device)
    optimizer = torch.optim.AdamW(reasoning_adapter.parameters(), lr=args.lr, weight_decay=config.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    for step in range(1, args.steps + 1):
        progress = (step - 1) / max(args.steps - 1, 1)
        stage_idx = curriculum.stage_index_for_progress(progress)
        pool = curriculum.active_pool(progress)
        item = rng.choice(pool)
        query, target, tree = build_example_batch(item, dataset, bank_lookup, curriculum.stages[stage_idx].name)

        apply_reasoning_and_knowledge(model.base_model, reasoning_adapter, tree)

        prompt = tokenizer.apply_chat_template([{"role": "user", "content": query}], tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
        target_ids = tokenizer(target + tokenizer.eos_token, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)
        labels = torch.cat([torch.full_like(prompt_ids, -100), target_ids], dim=1)
        answer_mask = torch.cat([torch.zeros_like(prompt_ids), torch.ones_like(target_ids)], dim=1)

        outputs = model.base_model(input_ids=input_ids)
        shift_logits = outputs.logits[:, :-1]
        shift_labels = labels[:, 1:]
        shift_mask = answer_mask[:, 1:]
        loss = compute_task_loss(shift_logits, shift_labels, shift_mask) / args.grad_accum_steps
        loss.backward()
        running_loss += loss.item()

        if step % args.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        model.reset()
        model.patch_lora_forward()

        if step % 50 == 0:
            print(f"step {step} stage={curriculum.stages[stage_idx].name} loss={running_loss / 50:.4f}")
            running_loss = 0.0

        if step % args.save_every == 0 or step == args.steps:
            torch.save({"reasoning_adapter": reasoning_adapter.state_dict(), "step": step}, args.output_dir / f"checkpoint-{step}.pt")

    with (args.output_dir / "run_config.json").open("w") as f:
        json.dump(vars(args), f, default=str, indent=2)


if __name__ == "__main__":
    main()
