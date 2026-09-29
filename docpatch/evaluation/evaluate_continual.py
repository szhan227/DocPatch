#!/usr/bin/env python
"""CLI: Table 4 style continual knowledge updating evaluation.

Maps the mixed DocPatch dataset's synthetic-conflict categories onto the
paper's five continual-update test buckets (Section 4.5 caption):

    Unchanged  -- multi_source_consistent_{synthetic,hotpot}: newer document
                   versions didn't change the knowledge relevant to the query
    Updated    -- b_new_synthetic: a newer version introduces knowledge that
                   contradicts the pretrained model's prior
    Old-Only   -- a_exclusive_synthetic: relevant knowledge only exists in an
                   older document version
    Override   -- multi_source_conflict_synthetic: newer knowledge directly
                   contradicts (and must override) an older counterpart
    Unrelated  -- unrelated_locality: no active LoRA carries knowledge
                   relevant to the query

Usage mirrors ``evaluate_qa.py`` -- point it at any of that script's modes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctx_to_lora.docpatch.data import DocPatchDataset
from docpatch.config import DocPatchConfig
from docpatch.evaluation.evaluate_qa import (
    get_ground_truth_answer,
    run_docpatch,
    run_frozen_llm,
    run_ground_truth_lora,
    run_rag,
)
from docpatch.evaluation.metrics import continual_update_accuracy
from docpatch.knowledge_bank import KnowledgeBank, load_d2l_model

CATEGORY_TO_TABLE4 = {
    "multi_source_consistent_synthetic": "unchanged",
    "multi_source_consistent_hotpot": "unchanged",
    "b_new_synthetic": "updated",
    "a_exclusive_synthetic": "old_only",
    "multi_source_conflict_synthetic": "override",
    "unrelated_locality": "unrelated",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, default=None)
    parser.add_argument("--backbone-checkpoint", type=Path, default=Path("trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"))
    parser.add_argument("--mode", choices=["frozen_llm", "rag", "ground_truth_lora", "docpatch", "docpatch_no_reasoning"], required=True)
    parser.add_argument("--router-checkpoint", type=Path, default=None)
    parser.add_argument("--reasoning-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    dataset = DocPatchDataset(args.dataset)
    categories = [CATEGORY_TO_TABLE4.get(ex["category"]) for ex in dataset.examples]
    keep = [i for i, c in enumerate(categories) if c is not None]
    dataset.examples = [dataset.examples[i] for i in keep]
    categories = [categories[i] for i in keep]

    model = load_d2l_model(args.backbone_checkpoint)
    from ctx_to_lora.model_loading import get_tokenizer

    tokenizer = get_tokenizer(model.ctx_encoder.base_model.name_or_path)

    if args.mode == "frozen_llm":
        preds = run_frozen_llm(model, tokenizer, dataset)
    elif args.mode == "rag":
        preds = run_rag(model, tokenizer, dataset)
    elif args.mode == "ground_truth_lora":
        bank = KnowledgeBank(args.bank_dir)
        preds = run_ground_truth_lora(model, tokenizer, dataset, bank)
    else:
        bank = KnowledgeBank(args.bank_dir)
        preds = run_docpatch(model, tokenizer, dataset, bank, args.router_checkpoint, args.reasoning_checkpoint, DocPatchConfig(), args.mode)

    refs = [get_ground_truth_answer(ex) for ex in dataset.examples]
    results = continual_update_accuracy(preds, refs, categories)
    print(json.dumps({"mode": args.mode, **results}, indent=2))


if __name__ == "__main__":
    main()
