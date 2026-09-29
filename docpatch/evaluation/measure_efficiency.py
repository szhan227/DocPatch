#!/usr/bin/env python
"""CLI: Table 6 style inference-efficiency measurement.

Reports input token count, per-query latency, and peak GPU memory for a
question-only DocPatch prompt vs. a RAG prompt that inlines the ground-truth
supporting chunk text (Section 4.7: "RAG processes approximately 4000 input
tokens per query, compared with about 50 for [DocPatch]").
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ctx_to_lora.docpatch.data import DocPatchDataset
from docpatch.config import DocPatchConfig
from docpatch.knowledge_bank import KnowledgeBank, load_d2l_model


def measure(fn, n_repeats: int = 5) -> dict[str, float]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / n_repeats
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")
    return {"latency_ms": elapsed_ms, "peak_gpu_gb": peak_gb}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--router-checkpoint", type=Path, default=None)
    parser.add_argument("--reasoning-checkpoint", type=Path, default=None)
    parser.add_argument("--backbone-checkpoint", type=Path, default=Path("trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"))
    parser.add_argument("--n-examples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    dataset = DocPatchDataset(args.dataset)
    dataset.examples = dataset.examples[: args.n_examples]

    model = load_d2l_model(args.backbone_checkpoint)
    from ctx_to_lora.model_loading import get_tokenizer

    tokenizer = get_tokenizer(model.ctx_encoder.base_model.name_or_path)
    bank = KnowledgeBank(args.bank_dir)

    from docpatch.evaluation.evaluate_qa import build_pipeline

    pipeline = build_pipeline(model, tokenizer, args.router_checkpoint, args.reasoning_checkpoint, DocPatchConfig())

    results = {}
    for ex in dataset.examples[:1]:
        rag_context = "\n\n".join(r.text for r in DocPatchDataset.relevant_chunk_refs(ex))
        rag_prompt = f"Context:\n{rag_context}\n\nQuestion: {ex['query']}"
        rag_tokens = len(tokenizer(rag_prompt).input_ids)
        dp_tokens = len(tokenizer(ex["query"]).input_ids)

        def rag_call():
            inputs = tokenizer(rag_prompt, return_tensors="pt").to(model.device)
            model.base_model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

        def dp_call():
            pipeline.answer(ex["query"], bank.records, max_new_tokens=args.max_new_tokens)

        results["rag"] = {"input_tokens": rag_tokens, **measure(rag_call)}
        results["docpatch"] = {"input_tokens": dp_tokens, **measure(dp_call)}

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
