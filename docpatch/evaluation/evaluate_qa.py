#!/usr/bin/env python
"""CLI: Table 1 / Table 3 / Table 5 style QA evaluation (EM / F1).

Modes:
  frozen_llm            -- question-only prompt, no knowledge at all
  rag                    -- ground-truth supporting chunk text inserted into the prompt
  ground_truth_lora       -- ground-truth chunk LoRAs composed directly (bypasses routing)
  docpatch                -- full pipeline: router -> conflict resolution -> composition -> reasoning adapter
  docpatch_no_reasoning    -- same routing/composition, reasoning adapter disabled (ablation, Table 5)
  uniform_composition      -- Table 3's Top-5 Uniform Composition baseline
  router_weighted          -- Table 3's Top-5 Router-Weighted Composition baseline
  top1_only                -- Table 5 ablation: only the single top-ranked LoRA is ever used

Usage:
    PYTHONPATH=src:. python docpatch/evaluation/evaluate_qa.py \\
        --dataset data_preprocess/outputs/mixed/docpatch_dev.jsonl \\
        --bank-dir cached_loras/docpatch_bank --mode docpatch \\
        --router-checkpoint checkpoints/docpatch_router/best.pt \\
        --reasoning-checkpoint checkpoints/docpatch_reasoning_adapter/checkpoint-40000.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ctx_to_lora.docpatch.data import DocPatchDataset
from docpatch.composition import compose_from_paths
from docpatch.config import DocPatchConfig
from docpatch.evaluation.metrics import qa_metrics
from docpatch.knowledge_bank import KnowledgeBank, load_d2l_model
from docpatch.lora_sketch import LoRAFeatureProjector
from docpatch.pipeline import DocPatchPipeline
from docpatch.query_encoder import QueryEncoder, QueryProjector
from docpatch.reasoning_adapter import apply_reasoning_and_knowledge, reasoning_adapter_from_hypernet
from docpatch.router import DocPatchRouter
from ctx_to_lora.docpatch.cache import chunk_cache_key


def get_ground_truth_answer(example: dict) -> str:
    answer = example["ground_truth"]["answer"]
    return answer[0] if isinstance(answer, list) else answer


def generate(model, tokenizer, prompt_content: str, max_new_tokens: int = 64) -> str:
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt_content}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)  # template has BOS
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def run_frozen_llm(model, tokenizer, dataset) -> list[str]:
    return [generate(model.base_model, tokenizer, ex["query"]) for ex in dataset.examples]


def run_rag(model, tokenizer, dataset) -> list[str]:
    preds = []
    for ex in dataset.examples:
        refs = DocPatchDataset.relevant_chunk_refs(ex)
        context = "\n\n".join(r.text for r in refs)
        prompt = f"Context:\n{context}\n\nQuestion: {ex['query']}"
        preds.append(generate(model.base_model, tokenizer, prompt))
    return preds


def run_ground_truth_lora(model, tokenizer, dataset, bank: KnowledgeBank, reasoning_adapter=None) -> list[str]:
    by_key = bank.by_cache_key()
    preds = []
    for ex in dataset.examples:
        refs = DocPatchDataset.relevant_chunk_refs(ex)
        paths = [by_key[chunk_cache_key(r)].lora_path for r in refs if chunk_cache_key(r) in by_key]
        tree = compose_from_paths(paths, [1.0 / len(paths)] * len(paths)) if paths else None
        if reasoning_adapter is not None:
            model.patch_lora_forward()
            apply_reasoning_and_knowledge(model.base_model, reasoning_adapter, tree)
        elif tree is not None:
            from ctx_to_lora.modeling.lora_layer import apply_lora_to_layers

            model.patch_lora_forward()
            n_qs = torch.ones(1, dtype=torch.long, device=model.device)
            apply_lora_to_layers(model.base_model, model.hypernet.layer_indices, tree, n_qs)
        preds.append(generate(model.base_model, tokenizer, ex["query"]))
        model.reset()
    return preds


def _load_weights(module, path, key: str, device) -> None:
    """Load ``module`` from ``path``: either a checkpoint dict holding ``key`` or a raw state_dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    module.load_state_dict(ckpt[key] if key in ckpt else ckpt)


def build_pipeline(
    model,
    tokenizer,
    router_ckpt_path,
    reasoning_ckpt_path,
    config: DocPatchConfig,
    query_projector_ckpt_path=None,
    lora_projector_ckpt_path=None,
) -> DocPatchPipeline:
    """Build query encoder + router + LoRA projector + reasoning adapter.

    Checkpoint paths of ``None`` mean random initialization. ``router_ckpt_path`` is the
    file written by ``train_router.py`` (router + query projector + LoRA projector). The
    two projectors can also be loaded from their own files, which take precedence over
    the router checkpoint. The bank is not part of the pipeline: pass a list of
    ``BankRecord`` to ``pipeline.answer(question, bank)``. Bank records must be built with
    ``sketch_dim=config.knowledge_bank.sketch_dim``.
    """
    # Sketch feature size: one sketch_dim x sketch_dim sketch per adapted module (Appendix A.3).
    feature_dim = len(model.hypernet.target_modules) * config.knowledge_bank.sketch_dim ** 2
    query_projector = QueryProjector(model.base_model.config.hidden_size, config.router.hidden_size).to(model.device)
    query_encoder = QueryEncoder(
        model.base_model, tokenizer, query_projector,
        pooling=config.query_encoder.pooling, max_length=config.query_encoder.max_length,
    )
    lora_projector = LoRAFeatureProjector(feature_dim, config.router.hidden_size).to(model.device)
    router = DocPatchRouter(config.router.hidden_size, config.router.router_hidden_size, config.router.max_sources, config.router.dropout).to(model.device)
    if router_ckpt_path is not None:
        ckpt = torch.load(router_ckpt_path, map_location=model.device, weights_only=False)
        router.load_state_dict(ckpt["router"])
        # if "query_projector" in ckpt:
        #     query_projector.load_state_dict(ckpt["query_projector"])
        # if "lora_projector" in ckpt:
        #     lora_projector.load_state_dict(ckpt["lora_projector"])
    if query_projector_ckpt_path is not None:
        _load_weights(query_projector, query_projector_ckpt_path, "query_projector", model.device)
    if lora_projector_ckpt_path is not None:
        _load_weights(lora_projector, lora_projector_ckpt_path, "lora_projector", model.device)
    router.eval()
    query_projector.eval()
    lora_projector.eval()

    reasoning_adapter = reasoning_adapter_from_hypernet(model, rank=config.reasoning_adapter.rank).to(model.device)
    if reasoning_ckpt_path is not None:
        ckpt = torch.load(reasoning_ckpt_path, map_location=model.device, weights_only=False)
        reasoning_adapter.load_state_dict(ckpt["reasoning_adapter"])
    reasoning_adapter.eval()

    return DocPatchPipeline(model, tokenizer, query_encoder, router, lora_projector, reasoning_adapter, config)


def run_docpatch(model, tokenizer, dataset, bank, router_ckpt, reasoning_ckpt, config, mode: str) -> list[str]:
    pipeline = build_pipeline(model, tokenizer, router_ckpt, reasoning_ckpt if mode != "docpatch_no_reasoning" else None, config)
    comp_mode = {"uniform_composition": "uniform", "router_weighted": "router_weighted"}.get(mode)
    preds = []
    for ex in dataset.examples:
        top_k = 1 if mode == "top1_only" else config.router.top_k
        pipeline.config.router.top_k = top_k
        answer = pipeline.answer(ex["query"], bank.records, composition_mode=comp_mode)
        preds.append(answer.text)
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, default=None)
    parser.add_argument("--backbone-checkpoint", type=Path, default=Path("trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"))
    parser.add_argument(
        "--mode",
        choices=["frozen_llm", "rag", "ground_truth_lora", "docpatch", "docpatch_no_reasoning", "uniform_composition", "router_weighted", "top1_only"],
        required=True,
    )
    parser.add_argument("--router-checkpoint", type=Path, default=None)
    parser.add_argument("--reasoning-checkpoint", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dataset = DocPatchDataset(args.dataset)
    if args.limit:
        dataset.examples = dataset.examples[: args.limit]

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
        config = DocPatchConfig()
        preds = run_docpatch(model, tokenizer, dataset, bank, args.router_checkpoint, args.reasoning_checkpoint, config, args.mode)

    refs = [get_ground_truth_answer(ex) for ex in dataset.examples]
    metrics = qa_metrics(preds, refs)
    print(json.dumps({"mode": args.mode, **metrics, "n": len(dataset.examples)}, indent=2))
    if args.output:
        with args.output.open("w") as f:
            json.dump({"mode": args.mode, "metrics": metrics, "predictions": preds, "references": refs}, f, indent=2)


if __name__ == "__main__":
    main()
