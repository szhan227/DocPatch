#!/usr/bin/env python
"""CLI: ask DocPatch a single question end to end (demo / smoke test).

    PYTHONPATH=src:. python docpatch/scripts/run_inference.py \\
        --bank-dir cached_loras/docpatch_bank \\
        --router-checkpoint checkpoints/docpatch_router/best.pt \\
        --reasoning-checkpoint checkpoints/docpatch_reasoning_adapter/checkpoint-40000.pt \\
        --question "What is the official color of Caldera Port?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from docpatch.config import DocPatchConfig
from docpatch.evaluation.evaluate_qa import build_pipeline
from docpatch.knowledge_bank import KnowledgeBank, load_d2l_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--router-checkpoint", type=Path, default=None)
    parser.add_argument("--reasoning-checkpoint", type=Path, default=None)
    parser.add_argument("--backbone-checkpoint", type=Path, default=Path("trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"))
    args = parser.parse_args()

    model = load_d2l_model(args.backbone_checkpoint)
    from ctx_to_lora.model_loading import get_tokenizer

    tokenizer = get_tokenizer(model.ctx_encoder.base_model.name_or_path)
    bank = KnowledgeBank(args.bank_dir)
    pipeline = build_pipeline(model, tokenizer, args.router_checkpoint, args.reasoning_checkpoint, DocPatchConfig())

    result = pipeline.answer(args.question, bank.records)
    print("Answer:", result.text)
    print("Activated knowledge:")
    for item, gate in zip(result.retrieved, result.gates):
        print(f"  source={item.metadata.source_id} chunk={item.metadata.chunk_id} version={item.metadata.version} "
              f"alpha={item.alpha:.3f} gate={gate:.3f}")


if __name__ == "__main__":
    main()
