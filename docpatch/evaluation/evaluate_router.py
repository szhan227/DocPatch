#!/usr/bin/env python
"""CLI: Table 2 style routing evaluation (Recall / F1 / Accuracy @ top-k, Eq. 16)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ctx_to_lora.docpatch.data import DocPatchDataset
from docpatch.config import RouterConfig
from docpatch.knowledge_bank import KnowledgeBank, load_d2l_model
from docpatch.lora_sketch import LoRAFeatureProjector
from docpatch.query_encoder import QueryEncoder, QueryProjector
from docpatch.router import DocPatchRouter
from docpatch.training.train_router import evaluate as evaluate_router_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--router-checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, default=Path("trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    config = RouterConfig()
    dataset = DocPatchDataset(args.dataset)
    bank = KnowledgeBank(args.bank_dir)

    model = load_d2l_model(args.backbone_checkpoint)
    from ctx_to_lora.model_loading import get_tokenizer

    tokenizer = get_tokenizer(model.ctx_encoder.base_model.name_or_path)

    query_projector = QueryProjector(model.base_model.config.hidden_size, config.hidden_size).to(model.device)
    query_encoder = QueryEncoder(model.base_model, tokenizer, query_projector)
    lora_projector = LoRAFeatureProjector(bank.feature_dim, config.hidden_size).to(model.device)
    router = DocPatchRouter(config.hidden_size, config.router_hidden_size, config.max_sources, config.dropout).to(model.device)

    ckpt = torch.load(args.router_checkpoint, map_location=model.device, weights_only=False)
    router.load_state_dict(ckpt["router"])
    query_projector.load_state_dict(ckpt["query_projector"])
    lora_projector.load_state_dict(ckpt["lora_projector"])
    router.eval()

    metrics = evaluate_router_dataset(dataset, bank, query_encoder, lora_projector, router, args.top_k)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
