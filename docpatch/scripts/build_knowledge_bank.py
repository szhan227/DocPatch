#!/usr/bin/env python
"""CLI: build/extend a DocPatch knowledge bank (Section 3.2).

Wraps ``docpatch.knowledge_bank.KnowledgeBank`` around the frozen, pretrained
Doc-to-LoRA hypernetwork. Two modes:

  --dataset <DocPatchDataset jsonl>   batch-build from a DocPatch training/eval
                                        jsonl (chunks + version/conflict metadata
                                        inferred per docpatch/metadata.py)
  --document <path> --source-id <id> --version <int>
                                        append one new document (or document
                                        version) to an existing bank

Run under the repo's usual PYTHONPATH=src convention, e.g.:

    PYTHONPATH=src:. python docpatch/scripts/build_knowledge_bank.py \\
        --dataset data_preprocess/outputs/mixed/docpatch_dev.jsonl \\
        --output-dir cached_loras/docpatch_bank_dev --max-chunks 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ctx_to_lora.docpatch.data import DocPatchDataset
from docpatch.knowledge_bank import KnowledgeBank, load_d2l_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, help="DocPatchDataset jsonl to batch-build from")
    parser.add_argument("--document", type=Path, help="Plain-text file to append as a new document")
    parser.add_argument("--source-id", type=str, help="Source id for --document")
    parser.add_argument("--version", type=int, default=0, help="Document version for --document")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("cached_loras/docpatch_bank"))
    parser.add_argument("--sketch-dim", type=int, default=64)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without touching the GPU/model")
    args = parser.parse_args()

    if not args.dataset and not args.document:
        parser.error("must pass --dataset or --document")

    bank = KnowledgeBank(args.output_dir, sketch_dim=args.sketch_dim)
    print(f"Existing bank records: {len(bank)}")

    if args.dry_run:
        if args.dataset:
            dataset = DocPatchDataset(args.dataset)
            print(f"Dataset examples: {len(dataset)}, categories: {dataset.category_counts()}")
        else:
            print(f"Would add document {args.source_id!r} version={args.version} from {args.document}")
        return

    model = load_d2l_model(args.checkpoint, use_flash_attn=args.use_flash_attn)
    from ctx_to_lora.model_loading import get_tokenizer

    tokenizer = get_tokenizer(model.ctx_encoder.base_model.name_or_path)

    if args.dataset:
        dataset = DocPatchDataset(args.dataset)
        records = bank.build_from_dataset(model, tokenizer, dataset, max_chunks=args.max_chunks, overwrite=args.overwrite)
        print(f"Added/updated {len(records)} records from {args.dataset}")
    else:
        if not args.source_id:
            parser.error("--document requires --source-id")
        text = args.document.read_text(encoding="utf-8")
        records = bank.add_document(
            model, tokenizer, source_id=args.source_id, text=text, version=args.version, overwrite=args.overwrite
        )
        print(f"Added {len(records)} chunk records for source {args.source_id!r} (version {args.version})")

    print(f"Bank now has {len(bank)} records at {args.output_dir}")


if __name__ == "__main__":
    main()
