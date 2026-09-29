# DocPatch

Implementation of *DocPatch: Continual Knowledge Internalization with
Conflict-Aware LoRA Memories* (`iclr2027_docpatch.pdf`, ICLR 2027 submission)
as a standalone package. See `MAPPING.md` for the full paper-section/equation
-> code index.

This package sits on top of the repo's pre-existing Doc-to-LoRA (D2L)
infrastructure under `src/ctx_to_lora/` -- in particular the pretrained
Qwen3-4B hypernetwork checkpoint at
`trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin` and the partial
DocPatch scaffolding already in `src/ctx_to_lora/docpatch/` (chunking, cache
bookkeeping, the router architecture, and the exact rank-concatenation
composition primitive). It reuses those pieces and adds what was missing:
explicit conflict-group/version resolution (Eq. 9-10), the global reasoning
adapter and its training curriculum (Section 3.6, Appendix A.6), the
paper-exact query encoder and LoRA-sketch feature (Appendix A.2/A.3), and an
end-to-end inference pipeline + full Table 1-6 evaluation suite.

## Layout

```
docpatch/
  config.py                 Section 3 config dataclasses (paper notation)
  metadata.py                mu_i: source/chunk/version/conflict-group (3.2, 3.4)
  types.py                    RetrievedKnowledge value type threaded through the pipeline
  knowledge_bank.py            Offline knowledge bank M = {(L_i, z_i, mu_i)} (3.2)
  query_encoder.py              e_q = P_q(Pool(F_LLM(q))) (Appendix A.2)
  lora_sketch.py                z_i sketch feature via fixed P_in/P_out + learned P_LoRA (Appendix A.3)
  router.py                     s_i, p_i, L_chunk, L_source, L_router (3.3, Eq. 2-8)
  conflict_resolution.py         m_i, g_i (3.4, Eq. 9-10)
  composition.py                  Delta W_know via exact rank-concat (3.5, Eq. 11-12)
  reasoning_adapter.py             Delta W_reason, Eq. 13-14 composition, model patching
  pipeline.py                       DocPatchPipeline: q -> ... -> answer
  data/
    counterfactual.py                Appendix A.6 counterfactual pairs
    curriculum.py                     Section 3.6 staged curriculum
  knowledge_bank / training CLIs   scripts/, training/
  evaluation/                       Tables 1, 2, 4, 5, 6
  tests/                            unit tests (pytest-compatible; also runnable without pytest, see below)
```

## Setup

Follow the repo's existing convention: everything runs with `PYTHONPATH`
pointing at both `src` (for `ctx_to_lora`) and the repo root (for
`docpatch`):

```bash
cd doc-to-lora-base
export PYTHONPATH=src:.
```

GPU-dependent steps (knowledge-bank construction, router/reasoning-adapter
training, and the end-to-end pipeline) need the pretrained D2L checkpoint
and a GPU (or a slow CPU fallback); this cluster's login node has no GPU
device, so those steps are meant to run as Slurm jobs, matching
`scripts/main_exp/*-slurm.sh`.

## End-to-end usage

```bash
# 1. Build a knowledge bank from an existing DocPatch dataset (chunks -> LoRAs
#    via the frozen D2L hypernetwork, plus conflict-group/version metadata).
python docpatch/scripts/build_knowledge_bank.py \
    --dataset data_preprocess/outputs/mixed/docpatch_train.jsonl \
    --output-dir cached_loras/docpatch_bank

# 2. Train the router (Eq. 8) against that bank.
python docpatch/training/train_router.py \
    --train data_preprocess/outputs/mixed/docpatch_train.jsonl \
    --dev data_preprocess/outputs/mixed/docpatch_dev.jsonl \
    --bank-dir cached_loras/docpatch_bank \
    --output-dir checkpoints/docpatch_router

# 3. Train the global reasoning adapter (Eq. 15, staged curriculum).
python docpatch/training/train_reasoning_adapter.py \
    --train data_preprocess/outputs/mixed/docpatch_train.jsonl \
    --bank-dir cached_loras/docpatch_bank \
    --output-dir checkpoints/docpatch_reasoning_adapter --steps 40000

# 4. Ask a question end to end.
python docpatch/scripts/run_inference.py \
    --bank-dir cached_loras/docpatch_bank \
    --router-checkpoint checkpoints/docpatch_router/best.pt \
    --reasoning-checkpoint checkpoints/docpatch_reasoning_adapter/checkpoint-40000.pt \
    --question "What is the official color of Caldera Port?"

# 5. Reproduce the paper's tables.
python docpatch/evaluation/evaluate_qa.py --dataset data_preprocess/outputs/mixed/docpatch_dev.jsonl \
    --bank-dir cached_loras/docpatch_bank --mode docpatch \
    --router-checkpoint checkpoints/docpatch_router/best.pt \
    --reasoning-checkpoint checkpoints/docpatch_reasoning_adapter/checkpoint-40000.pt   # Table 1
python docpatch/evaluation/evaluate_router.py ...                                        # Table 2
python docpatch/evaluation/evaluate_continual.py ...                                     # Table 4
python docpatch/evaluation/evaluate_qa.py --mode docpatch_no_reasoning ...                # Table 5 ablations
python docpatch/evaluation/measure_efficiency.py ...                                     # Table 6
```

Every script also has a `--dry-run` flag that exercises the data/shape
plumbing without touching the GPU or the model, useful for checking a config
change before submitting a training job.

## Tests

There is no network access on this cluster to install `pytest` into the
existing conda environments, so `docpatch/tests/` is plain-`assert`-based and
also runnable standalone:

```bash
PYTHONPATH=src:. python docpatch/tests/run_tests.py
```

(the same files are ordinary pytest test modules too -- `pytest
docpatch/tests` works once pytest is available). Coverage is everything that
doesn't require GPU or model weights: conflict resolution (Eq. 9-10), exact
rank-concatenation composition (Eq. 25) including the reasoning adapter's
additive composition (Eq. 14), the router's multi-hot/source losses (Eq.
4-8), the LoRA sketch feature (Appendix A.3), and metadata/counterfactual/
curriculum construction against the real
`data_preprocess/outputs/mixed/docpatch_dev.jsonl`. The GPU-dependent paths
(knowledge-bank building, router/reasoning-adapter training, end-to-end
`DocPatchPipeline.answer`) are exercised via each script's `--dry-run` mode
instead, since they need the pretrained D2L checkpoint and a GPU that aren't
available in this interactive session.

**Known environment issue on this login node:** `import peft` (a dependency
of `ctx_to_lora.utils`/`ctx_to_lora.model_loading`, and therefore of
anything touching the D2L hypernetwork) blocks for minutes here -- confirmed
via `py-spy`-style process inspection to be stuck opening `/dev/nvidia*`
during CUDA initialization, unrelated to this package's code (a bare `import
peft` with nothing else from this repo hangs the same way; `import torch`
alone does not). It is a pre-existing property of this compute node/driver,
not something introduced by `docpatch/`, and it will also affect every other
script in the repo that imports `peft` (`train.py`, `run_eval.py`,
`scripts/precompute_docpatch_chunk_loras.py`, etc.) when run interactively
here. Run those on an actual GPU allocation (`scripts/main_exp/*-slurm.sh`
shows the pattern) rather than the login node.
