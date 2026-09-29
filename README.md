# DocPatch

Continual knowledge internalization with conflict-aware LoRA memories — an
implementation of *DocPatch* (ICLR 2027 submission) on top of Doc-to-LoRA.

Documents are turned into LoRA "memories" by a frozen Doc-to-LoRA hypernetwork and
kept in a bank. To answer a question, DocPatch

1. **routes** the question to the relevant LoRAs in the bank,
2. **resolves conflicts** between document versions (newer knowledge overrides older),
3. **composes** the surviving LoRAs exactly (rank concatenation),
4. applies them, together with a small **reasoning adapter**, to the frozen LLM, and
5. generates the answer from the **question alone** — no document text in the prompt.


## Setup

**Use the same conda environment as Doc-to-LoRA.** DocPatch needs no extra packages: it
only uses what Doc-to-LoRA already depends on (PyTorch, transformers, peft, …).

```bash
conda activate <your Doc-to-LoRA env>
cd DocPatch
```

There is nothing to `pip install`. Doc-to-LoRA's source is included in `src/ctx_to_lora`,
and `docpatch/main.py` puts `src/` and the repo root on `sys.path` itself.

You also need:

- **A GPU.** Loading the hypernetwork and the base LLM is not practical on CPU.
- **A Doc-to-LoRA checkpoint** (`pytorch_model.bin`), e.g. one from your Doc-to-LoRA
  `trained_d2l/` folder. Its base LLM (e.g. `google/gemma-2-2b-it`) must be in your
  Hugging Face cache or downloadable.

## Quick test: run `main.py`

`docpatch/main.py` is a playground: it loads the model, turns your texts into a bank of
LoRAs, asks one question, and prints the answer.

**1. Edit the settings in `docpatch/main.py`:**

| Variable | What to put |
|---|---|
| `D2L_CHECKPOINT` | Path to a Doc-to-LoRA `pytorch_model.bin` (hypernetwork + base LLM). |
| `router_path` | Router weights: a `.pt` dict with a `"router"` key (what `training/train_router.py` writes), or `None`. |
| `reasoning_ckpt_path` | Reasoning-adapter weights: a `.pt` dict with a `"reasoning_adapter"` key (what `training/train_reasoning_adapter.py` writes), or `None`. |
| `query_projector_path`, `lora_projector_path` | Projector weights: a raw `state_dict`, or a dict with a `"query_projector"` / `"lora_projector"` key, or `None`. |
| `TEXTS` | Your documents. Each string becomes one LoRA in the bank. |
| `QUESTION` | Your question. |

Any path set to `None` is **randomly initialized**. That is enough to check that the code
runs end to end, but the router and reasoning adapter only behave meaningfully once
trained (see [Training and evaluation](#training-and-evaluation)).

**2. Run it** from the repo root:

```bash
python docpatch/main.py
```

It prints the question and the model's answer, for example:

```
Q: Who founded the Zorblax Institute?
A: Dr. Maria Costa
```


## Training and evaluation

These are for reproducing the paper's experiments and need a GPU and a Doc-to-LoRA
checkpoint (pass `--backbone-checkpoint`; the default paths point at
`trained_d2l/...` and will not exist in a fresh clone). Run them from the repo root:

| Script | Purpose |
|---|---|
| `docpatch/scripts/build_knowledge_bank.py` | Build an on-disk bank from a dataset (chunks → LoRAs). |
| `docpatch/training/train_router.py` | Train the router and the two projectors (paper Eq. 8). |
| `docpatch/training/train_reasoning_adapter.py` | Train the reasoning adapter (Eq. 15, staged curriculum). |
| `docpatch/evaluation/evaluate_qa.py` | QA evaluation (EM/F1) and ablations. |
| `docpatch/evaluation/evaluate_router.py` | Routing Recall / F1 / Accuracy. |
| `docpatch/evaluation/evaluate_continual.py` | Continual-update categories. |
| `docpatch/evaluation/measure_efficiency.py` | Tokens, latency, GPU memory. |

Each supports `--help`; the training and build scripts also take `--dry-run` to check the
data plumbing without loading a model.

## Tests

Unit tests run on CPU without model weights:

```bash
python docpatch/tests/run_tests.py
```

`pytest docpatch/tests` also works if pytest is installed.

## Layout

```
docpatch/
  main.py                 playground: texts -> bank -> question -> answer
  pipeline.py             DocPatchPipeline: route -> resolve -> compose -> answer
  knowledge_bank.py       text -> LoRA record, on-disk bank, D2L model loading
  router.py               router losses (Eq. 4-8)
  conflict_resolution.py  version/conflict-group resolution (Eq. 9-10)
  composition.py          exact multi-LoRA composition (Eq. 11-12)
  reasoning_adapter.py    global reasoning adapter (Eq. 13-14)
  query_encoder.py        question representation (App. A.2)
  lora_sketch.py          LoRA -> router feature (App. A.3)
  data/                   counterfactual pairs, training curriculum
  training/  evaluation/  scripts/  tests/
src/ctx_to_lora/          Doc-to-LoRA source (hypernetwork, model loading, …)
```
