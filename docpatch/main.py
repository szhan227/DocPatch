"""DocPatch playground: turn texts into a bank (list) of LoRAs, ask a question.

Only the base LLM + Doc-to-LoRA hypernetwork are loaded from disk. The query
encoder, router, LoRA projector and reasoning adapter are created from scratch
with random parameters (checkpoint = None) -- this checks that the code runs
end to end, not that the answers are good.

Edit the checkpoint path, TEXTS and QUESTION below, then run:  python docpatch/main.py
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]

from ctx_to_lora.model_loading import get_tokenizer
from docpatch.config import DocPatchConfig, QueryEncoderConfig, ReasoningAdapterConfig, RouterConfig
from docpatch.evaluation.evaluate_qa import build_pipeline
from docpatch.knowledge_bank import load_d2l_model, text_to_lora_record

from docpatch.query_encoder import QueryProjector
from docpatch.lora_sketch import LoRAFeatureProjector
from docpatch.router import DocPatchRouter

D2L_CHECKPOINT = "/path/to/hypernet"  # frozen Doc-to-LoRA hypernetwork + base LLM

# ---- config for the modules that are NOT the base LLM / hypernetwork ----
config = DocPatchConfig(
    query_encoder=QueryEncoderConfig(pooling="mean", max_length=64),
    router=RouterConfig(hidden_size=256, router_hidden_size=512, top_k=5),
    reasoning_adapter=ReasoningAdapterConfig(rank=8),
    seed=0,
)
torch.manual_seed(config.seed)

# ---- load base LLM + hypernetwork ----
model = load_d2l_model(D2L_CHECKPOINT)
model.reset()
tokenizer = get_tokenizer(model.ctx_encoder.base_model.name_or_path)



# ---- pipeline ----
router_path = "/path/to/router"
reasoning_ckpt_path = "/path/to/reasoning_adapter"
lora_projector_path = "/path/to/lora_projector"
query_projector_path = "/path/to/query_projector"

pipeline = build_pipeline(
    model, tokenizer,
    router_ckpt_path=router_path,
    reasoning_ckpt_path=reasoning_ckpt_path,
    config=config,
    query_projector_ckpt_path=query_projector_path,
    lora_projector_ckpt_path=lora_projector_path,
)

# ---- bank: a list of LoRAs, each computed from a text by the hypernetwork ----
bank = []
TEXTS = [
    "The Zorblax Institute was founded in 1987 in Lisbon by Dr. Maria Costa.",
    "The Zorblax Institute's flagship product is the Quill-9 synthesizer.",
]
for i, text in enumerate(TEXTS):
    bank.append(text_to_lora_record(model, tokenizer, text, source_id=f"doc{i}",
                                    sketch_dim=config.knowledge_bank.sketch_dim))


if __name__ == "__main__":
    QUESTION = "Who founded the Zorblax Institute?"
    result = pipeline.answer(QUESTION, bank)
    print("Q:", QUESTION)
    print("A:", result.text)


