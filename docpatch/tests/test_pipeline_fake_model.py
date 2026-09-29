"""End-to-end test of the main.py flow on CPU with a tiny fake model.

Fakes stand in for the real base LLM / hypernetwork / tokenizer, but the layers
are real ``nn.Linear`` modules patched with the repo's real ``lora_forward``, so
this exercises: text -> LoRA record -> bank list -> route -> conflict resolution
-> composition (+ reasoning adapter) -> patched forward -> generate -> reset.
"""

from functools import partial
from types import SimpleNamespace

import torch
from torch import nn

from ctx_to_lora.modeling.lora_layer import lora_forward
from docpatch.config import DocPatchConfig, KnowledgeBankConfig, QueryEncoderConfig, RouterConfig
from docpatch.evaluation.evaluate_qa import build_pipeline
from docpatch.knowledge_bank import text_to_lora_record

HID, INTER, N_LAYERS, RANK = 32, 48, 2, 4


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(HID, INTER)
        self.down_proj = nn.Linear(INTER, HID)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _Mlp()


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(64, HID)
        self.layers = nn.ModuleList([_Layer() for _ in range(N_LAYERS)])


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()
        self.config = SimpleNamespace(hidden_size=HID)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        h = self.model.emb(input_ids)
        for layer in self.model.layers:
            h = h + layer.mlp.down_proj(torch.relu(layer.mlp.up(h)))
        return SimpleNamespace(hidden_states=[h])

    def generate(self, input_ids, attention_mask=None, max_new_tokens=4, do_sample=False):
        out = self.forward(input_ids).hidden_states[-1]
        next_ids = out[:, -1].argmax(-1, keepdim=True) % 64
        return torch.cat([input_ids, next_ids.expand(-1, max_new_tokens)], dim=1)


class FakeD2L:
    """Stands in for ModulatedPretrainedModel: patch/reset/generate_weights/hypernet."""

    device = torch.device("cpu")

    def __init__(self):
        self.base_model = FakeBase()
        self.hypernet = SimpleNamespace(
            target_modules=("down_proj",),
            d_in={"down_proj": INTER},
            d_out={"down_proj": HID},
            layer_indices=list(range(N_LAYERS)),
        )

    def _modules(self):
        return [layer.mlp.down_proj for layer in self.base_model.model.layers]

    def patch_lora_forward(self):
        for m in self._modules():
            if getattr(m, "patched_forward", False):
                continue
            m.forward_orig, m.patched_forward = m.forward, True
            m.forward = partial(lora_forward, self=m, lora_dropout_p=0.0, scaling=1.0)

    def reset(self):
        for m in self._modules():
            if getattr(m, "patched_forward", False):
                m.forward, m.patched_forward = m.forward_orig, False

    def generate_weights(self, ctx_ids, ctx_attn_mask, ctx_position_ids=None):
        g = torch.Generator().manual_seed(int(ctx_ids.sum()))
        return {"down_proj": {"A": torch.randn(1, N_LAYERS, RANK, INTER, generator=g),
                              "B": torch.randn(1, N_LAYERS, RANK, HID, generator=g)}}, None


class _Enc(dict):
    def to(self, device):
        return _Enc({k: v.to(device) for k, v in self.items()})


class FakeTokenizer:
    @staticmethod
    def _ids(text):
        return [ord(c) % 60 + 1 for c in text[:12]]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, return_dict=False, **kw):
        if tokenize:
            return {"input_ids": [self._ids(messages[0][0]["content"])]}
        return messages[0]["content"]

    def __call__(self, texts, return_tensors=None, padding=False, truncation=False, max_length=None, add_special_tokens=True):
        texts = [texts] if isinstance(texts, str) else texts
        rows = [self._ids(t) for t in texts]
        width = max(len(r) for r in rows)
        ids = torch.tensor([r + [0] * (width - len(r)) for r in rows])
        mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])
        return _Enc(input_ids=ids, attention_mask=mask)

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def _config():
    return DocPatchConfig(
        knowledge_bank=KnowledgeBankConfig(sketch_dim=8),
        query_encoder=QueryEncoderConfig(pooling="mean", max_length=16),
        router=RouterConfig(hidden_size=16, router_hidden_size=32, top_k=3),
    )


def test_main_flow_bank_list_into_answer_and_layers_reset():
    torch.manual_seed(0)
    model, tokenizer, config = FakeD2L(), FakeTokenizer(), _config()
    pipeline = build_pipeline(model, tokenizer, None, None, config)  # random-init modules

    texts = ["alpha fact about x", "beta fact about y", "gamma fact about z", "delta old fact", "delta new fact"]
    bank = [
        text_to_lora_record(model, tokenizer, t, source_id=f"doc{i}", sketch_dim=8,
                            version=1 if t == "delta new fact" else 0,
                            conflict_group="delta" if t.startswith("delta") else None)
        for i, t in enumerate(texts)
    ]
    assert all(r.lora_tree is not None and r.sketch is not None for r in bank)
    assert bank[0].sketch.shape == (8 * 8,)  # 1 module * sketch_dim^2

    result = pipeline.answer("what is x?", bank, max_new_tokens=3)

    assert isinstance(result.text, str) and result.text
    assert len(result.retrieved) == 3 and len(result.gates) == 3
    assert abs(sum(result.gates) - 1.0) < 1e-5 or sum(result.gates) > 0
    # Layers must be restored after answering (query encoding needs them unpatched).
    assert not any(getattr(m, "patched_forward", False) for m in model._modules())
    # ... so answering twice works (route runs the base model unpatched each time).
    pipeline.answer("what is y?", bank, max_new_tokens=3)


def test_older_conflicting_lora_is_suppressed_end_to_end():
    torch.manual_seed(1)
    model, tokenizer, config = FakeD2L(), FakeTokenizer(), _config()
    config.router.top_k = 2
    pipeline = build_pipeline(model, tokenizer, None, None, config)
    bank = [
        text_to_lora_record(model, tokenizer, "old", "docA", version=0, conflict_group="g", sketch_dim=8),
        text_to_lora_record(model, tokenizer, "new", "docB", version=1, conflict_group="g", sketch_dim=8),
    ]
    result = pipeline.answer("q", bank, max_new_tokens=2)
    by_source = {item.metadata.source_id: gate for item, gate in zip(result.retrieved, result.gates)}
    assert by_source["docA"] == 0.0 and by_source["docB"] > 0.0


def test_empty_bank_still_answers_with_reasoning_adapter_only():
    model, tokenizer = FakeD2L(), FakeTokenizer()
    pipeline = build_pipeline(model, tokenizer, None, None, _config())
    result = pipeline.answer("anything", [], max_new_tokens=2)
    assert result.retrieved == [] and result.gates == []
