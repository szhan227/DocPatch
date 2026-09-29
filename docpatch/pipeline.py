"""End-to-end DocPatch inference pipeline (Section 3.6, closing summary):

    q -> knowledge routing -> conflict resolution -> LoRA composition
      -> frozen LLM + reasoning adapter -> answer

This wires together every module in the package:
  1. ``docpatch.query_encoder.QueryEncoder``     -- e_q = P_q(Pool(F_LLM(q)))         (A.2)
  2. ``docpatch.router.DocPatchRouter``           -- s_i = R(e_q, z_i), p_i = softmax  (Eq. 2-3)
  3. ``docpatch.conflict_resolution``             -- m_i, g_i                          (Eq. 9-10)
  4. ``docpatch.reasoning_adapter.ReasoningAdapter`` -- Delta W_know + Delta W_reason  (Eq. 11-14)
  5. the frozen base LLM's own ``generate`` -- answer, from a question-only prompt
     (Section 3.1: "without reintroducing the original document text into the
     context window").
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from docpatch.composition import compose
from docpatch.config import DocPatchConfig
from docpatch.conflict_resolution import resolve_conflicts
from docpatch.knowledge_bank import BankRecord
from docpatch.query_encoder import QueryEncoder
from docpatch.reasoning_adapter import ReasoningAdapter, apply_reasoning_and_knowledge
from docpatch.router import DocPatchRouter, topk_chunk_indices
from docpatch.value_types import RetrievedKnowledge


@dataclass
class DocPatchAnswer:
    text: str
    retrieved: list[RetrievedKnowledge]
    gates: list[float]


class DocPatchPipeline:
    def __init__(
        self,
        model,  # ctx_to_lora.modeling.hypernet.ModulatedPretrainedModel (frozen base LLM + D2L hypernet)
        tokenizer,
        query_encoder: QueryEncoder,
        router: DocPatchRouter,
        lora_projector,  # learned P_LoRA (Appendix A.3), trained jointly with the router
        reasoning_adapter: ReasoningAdapter,
        config: DocPatchConfig | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.query_encoder = query_encoder
        self.router = router
        self.router_lora_projector = lora_projector
        self.reasoning_adapter = reasoning_adapter
        self.config = config or DocPatchConfig()
        # Note: the model's linear layers are only patched (patch_lora_forward)
        # inside answer(), because the query encoder in route() runs the same
        # base LLM and needs its layers unpatched.

    @torch.no_grad()
    def route(self, question: str, bank: list[BankRecord]) -> list[RetrievedKnowledge]:
        """Sections 3.2-3.3: score every knowledge LoRA in ``bank`` against q,
        return the top-k as ``RetrievedKnowledge`` with routing weight alpha_i."""

        records = bank
        if not records:
            return []
        device = next(self.router.parameters()).device
        e_q = self.query_encoder([question]).to(device)  # [1, hidden]
        z = torch.stack([r.sketch for r in records], dim=0).to(device=device, dtype=torch.float32)
        z = self.router_lora_projector(z)  # projected into routing space
        # Source ids by order of first appearance (same convention as router training).
        vocab: dict[str, int] = {}
        source_ids = torch.tensor(
            [[vocab.setdefault(r.source_id, len(vocab)) for r in records]], dtype=torch.long, device=device
        )
        output = self.router(e_q, z.unsqueeze(0), source_ids)
        indices, values = topk_chunk_indices(output.chunk_gates, top_k=self.config.router.top_k)
        indices = indices[0].tolist()
        values = values[0].tolist()
        # Renormalize alpha_i over the retrieved top-k set C_q, matching Eq. 3's
        # softmax being defined over whatever candidate set is under consideration.
        denom = sum(values) or 1.0
        retrieved = []
        for idx, val in zip(indices, values):
            record = records[idx]
            retrieved.append(
                RetrievedKnowledge(
                    cache_key=record.cache_key,
                    lora_path=record.lora_path,
                    metadata=record.metadata,
                    router_score=float(output.chunk_logits[0, idx]),
                    alpha=val / denom,
                    lora_tree=record.lora_tree,
                )
            )
        return retrieved

    def _with_hypernet_bias(self, tree: dict) -> dict:
        """Append the hypernetwork's learned head-bias LoRA block, once, unweighted.

        Doc-to-LoRA's ``combine_lora`` adds this extra rank block to every context's
        LoRA when ``use_bias`` is set (as in the released checkpoints). It is a constant
        of the hypernetwork, not of any one text, so it is added once here rather than
        stored per bank record (which would count it once per retrieved LoRA). With a
        single text in the bank this makes the composed LoRA identical to plain D2L's.
        """

        if not self.model.hypernet.config.use_bias:
            return tree
        bias = self.model.hypernet.get_head_bias()
        out = {}
        for module, weights in tree.items():
            out[module] = {}
            for key in ("A", "B"):
                w = weights[key]  # [1, n_layers, rank, dim]
                b = bias[module][key].detach().to(device=w.device, dtype=w.dtype)  # [n_layers, r, dim]
                out[module][key] = torch.cat([w, b.unsqueeze(0)], dim=-2)
        return out

    @torch.no_grad()
    def answer(
        self,
        question: str,
        bank: list[BankRecord],
        max_new_tokens: int = 64,
        composition_mode: str | None = None,
    ) -> DocPatchAnswer:
        """Answer ``question`` using the LoRAs in ``bank`` (a list of ``BankRecord``)."""

        retrieved = self.route(question, bank)
        gates = resolve_conflicts(retrieved, self.config.conflict_resolution)
        mode = composition_mode or self.config.composition.mode
        knowledge_tree = compose(retrieved, gates, mode=mode) if retrieved else None
        if knowledge_tree is not None:
            knowledge_tree = self._with_hypernet_bias(knowledge_tree)
        ctx_lora = " ".join([e.text for e in bank])

        use_mode = "d2l"
        try:
            if use_mode == "route":
                self.model.patch_lora_forward()
                apply_reasoning_and_knowledge(self.model.base_model, self.reasoning_adapter, knowledge_tree)
            elif use_mode == "d2l":
                self.model.internalize(ctx_lora)
            else:
                question = f"Given that: {ctx_lora}\nPlease tell me: " + question
            
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)
            # The chat template already contains BOS; don't add a second one.
            # inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            with torch.inference_mode():
                # output_ids = self.model.base_model.generate(
                #     **inputs, max_new_tokens=max_new_tokens, do_sample=False
                # )
                output_ids = self.model.generate(input_ids=prompt, max_new_tokens=max_new_tokens)
                
            # chat = [{"role": "user", "content": "Who founded the Zorblax Institute?"}]
            # chat_ids = tokenizer.apply_chat_template(
            #     chat,
            #     add_special_tokens=False,
            #     return_attention_mask=False,
            #     add_generation_prompt=False,
            #     return_tensors="pt",
            # ).to(model.device)


            # outputs = model.generate(input_ids=chat_ids, max_new_tokens=512)                            
                
                
                
                
            text = self.tokenizer.decode(
                output_ids[0, prompt.shape[1] :], skip_special_tokens=True
            ).strip()
        finally:
            self.model.reset()
        return DocPatchAnswer(text=text, retrieved=retrieved, gates=gates)

    @property
    def router_lora_projector(self):
        """Learned P_LoRA (Appendix A.3), trained jointly with the router.

        Exposed as a settable attribute so ``DocPatchPipeline`` can be built
        either with a freshly-initialized projector (smoke tests) or a
        trained checkpoint loaded by the caller.
        """

        return self._lora_projector

    @router_lora_projector.setter
    def router_lora_projector(self, projector) -> None:
        self._lora_projector = projector
