"""Appendix A.2 -- query representation extraction.

    H_q = F_LLM(q)                                (Eq. 26, frozen base LLM)
    h_q = Pool({H_{q,t} : t in T_q})               (Eq. 27, non-padding tokens)
    e_q = P_q(h_q)                                 (Eq. 28, learned MLP projector)

The base LLM stays frozen throughout (including at router-training time);
only ``QueryProjector`` (P_q) is trained. This mirrors the frozen backbone +
tokenizer loading convention used across the repo
(``ModulatedPretrainedModel.from_state_dict`` /
``run_eval.py``/``precompute_docpatch_chunk_loras.py``), so ``QueryEncoder``
can be constructed directly from an already-loaded D2L
``ModulatedPretrainedModel`` (its ``.base_model`` + tokenizer) without a
second copy of the backbone.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class QueryProjector(nn.Module):
    """P_q: maps pooled frozen-LLM hidden states into the routing space."""

    def __init__(self, llm_hidden_size: int, routing_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or max(routing_dim, llm_hidden_size // 4)
        self.net = nn.Sequential(
            nn.LayerNorm(llm_hidden_size),
            nn.Linear(llm_hidden_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, routing_dim),
        )

    def forward(self, pooled_hidden: Tensor) -> Tensor:
        return self.net(pooled_hidden)


class QueryEncoder(nn.Module):
    """E(q) = P_q(Pool(F_LLM(q))) end to end, given a frozen causal LM + tokenizer."""

    def __init__(
        self,
        base_model: nn.Module,
        tokenizer,
        projector: QueryProjector,
        pooling: str = "mean",
        max_length: int = 64,
    ):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.projector = projector
        if pooling not in {"mean", "last"}:
            raise ValueError("pooling must be 'mean' or 'last'")
        self.pooling = pooling
        self.max_length = max_length
        for param in self.base_model.parameters():
            param.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.base_model.parameters()).device

    @torch.no_grad()
    def _pooled_hidden(self, questions: list[str]) -> Tensor:
        batch = self.tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.base_model(**batch, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # H_q, [batch, seq, d]
        mask = batch["attention_mask"].to(hidden.dtype).unsqueeze(-1)  # T_q indicator
        if self.pooling == "mean":
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        # "last": last non-padding token position per sequence.
        lengths = batch["attention_mask"].sum(dim=1) - 1
        return hidden[torch.arange(hidden.shape[0]), lengths]

    def forward(self, questions: list[str]) -> Tensor:
        pooled = self._pooled_hidden(questions)
        return self.projector(pooled.to(next(self.projector.parameters()).dtype))

    def encode(self, questions: list[str]) -> Tensor:
        return self.forward(questions)
