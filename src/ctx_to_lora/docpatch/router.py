from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class RouterOutput:
    source_logits: Tensor
    chunk_logits: Tensor
    chunk_gates: Tensor
    source_gates: Tensor


@dataclass
class RankRouterOutput:
    router_output: RouterOutput
    rank_logits: Tensor
    rank_gates: Tensor


class SourceAwareRouter(nn.Module):
    """Query-conditioned router over cached LoRA candidates.

    Args:
        query_embeds: FloatTensor [batch, hidden_size]
        chunk_embeds: FloatTensor [batch, num_candidates, hidden_size].
            In the intended DocPatch setup these are projected LoRA features,
            not original chunk-text embeddings.
        source_ids: LongTensor [batch, num_candidates], source index per LoRA
        chunk_mask: BoolTensor [batch, num_candidates], True for valid LoRAs

    Returns:
        source logits/gates and chunk logits/gates. The gates are probabilities
        over valid LoRAs/sources and can later be used for top-k or weighted
        LoRA composition.
    """

    def __init__(
        self,
        hidden_size: int,
        router_hidden_size: int = 512,
        num_sources: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_sources = num_sources
        self.source_embedding = nn.Embedding(num_sources, hidden_size)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, router_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_size, 1),
        )

    def forward(
        self,
        query_embeds: Tensor,
        chunk_embeds: Tensor,
        source_ids: Tensor,
        chunk_mask: Tensor | None = None,
    ) -> RouterOutput:
        if query_embeds.ndim != 2:
            raise ValueError("query_embeds must have shape [batch, hidden_size]")
        if chunk_embeds.ndim != 3:
            raise ValueError("chunk_embeds must have shape [batch, num_chunks, hidden_size]")
        if source_ids.shape != chunk_embeds.shape[:2]:
            raise ValueError("source_ids must have shape [batch, num_chunks]")

        batch_size, num_chunks, _ = chunk_embeds.shape
        if chunk_mask is None:
            chunk_mask = torch.ones(
                batch_size,
                num_chunks,
                dtype=torch.bool,
                device=chunk_embeds.device,
            )
        source_ids = source_ids.clamp_min(0).clamp_max(self.num_sources - 1)
        source_embeds = self.source_embedding(source_ids)
        query_expanded = query_embeds[:, None, :].expand(-1, num_chunks, -1)
        features = torch.cat(
            [
                query_expanded,
                chunk_embeds,
                source_embeds,
                query_expanded * chunk_embeds,
            ],
            dim=-1,
        )
        chunk_logits = self.scorer(features).squeeze(-1)
        chunk_logits = chunk_logits.masked_fill(~chunk_mask, torch.finfo(chunk_logits.dtype).min)
        chunk_gates = torch.softmax(chunk_logits, dim=-1)

        source_logits = torch.full(
            (batch_size, self.num_sources),
            torch.finfo(chunk_logits.dtype).min,
            dtype=chunk_logits.dtype,
            device=chunk_logits.device,
        )
        source_logits = source_logits.scatter_reduce(
            dim=1,
            index=source_ids,
            src=chunk_logits,
            reduce="amax",
            include_self=True,
        )
        # scatter_reduce is not implemented for Bool on CUDA, so reduce in int32.
        valid_sources = torch.zeros_like(source_logits, dtype=torch.int32)
        valid_sources = valid_sources.scatter_reduce(
            dim=1,
            index=source_ids,
            src=chunk_mask.to(torch.int32),
            reduce="amax",
            include_self=False,
        ).bool()
        source_logits = source_logits.masked_fill(
            ~valid_sources,
            torch.finfo(source_logits.dtype).min,
        )
        source_gates = torch.softmax(source_logits, dim=-1)
        return RouterOutput(
            source_logits=source_logits,
            chunk_logits=chunk_logits,
            chunk_gates=chunk_gates,
            source_gates=source_gates,
        )


def topk_chunk_indices(
    chunk_gates: Tensor,
    chunk_mask: Tensor | None = None,
    top_k: int = 4,
) -> tuple[Tensor, Tensor]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    scores = chunk_gates
    if chunk_mask is not None:
        scores = scores.masked_fill(~chunk_mask, -1.0)
    k = min(top_k, scores.shape[-1])
    values, indices = torch.topk(scores, k=k, dim=-1)
    return indices, values


class MeanPooledTextEmbedder(nn.Module):
    """Text embedder baseline. Not the default privacy-preserving router path."""

    def __init__(self, model, tokenizer, device: str | torch.device | None = None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        if device is not None:
            self.model.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def encode(self, texts: list[str], max_length: int = 512) -> Tensor:
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        outputs = self.model(**batch, output_hidden_states=True)
        hidden = outputs.hidden_states[-1] if outputs.hidden_states else outputs.last_hidden_state
        mask = batch["attention_mask"].to(hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class LoraFeatureRouter(SourceAwareRouter):
    """Router alias for the LoRA-feature-only design."""


class HierarchicalRankRouter(SourceAwareRouter):
    """Route first over chunk LoRAs, then over their rank-1 atoms."""

    def __init__(
        self,
        hidden_size: int,
        router_hidden_size: int = 512,
        num_sources: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__(hidden_size, router_hidden_size, num_sources, dropout)
        self.rank_scorer = nn.Sequential(
            nn.LayerNorm(hidden_size * 5),
            nn.Linear(hidden_size * 5, router_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_size, 1),
        )

    def forward_rank1(
        self,
        query_embeds: Tensor,
        chunk_embeds: Tensor,
        rank_embeds: Tensor,
        source_ids: Tensor,
        chunk_mask: Tensor | None = None,
        rank_mask: Tensor | None = None,
        top_k_ranks: int | None = None,
        temperature: float = 1.0,
    ) -> RankRouterOutput:
        """Return query-level gates over candidate rank-1 atoms.

        rank_embeds must have shape [batch, candidates, rank, hidden_size].
        The chunk logit acts as a prior for every atom belonging to that chunk.
        """

        if rank_embeds.ndim != 4:
            raise ValueError(
                "rank_embeds must have shape [batch, candidates, rank, hidden_size]"
            )
        if rank_embeds.shape[:2] != chunk_embeds.shape[:2]:
            raise ValueError("rank_embeds and chunk_embeds must share candidates")
        if rank_embeds.shape[-1] != self.hidden_size:
            raise ValueError("rank_embeds hidden size does not match router")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        router_output = self.forward(
            query_embeds, chunk_embeds, source_ids, chunk_mask
        )
        batch_size, num_candidates, rank, _ = rank_embeds.shape
        if chunk_mask is None:
            chunk_mask = torch.ones(
                batch_size,
                num_candidates,
                dtype=torch.bool,
                device=rank_embeds.device,
            )
        if rank_mask is None:
            rank_mask = chunk_mask[:, :, None].expand(-1, -1, rank)
        elif rank_mask.shape != rank_embeds.shape[:3]:
            raise ValueError("rank_mask must have shape [batch, candidates, rank]")
        rank_mask = rank_mask & chunk_mask[:, :, None]

        query = query_embeds[:, None, None, :].expand(-1, num_candidates, rank, -1)
        chunk = chunk_embeds[:, :, None, :].expand(-1, -1, rank, -1)
        features = torch.cat(
            [query, chunk, rank_embeds, query * rank_embeds, chunk * rank_embeds],
            dim=-1,
        )
        rank_logits = self.rank_scorer(features).squeeze(-1)
        rank_logits = rank_logits + router_output.chunk_logits[:, :, None]
        rank_logits = rank_logits.masked_fill(
            ~rank_mask, torch.finfo(rank_logits.dtype).min
        )

        flat_logits = rank_logits.flatten(1)
        flat_mask = rank_mask.flatten(1)
        if top_k_ranks is not None:
            if top_k_ranks <= 0:
                raise ValueError("top_k_ranks must be positive")
            k = min(top_k_ranks, flat_logits.shape[-1])
            top_indices = flat_logits.topk(k, dim=-1).indices
            top_mask = torch.zeros_like(flat_mask)
            top_mask.scatter_(1, top_indices, True)
            flat_mask = flat_mask & top_mask
        flat_logits = flat_logits.masked_fill(
            ~flat_mask, torch.finfo(flat_logits.dtype).min
        )
        rank_gates = torch.softmax(flat_logits / temperature, dim=-1).reshape_as(
            rank_logits
        )
        return RankRouterOutput(router_output, rank_logits, rank_gates)


def topk_rank1_indices(
    rank_gates: Tensor,
    top_k: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return candidate indices, rank indices, and weights for top-k atoms."""

    if rank_gates.ndim != 3:
        raise ValueError("rank_gates must have shape [batch, candidates, rank]")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    rank = rank_gates.shape[-1]
    flat = rank_gates.flatten(1)
    values, indices = flat.topk(min(top_k, flat.shape[-1]), dim=-1)
    return indices // rank, indices % rank, values
