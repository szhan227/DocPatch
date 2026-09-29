"""Unit tests for Section 3.3, Eq. 2-8 (docpatch/router.py)."""

import math

import torch

from docpatch.router import (
    build_multi_hot_targets,
    compute_chunk_selection_loss,
    compute_router_loss,
    compute_source_loss,
    compute_source_probabilities,
)
from ctx_to_lora.docpatch.router import RouterOutput


def test_build_multi_hot_targets_eq4():
    targets = build_multi_hot_targets([[0, 2], [1], []], num_candidates=3)
    assert torch.allclose(targets[0], torch.tensor([0.5, 0.0, 0.5]))
    assert torch.allclose(targets[1], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(targets[2], torch.tensor([0.0, 0.0, 0.0]))


def test_chunk_selection_loss_matches_hand_computation_eq5():
    # p = softmax([0, 0, 0]) = uniform 1/3 each; single ground-truth at index 1.
    chunk_gates = torch.tensor([[1 / 3, 1 / 3, 1 / 3]])
    targets = build_multi_hot_targets([[1]], num_candidates=3)
    loss = compute_chunk_selection_loss(chunk_gates, targets)
    assert math.isclose(loss.item(), -math.log(1 / 3), rel_tol=1e-5)


def test_chunk_selection_loss_multi_hot_averages_over_targets():
    chunk_gates = torch.tensor([[0.5, 0.3, 0.2]])
    targets = build_multi_hot_targets([[0, 1]], num_candidates=3)  # y = [0.5, 0.5, 0]
    loss = compute_chunk_selection_loss(chunk_gates, targets)
    expected = -(0.5 * math.log(0.5) + 0.5 * math.log(0.3))
    assert math.isclose(loss.item(), expected, rel_tol=1e-5)


def test_source_probabilities_eq6():
    chunk_gates = torch.tensor([[0.5, 0.3, 0.2]])
    source_ids = torch.tensor([[0, 0, 1]])
    probs = compute_source_probabilities(chunk_gates, source_ids, num_sources=2)
    assert torch.allclose(probs, torch.tensor([[0.8, 0.2]]), atol=1e-6)


def test_source_loss_eq7():
    probs = torch.tensor([[0.8, 0.2]])
    loss = compute_source_loss(probs, torch.tensor([0]))
    assert math.isclose(loss.item(), -math.log(0.8), rel_tol=1e-5)


def test_router_loss_combines_with_lambda_eq8():
    chunk_gates = torch.tensor([[0.5, 0.3, 0.2]])
    chunk_logits = torch.log(chunk_gates)  # consistent logits for this gate
    output = RouterOutput(source_logits=chunk_logits, chunk_logits=chunk_logits, chunk_gates=chunk_gates, source_gates=chunk_gates)
    source_ids = torch.tensor([[0, 0, 1]])
    targets = build_multi_hot_targets([[0]], num_candidates=3)
    losses = compute_router_loss(output, targets, source_ids, torch.tensor([0]), num_sources=2, source_loss_weight=0.5)
    chunk_loss = compute_chunk_selection_loss(chunk_gates, targets)
    source_probs = compute_source_probabilities(chunk_gates, source_ids, 2)
    source_loss = compute_source_loss(source_probs, torch.tensor([0]))
    assert math.isclose(losses["total"].item(), (chunk_loss + 0.5 * source_loss).item(), rel_tol=1e-5)
