"""Unit tests for Appendix A.3 (docpatch/lora_sketch.py)."""

import torch

from docpatch.lora_sketch import FixedRandomProjections, LoRAFeatureProjector, compute_lora_sketch

MODULE = "down_proj"


def test_projections_are_deterministic_given_seed():
    proj1 = FixedRandomProjections(sketch_dim=4, seed=0)
    proj2 = FixedRandomProjections(sketch_dim=4, seed=0)
    p_in1, p_out1 = proj1.get(MODULE, d_in=8, d_out=6, device="cpu", dtype=torch.float32)
    p_in2, p_out2 = proj2.get(MODULE, d_in=8, d_out=6, device="cpu", dtype=torch.float32)
    assert torch.equal(p_in1, p_in2)
    assert torch.equal(p_out1, p_out2)


def test_different_seed_gives_different_projections():
    proj1 = FixedRandomProjections(sketch_dim=4, seed=0)
    proj2 = FixedRandomProjections(sketch_dim=4, seed=1)
    p_in1, _ = proj1.get(MODULE, 8, 6, "cpu", torch.float32)
    p_in2, _ = proj2.get(MODULE, 8, 6, "cpu", torch.float32)
    assert not torch.equal(p_in1, p_in2)


def test_sketch_layer_matches_factored_definition():
    proj = FixedRandomProjections(sketch_dim=4, seed=0)
    a = torch.randn(2, 3, 8)  # [..., r, d_in]
    b = torch.randn(2, 3, 6)  # [..., r, d_out]
    sketch = proj.sketch_layer(a, b, MODULE)
    assert sketch.shape == (2, 4, 4)

    p_in, p_out = proj.get(MODULE, 8, 6, "cpu", torch.float32)
    # Direct definition: S = (P_out^T B_paper)(A P_in) where B_paper = B^T (repo
    # convention), i.e. S[i] = (B[i] @ P_out)^T @ (A[i] @ P_in).
    for i in range(2):
        b_proj = b[i] @ p_out
        a_proj = a[i] @ p_in
        expected = b_proj.transpose(-1, -2) @ a_proj
        assert torch.allclose(sketch[i], expected, atol=1e-5)


def test_compute_lora_sketch_pools_over_layers_and_sorts_modules():
    proj = FixedRandomProjections(sketch_dim=4, seed=0)
    tree = {
        "up_proj": {"A": torch.randn(3, 4, 8), "B": torch.randn(3, 4, 5)},
        "down_proj": {"A": torch.randn(3, 4, 8), "B": torch.randn(3, 4, 6)},
    }
    feature = compute_lora_sketch(tree, proj)
    # 2 modules * (sketch_dim ** 2) flattened.
    assert feature.shape == (2 * 4 * 4,)


def test_projector_maps_into_routing_space():
    projector = LoRAFeatureProjector(sketch_feature_dim=32, routing_dim=16)
    features = torch.randn(5, 32)
    out = projector(features)
    assert out.shape == (5, 16)
