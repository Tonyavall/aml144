import pytest
import torch
import torch.nn as nn

from src.models.lora_train import CosineHead, build_head


def test_cosine_head_output_shape():
    head = CosineHead(in_dim=8, n_classes=3, scale_init=10.0)
    out = head(torch.randn(4, 8))
    assert out.shape == (4, 3)


def test_cosine_head_is_invariant_to_feature_norm():
    # cosine head l2-normalizes features, so scaling the input must not change logits
    head = CosineHead(in_dim=8, n_classes=3)
    feat = torch.randn(4, 8)
    assert torch.allclose(head(feat), head(feat * 5.0), atol=1e-5)


def test_cosine_head_scale_is_learnable():
    head = CosineHead(in_dim=8, n_classes=3, scale_init=7.0)
    assert head.scale.requires_grad
    assert abs(float(head.scale) - 7.0) < 1e-6


def test_build_head_default_is_linear():
    h = build_head("linear", 8, 3)
    assert isinstance(h, nn.Linear)
    assert h.out_features == 3


def test_build_head_cosine():
    h = build_head("cosine", 8, 3, 5.0)
    assert isinstance(h, CosineHead)
    assert abs(float(h.scale) - 5.0) < 1e-6


def test_build_head_unknown_raises():
    with pytest.raises(ValueError):
        build_head("bogus", 8, 3)
