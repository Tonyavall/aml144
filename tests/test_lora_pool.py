import pytest
import torch
import torch.nn as nn

from src.models.lora_train import LoraClassifier, head_input_dim, pool_tokens


def test_pool_tokens_cls_meanpatch_shape():
    b, t, d = 2, 5, 8
    out = pool_tokens(torch.zeros(b, t, d), num_prefix=1, pool_mode="cls_meanpatch")
    assert out.shape == (b, 2 * d)


def test_pool_tokens_avg_means_all_tokens():
    b, t, d = 2, 4, 8
    tokens = torch.arange(b * t * d, dtype=torch.float32).reshape(b, t, d)
    out = pool_tokens(tokens, num_prefix=0, pool_mode="avg")
    assert out.shape == (b, d)
    assert torch.allclose(out, tokens.mean(dim=1))


def test_pool_tokens_unknown_mode_raises():
    with pytest.raises(ValueError):
        pool_tokens(torch.zeros(1, 3, 4), num_prefix=0, pool_mode="bogus")


def test_head_input_dim_per_mode():
    assert head_input_dim("cls_meanpatch", 1024) == 2048
    assert head_input_dim("avg", 1152) == 1152
    with pytest.raises(ValueError):
        head_input_dim("bogus", 10)


class _StubBackbone(nn.Module):
    def __init__(self, tokens):
        super().__init__()
        self._tokens = tokens

    def forward_features(self, x):
        return self._tokens


def test_lora_classifier_avg_mode_uses_all_tokens():
    b, t, d, c = 2, 4, 8, 3
    tokens = torch.zeros(b, t, d)
    model = LoraClassifier(_StubBackbone(tokens), nn.Linear(d, c), 0, pool_mode="avg")
    out = model(torch.zeros(b, 3, 16, 16))
    assert out.shape == (b, c)

    model.backbone._tokens = tokens.clone()
    model.backbone._tokens[:, 0, :] = 1.0
    out2 = model(torch.zeros(b, 3, 16, 16))
    assert not torch.equal(out, out2)
