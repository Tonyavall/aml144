import torch
import torch.nn as nn

from src.models.backbone import _embed


class _Stub(nn.Module):
    # forward_features returns fixed tokens; forward returns a fixed pooled vector
    def __init__(self, tokens, pooled):
        super().__init__()
        self._tokens = tokens
        self._pooled = pooled

    def forward_features(self, x):
        return self._tokens

    def forward(self, x):
        return self._pooled


def test_embed_cls_meanpatch_concats_cls_and_patches():
    b, t, d = 2, 5, 8
    tokens = torch.zeros(b, t, d)
    stub = _Stub(tokens, torch.zeros(b, 4))

    out = _embed(stub, torch.zeros(b, 3, 16, 16), num_prefix=1, pool="cls_meanpatch")
    assert out.shape == (b, 2 * d)

    tokens2 = tokens.clone()
    tokens2[:, 1:, :] = 1.0
    stub._tokens = tokens2
    out2 = _embed(stub, torch.zeros(b, 3, 16, 16), num_prefix=1, pool="cls_meanpatch")
    assert not torch.equal(out, out2)

    # cls token (index 0) must also participate, not just the patch tokens
    tokens3 = tokens.clone()
    tokens3[:, 0, :] = 1.0
    stub._tokens = tokens3
    out3 = _embed(stub, torch.zeros(b, 3, 16, 16), num_prefix=1, pool="cls_meanpatch")
    assert not torch.equal(out, out3)


def test_embed_default_returns_model_pooled_output():
    b, d = 2, 8
    pooled = torch.randn(b, 16)
    stub = _Stub(torch.zeros(b, 5, d), pooled)

    out = _embed(stub, torch.zeros(b, 3, 16, 16), num_prefix=1, pool="default")
    assert out.shape == (b, 16)
    assert torch.equal(out, pooled)
