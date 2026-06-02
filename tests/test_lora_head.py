import torch
import torch.nn as nn

from src.models.lora_train import LoraClassifier


class _StubBackbone(nn.Module):
    # returns a fixed (b, t, d) token tensor regardless of input
    def __init__(self, tokens):
        super().__init__()
        self._tokens = tokens

    def forward_features(self, x):
        return self._tokens


def test_lora_classifier_uses_cls_and_mean_patch():
    b, t, d, c = 2, 5, 8, 3
    num_prefix = 1
    tokens = torch.zeros(b, t, d)
    backbone = _StubBackbone(tokens)
    head = nn.Linear(2 * d, c)
    model = LoraClassifier(backbone, head, num_prefix)

    out = model(torch.zeros(b, 3, 16, 16))
    assert out.shape == (b, c)

    # changing only the patch tokens must change the output -> patches are used
    backbone._tokens = tokens.clone()
    backbone._tokens[:, num_prefix:, :] = 1.0
    out2 = model(torch.zeros(b, 3, 16, 16))
    assert not torch.equal(out, out2)
