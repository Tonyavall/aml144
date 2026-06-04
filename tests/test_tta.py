import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.lora_train import eval_probs


class _FlipAwareModel(nn.Module):
    # logits depend on the left half of the image, so an hflip changes the output.
    # lets us verify that tta actually averages distinct views.
    def forward(self, x):
        left = x[:, :, :, : x.shape[3] // 2].mean(dim=(1, 2, 3))
        return torch.stack([left, -left], dim=1)


def _loader(imgs):
    labels = torch.zeros(len(imgs), dtype=torch.long)
    return DataLoader(TensorDataset(imgs, labels), batch_size=2)


def test_eval_probs_identity_matches_single_forward():
    model = _FlipAwareModel()
    imgs = torch.randn(5, 3, 8, 8)
    out = eval_probs(model, _loader(imgs), "cpu", ["identity"])
    expected = model(imgs).softmax(dim=1).numpy()
    assert np.allclose(out, expected, atol=1e-5)


def test_eval_probs_hflip_averages_two_views():
    model = _FlipAwareModel()
    imgs = torch.randn(4, 3, 8, 8)
    out = eval_probs(model, _loader(imgs), "cpu", ["identity", "hflip"])
    p_id = model(imgs).softmax(dim=1)
    p_hf = model(torch.flip(imgs, dims=[3])).softmax(dim=1)
    expected = ((p_id + p_hf) / 2).numpy()
    assert np.allclose(out, expected, atol=1e-5)


def test_eval_probs_flip_invariant_model_tta_equals_identity():
    class _Invariant(nn.Module):
        def forward(self, x):
            s = x.mean(dim=(1, 2, 3))
            return torch.stack([s, -s], dim=1)

    model = _Invariant()
    imgs = torch.randn(3, 3, 8, 8)
    id_out = eval_probs(model, _loader(imgs), "cpu", ["identity"])
    tta_out = eval_probs(model, _loader(imgs), "cpu", ["identity", "hflip"])
    assert np.allclose(id_out, tta_out, atol=1e-5)


def test_eval_probs_unknown_view_raises():
    model = _FlipAwareModel()
    with pytest.raises(ValueError):
        eval_probs(model, _loader(torch.randn(2, 3, 8, 8)), "cpu", ["bogus"])
