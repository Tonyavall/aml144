import numpy as np
import torch
from PIL import Image

from src.models.lora_train import (
    build_lora_transforms,
    lora_target_modules,
    make_lr_lambda,
    stratified_split,
)


def test_stratified_split_covers_and_separates():
    # 5 classes, 20 each = 100 samples
    y = np.repeat(np.arange(5), 20)
    train_idx, val_idx = stratified_split(y, val_frac=0.2, seed=42)

    # disjoint and complete
    assert set(train_idx).isdisjoint(set(val_idx))
    assert sorted(np.concatenate([train_idx, val_idx])) == list(range(100))

    # ~20% in val, every class present on both sides
    assert len(val_idx) == 20
    assert set(y[train_idx]) == set(range(5))
    assert set(y[val_idx]) == set(range(5))


def test_stratified_split_is_deterministic():
    y = np.repeat(np.arange(5), 20)
    a = stratified_split(y, 0.2, 7)
    b = stratified_split(y, 0.2, 7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_lora_target_modules_last_four_of_24():
    mods = lora_target_modules(total_blocks=24, n_last=4)
    assert mods == [
        "blocks.20.attn.qkv",
        "blocks.20.attn.proj",
        "blocks.21.attn.qkv",
        "blocks.21.attn.proj",
        "blocks.22.attn.qkv",
        "blocks.22.attn.proj",
        "blocks.23.attn.qkv",
        "blocks.23.attn.proj",
    ]


def test_lr_lambda_warmup_then_cosine():
    fn = make_lr_lambda(total_steps=100, warmup_steps=10)
    assert fn(0) == 0.0  # starts at 0
    assert abs(fn(10) - 1.0) < 1e-9  # peaks at end of warmup
    assert 0.0 <= fn(100) <= 1e-6  # decays to ~0 at the end
    # monotonic decay after warmup
    assert fn(20) > fn(50) > fn(90)


def test_eval_transform_deterministic_train_random():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf, eval_tf = build_lora_transforms(256, mean, std)
    # non-uniform content so random crops actually differ (a solid color would not)
    arr = np.random.default_rng(0).integers(0, 256, size=(300, 400, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    e1, e2 = eval_tf(img), eval_tf(img)
    assert isinstance(e1, torch.Tensor) and e1.shape == (3, 256, 256)
    assert torch.equal(e1, e2)  # deterministic

    t1, t2 = train_tf(img), train_tf(img)
    assert t1.shape == (3, 256, 256)
    assert not torch.equal(t1, t2)  # random aug differs run to run
