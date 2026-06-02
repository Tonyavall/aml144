import numpy as np

from src.ensemble_ft import _member_cfg, shared_folds


def test_member_cfg_sets_backbone_and_pool_without_mutating():
    cfg = {
        "model": {"name": "orig", "img_size": 256},
        "lora": {"blocks": 4, "batch_size": 32, "img_size": 256},
    }
    member = {
        "name": "sig",
        "img_size": 378,
        "pool_mode": "avg",
        "lora": {"batch_size": 16},
    }
    c = _member_cfg(cfg, member)

    assert c["model"] == {"name": "sig", "img_size": 378}
    assert c["lora"]["pool_mode"] == "avg"
    assert c["lora"]["img_size"] == 378
    assert c["lora"]["batch_size"] == 16  # member override wins
    assert c["lora"]["blocks"] == 4  # base default kept
    # original cfg is not mutated
    assert cfg["model"] == {"name": "orig", "img_size": 256}
    assert cfg["lora"]["batch_size"] == 32
    assert "pool_mode" not in cfg["lora"]


def test_member_cfg_without_member_lora_overrides():
    cfg = {"model": {"name": "o", "img_size": 256}, "lora": {"blocks": 4, "batch_size": 32}}
    member = {"name": "aim", "img_size": 336, "pool_mode": "avg"}
    c = _member_cfg(cfg, member)
    assert c["lora"]["batch_size"] == 32
    assert c["lora"]["pool_mode"] == "avg"
    assert c["lora"]["img_size"] == 336


def test_shared_folds_aligned_and_cover_all():
    y = np.repeat(np.arange(5), 8)
    f1 = shared_folds(y, 4, 42)
    f2 = shared_folds(y, 4, 42)
    # identical across calls -> every member gets the same folds -> oof rows align
    for (_, a_va), (_, b_va) in zip(f1, f2):
        assert np.array_equal(a_va, b_va)
    # every sample is a validation row exactly once
    all_va = np.concatenate([va for _, va in f1])
    assert sorted(all_va.tolist()) == list(range(len(y)))
