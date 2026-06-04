from src.deprecated.ensemble import _bb_cfg


def test_bb_cfg_sets_model_without_mutating_original():
    cfg = {"model": {"name": "orig", "img_size": 256}, "batch_size": 32, "cv": {"n_folds": 4}}
    bb = {"name": "sig", "img_size": 378, "pool": "default", "aug_views": 1}

    c = _bb_cfg(cfg, bb)
    assert c["model"] == {"name": "sig", "img_size": 378}
    assert c["batch_size"] == 32
    # original cfg is not mutated
    assert cfg["model"] == {"name": "orig", "img_size": 256}
