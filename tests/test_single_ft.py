import json

import numpy as np
import pandas as pd

import src.single_ft as sft
from src.single_ft import decision_outcome


def test_decision_win_when_single_at_or_above_ensemble():
    assert decision_outcome(0.95, 0.002, 0.94, 0.002) == "win"


def test_fold_series_builds_per_fold_metric_lists():
    fh = {
        0: [
            {"epoch": 0, "train_loss": 1.0, "train_acc": 0.5, "val_loss": 0.9, "val_acc": 0.6},
            {"epoch": 1, "train_loss": 0.8, "train_acc": 0.7, "val_loss": 0.7, "val_acc": 0.75},
        ],
        1: [
            {"epoch": 0, "train_loss": 1.1, "train_acc": 0.4, "val_loss": 1.0, "val_acc": 0.55},
        ],
    }
    s = sft._fold_series(fh)
    assert set(s) == {"fold 0", "fold 1"}
    assert s["fold 0"]["train_loss"] == [1.0, 0.8]
    assert s["fold 0"]["val_acc"] == [0.6, 0.75]
    assert s["fold 1"]["train_acc"] == [0.4]


def test_decision_tie_within_combined_std():
    # gap 0.0020 <= combined std sqrt(0.002^2 + 0.002^2) ~ 0.00283 -> tie
    assert decision_outcome(0.9385, 0.002, 0.9405, 0.002) == "tie"


def test_decision_loss_when_far_below():
    assert decision_outcome(0.92, 0.002, 0.9405, 0.002) == "loss"


def test_single_ft_main_writes_outputs(tmp_path, monkeypatch):
    # 3 classes x 4 images = 12 train rows; 5 test images. training/eval are
    # monkeypatched, so the dummy files only need to exist (not be valid images).
    train = tmp_path / "train"
    for c in range(3):
        d = train / str(c)
        d.mkdir(parents=True)
        for i in range(4):
            (d / f"{i}.jpg").write_bytes(b"x")
    test = tmp_path / "test"
    test.mkdir()
    for i in range(5):
        (test / f"{i}.jpg").write_bytes(b"x")

    n_classes, n_train = 3, 12
    cfg = {
        "data": {"train_dir": str(train), "test_dir": str(test)},
        "seed": 42,
        "output_dir": str(tmp_path / "outputs"),
        "cv": {"n_folds": 4},
        "single_ft": {
            "deploy_seed": 42,
            "oof_seeds": [42, 43],
            "output_submission": "submission_siglip2.csv",
            "member": {
                "name": "m", "img_size": 378, "pool_mode": "avg",
                "lora": {"batch_size": 2, "head_type": "cosine",
                         "class_balanced": True, "tta_views": ["identity", "hflip"]},
            },
        },
    }
    rng = np.random.default_rng(0)

    def fake_oof(member, cfg, paths, y, mean, std, device, seed, capture_history=False):
        oof = rng.random((len(y), n_classes))
        folds = [{"lora": {}, "head": {}} for _ in range(cfg["cv"]["n_folds"])]
        return oof, folds, n_classes, {}

    monkeypatch.setattr(sft, "load_config", lambda p: cfg)
    monkeypatch.setattr(sft, "collect_metadata", lambda cfg, extra=None: dict(extra or {}))
    monkeypatch.setattr(
        sft, "load_backbone",
        lambda name, img, dev: (None, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    )
    monkeypatch.setattr(sft, "member_oof_and_bundle", fake_oof)
    monkeypatch.setattr(sft, "member_val_oof", lambda *a, **k: rng.random((n_train, n_classes)))
    monkeypatch.setattr(sft, "member_test_probs", lambda *a, **k: rng.random((5, n_classes)))

    sft.main("ignored.yaml")

    out = tmp_path / "outputs"
    sub = pd.read_csv(out / "submission_siglip2.csv")
    assert sub["ID"].tolist() == ["0.jpg", "1.jpg", "2.jpg", "3.jpg", "4.jpg"]
    assert len(sub) == 5

    metrics = json.loads((out / "single_ft" / "metrics.json").read_text())
    for key in [
        "seed_oof_acc", "oof_mean", "oof_std", "tta_oof_acc",
        "identity_oof_acc", "tta_delta", "decision_outcome", "recommended_deploy",
    ]:
        assert key in metrics

    assert (out / "single_ft" / "single_ft_bundle.pkl").exists()
    # non-destructive: the ensemble/baseline submissions are not created or touched
    assert not (out / "submission.csv").exists()
    assert not (out / "submission_ensemble_ft.csv").exists()
