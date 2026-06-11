"""tests for the final blend: hf extraction pure parts, probes, class names."""

import csv
from pathlib import Path

import numpy as np

from src.final_blend import load_class_names, probe_member
from src.models.hf_siglip2 import zero_shot_probs
from src.models.lora_members import shared_folds


def test_zero_shot_probs_softmax_rows():
    img = np.eye(3, 5, dtype=np.float32)
    txt = np.eye(4, 5, dtype=np.float32)
    p = zero_shot_probs(img, txt, 100.0, -10.0)
    assert p.shape == (3, 4)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p[0].argmax() == 0


def test_zero_shot_probs_bias_does_not_change_argmax():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(6, 8)).astype(np.float32)
    txt = rng.normal(size=(4, 8)).astype(np.float32)
    a = zero_shot_probs(img, txt, 50.0, 0.0).argmax(axis=1)
    b = zero_shot_probs(img, txt, 50.0, -12.9).argmax(axis=1)
    assert (a == b).all()


def _blobs(n_per, k, d, seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * 5
    x = np.concatenate([centers[i] + rng.normal(size=(n_per, d)) for i in range(k)])
    y = np.repeat(np.arange(k), n_per)
    return x.astype(np.float32), y


def test_probe_member_oof_aligned_and_accurate():
    x, y = _blobs(12, 4, 8, 0)
    folds = shared_folds(y, 4, 42)
    oof, models, best_c, acc = probe_member(x, y, folds, [0.1, 1.0, 10.0])
    assert oof.shape == (48, 4)
    assert np.allclose(oof.sum(axis=1), 1.0)
    assert acc > 0.9
    assert len(models) == 4
    assert best_c in (0.1, 1.0, 10.0)


def test_load_class_names_orders_by_index(tmp_path):
    p = tmp_path / "names.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "name"])
        for i in reversed(range(5)):
            w.writerow([str(i), f"thing {i}"])
    c2i = {str(i): i for i in range(5)}
    assert load_class_names(p, c2i) == [f"thing {i}" for i in range(5)]


def test_load_class_names_raises_on_missing(tmp_path):
    p = tmp_path / "names.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "name"])
        w.writerow(["0", "thing"])
    c2i = {"0": 0, "1": 1}
    try:
        load_class_names(p, c2i)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_repo_class_names_file_valid():
    with open(Path("src/data/class_names.csv"), encoding="ascii") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 100
    assert {r["class_id"] for r in rows} == {str(i) for i in range(100)}
    assert all(r["name"].strip() for r in rows)
