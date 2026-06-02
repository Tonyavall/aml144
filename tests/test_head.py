import numpy as np

from src.models.head import (
    _fit_one,
    cross_validate,
    fit_fold_models,
    l2_normalize,
    oof_accuracy,
    predict_proba,
)


def _separable_blobs(n_per=12, seed=0):
    # four well-separated gaussian blobs, 12 points each = 48 samples
    rng = np.random.default_rng(seed)
    centers = np.array([[5, 0], [0, 5], [-5, 0], [0, -5]], dtype=float)
    x = np.concatenate([c + rng.normal(0, 0.3, size=(n_per, 2)) for c in centers])
    y = np.repeat(np.arange(4), n_per)
    return x.astype(np.float64), y


def test_l2_normalize_makes_unit_norm_rows():
    x = np.array([[3.0, 4.0], [1.0, 0.0]])
    xn = l2_normalize(x)
    assert np.allclose(np.linalg.norm(xn, axis=1), 1.0)


def test_fit_one_uses_balanced_class_weight():
    x, y = _separable_blobs()
    clf = _fit_one(x, y, c=1.0)
    assert clf.class_weight == "balanced"


def test_cross_validate_is_deterministic_and_grids_all_c():
    x, y = _separable_blobs()
    grid = [0.1, 1.0, 10.0]
    r1, c1 = cross_validate(x, y, grid, n_splits=4, n_repeats=3, seed=42)
    r2, c2 = cross_validate(x, y, grid, n_splits=4, n_repeats=3, seed=42)
    assert c1 == c2
    assert set(r1.keys()) == {"0.1", "1.0", "10.0"}


def test_fit_fold_models_oof_aligned_and_accurate():
    x, y = _separable_blobs()
    fold_models, oof = fit_fold_models(x, y, c=1.0, n_splits=4, n_repeats=3, seed=42)
    assert len(fold_models) == 4 * 3  # n_splits * n_repeats
    assert oof.shape == (len(y), 4)
    assert np.allclose(oof.sum(axis=1), 1.0)
    assert oof_accuracy(oof, y) > 0.9


def test_predict_proba_shape_and_rows_sum_to_one():
    x, y = _separable_blobs()
    fold_models, _ = fit_fold_models(x, y, c=1.0, n_splits=4, n_repeats=3, seed=42)
    probs = predict_proba(fold_models, x)
    assert probs.shape == (len(y), 4)
    assert np.allclose(probs.sum(axis=1), 1.0)
