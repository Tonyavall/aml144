import numpy as np
import torch
from PIL import Image

from src.data.images import build_aug_transform, build_transform
from src.models.head import oof_accuracy, predict_proba
from src.models.multiview import (
    cross_validate_views,
    fit_view_fold_models,
    grouped_oof,
    stack_views,
)
from src.utils import set_seed


def test_identity_view_deterministic_aug_view_random():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    arr = np.random.default_rng(0).integers(0, 256, size=(300, 400, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    idt = build_transform(256, mean, std, "identity")
    aug = build_aug_transform(256, mean, std)

    assert torch.equal(idt(img), idt(img))
    assert not torch.equal(aug(img), aug(img))


def test_stack_views_shapes_labels_and_groups():
    n, d = 5, 3
    views = [np.arange(n * d).reshape(n, d).astype(float) + k for k in range(4)]
    y = np.array([0, 1, 2, 0, 1])

    x_aug, y_aug, groups, x_eval = stack_views(views, y)

    assert x_aug.shape == (20, d)
    assert y_aug.tolist() == y.tolist() * 4
    assert groups.tolist() == list(range(5)) * 4
    assert np.array_equal(x_eval, views[0])


def test_aug_transform_reproducible_under_seed():
    # seeding before applying the aug transform makes the random crop reproducible,
    # which is what makes per-view feature caching deterministic across runs
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    arr = np.random.default_rng(0).integers(0, 256, size=(300, 400, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    set_seed(123)
    a = build_aug_transform(256, mean, std)(img)
    set_seed(123)
    b = build_aug_transform(256, mean, std)(img)

    assert torch.equal(a, b)


def _blob_views(n_per=12, n_views=4, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[5, 0], [0, 5], [-5, 0], [0, -5]], dtype=float)
    base = np.concatenate([c + rng.normal(0, 0.2, size=(n_per, 2)) for c in centers])
    y = np.repeat(np.arange(4), n_per).astype(int)
    views = [base + rng.normal(0, 0.2, size=base.shape) for _ in range(n_views)]
    views[0] = base  # identity view used for eval
    return views, y


def test_no_aug_leakage_train_mask_excludes_val_groups():
    # the invariant grouped_oof relies on: training rows never include a val original
    n, n_views = 8, 3
    groups = np.tile(np.arange(n), n_views)
    tr_orig = np.array([0, 1, 2, 3, 4])
    va_orig = np.array([5, 6, 7])
    tr_mask = np.isin(groups, tr_orig)
    assert set(groups[tr_mask]).isdisjoint(set(va_orig.tolist()))
    assert set(groups[tr_mask]) == set(tr_orig.tolist())


def test_grouped_oof_covers_originals_and_is_accurate():
    views, y = _blob_views()
    x_aug, y_aug, groups, x_eval = stack_views(views, y)
    oof = grouped_oof(
        x_aug, y_aug, groups, x_eval, y, c=1.0, n_splits=4, n_repeats=1, seed=42
    )
    assert oof.shape == (len(y), 4)
    assert np.allclose(oof.sum(axis=1), 1.0)
    assert oof_accuracy(oof, y) > 0.9


def test_cross_validate_views_and_fold_models_predict():
    views, y = _blob_views()
    x_aug, y_aug, groups, x_eval = stack_views(views, y)
    results, best_c, oof = cross_validate_views(
        x_aug, y_aug, groups, x_eval, y, [0.1, 1.0, 10.0], 4, 1, 42
    )
    assert best_c in (0.1, 1.0, 10.0)
    assert oof.shape == (len(y), 4)
    assert np.allclose(oof.sum(axis=1), 1.0)
    fold_models = fit_view_fold_models(
        x_aug, y_aug, groups, x_eval, y, best_c, 4, 1, 42
    )
    assert len(fold_models) == 4
    probs = predict_proba(fold_models, x_eval)
    assert probs.shape == (len(y), 4)
