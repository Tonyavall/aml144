import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from src.models.head import _fit_one, l2_normalize, oof_accuracy


def stack_views(views, y):
    # views: list of (n, d) arrays where views[0] is the identity view.
    # returns the augmented training matrix (len(views)*n rows), repeated labels,
    # original-image group ids, and the identity-view eval matrix.
    n = len(y)
    x_aug = np.concatenate(views, axis=0)
    y_aug = np.tile(y, len(views))
    groups = np.tile(np.arange(n), len(views))

    return x_aug, y_aug, groups, views[0]


def grouped_oof(x_aug, y_aug, groups, x_eval, y_eval, c, n_splits, n_repeats, seed):
    # split the original images; train on augmented views of train-fold originals only;
    # evaluate on held-out originals' identity features. no augmented twin leaks across.
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    n_classes = len(np.unique(y_eval))
    oof = np.zeros((len(y_eval), n_classes), dtype=float)
    counts = np.zeros(len(y_eval), dtype=float)

    for tr_orig, va_orig in rskf.split(x_eval, y_eval):
        tr_mask = np.isin(groups, tr_orig)
        clf = _fit_one(x_aug[tr_mask], y_aug[tr_mask], c)
        oof[va_orig] += clf.predict_proba(l2_normalize(x_eval[va_orig]))
        counts[va_orig] += 1.0

    oof /= counts[:, None]

    return oof


def fit_view_fold_models(
    x_aug, y_aug, groups, x_eval, y_eval, c, n_splits, n_repeats, seed
):
    # deployed fold ensemble: one probe per fold trained on that fold's augmented views
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    fold_models = []

    for tr_orig, _ in rskf.split(x_eval, y_eval):
        tr_mask = np.isin(groups, tr_orig)
        fold_models.append(_fit_one(x_aug[tr_mask], y_aug[tr_mask], c))

    return fold_models


def cross_validate_views(
    x_aug, y_aug, groups, x_eval, y_eval, c_grid, n_splits, n_repeats, seed
):
    # pick c by grouped-oof accuracy on the original images; return the best-c oof too
    results = {}
    oofs = {}

    for c in c_grid:
        oof = grouped_oof(
            x_aug, y_aug, groups, x_eval, y_eval, c, n_splits, n_repeats, seed
        )
        oofs[c] = oof
        results[str(c)] = {"mean": oof_accuracy(oof, y_eval)}

    best_c = max(c_grid, key=lambda c: results[str(c)]["mean"])

    return results, best_c, oofs[best_c]
