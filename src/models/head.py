import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import normalize


def l2_normalize(x):
    # per-sample unit norm turns the logistic head into a cosine classifier
    return normalize(x, norm="l2", axis=1)


def _fit_one(x, y, c):
    # a fold model is a logistic regression fit on l2-normalized features
    clf = LogisticRegression(
        C=c, max_iter=2000, solver="lbfgs", class_weight="balanced"
    )
    clf.fit(l2_normalize(x), y)
    return clf


def cross_validate(x, y, c_grid, n_splits, n_repeats, seed):
    # pick the l2 strength that maximizes mean repeated stratified k-fold accuracy
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    results = {}

    for c in c_grid:
        accs = []

        for tr, va in rskf.split(x, y):
            clf = _fit_one(x[tr], y[tr], c)
            accs.append(float((clf.predict(l2_normalize(x[va])) == y[va]).mean()))

        results[str(c)] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}

    best_c = max(c_grid, key=lambda c: results[str(c)]["mean"])

    return results, best_c


def fit_fold_models(x, y, c, n_splits, n_repeats, seed):
    # deployed ensemble; oof averages a sample's probs across the repeats it is held in
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    n_classes = len(np.unique(y))
    fold_models = []
    oof = np.zeros((len(y), n_classes), dtype=float)
    counts = np.zeros(len(y), dtype=float)

    for tr, va in rskf.split(x, y):
        clf = _fit_one(x[tr], y[tr], c)
        fold_models.append(clf)
        oof[va] += clf.predict_proba(l2_normalize(x[va]))
        counts[va] += 1.0

    oof /= counts[:, None]

    return fold_models, oof


def oof_accuracy(oof, y):
    return float((oof.argmax(axis=1) == y).mean())


def predict_proba(fold_models, x):
    # average predicted probabilities across the fold models on l2-normalized features
    xn = l2_normalize(x)
    n_classes = len(fold_models[0].classes_)
    probs = np.zeros((x.shape[0], n_classes), dtype=float)

    for clf in fold_models:
        probs += clf.predict_proba(xn)

    return probs / len(fold_models)
