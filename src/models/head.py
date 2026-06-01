import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def _fit_one(x, y, c):
    # a fold model is a self-contained scaler + logistic regression
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=c, max_iter=5000, solver="lbfgs")
    clf.fit(scaler.transform(x), y)

    return scaler, clf


def cross_validate(x, y, c_grid, n_folds, seed):
    # pick the l2 strength that maximizes mean stratified k-fold accuracy
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    results = {}

    for c in c_grid:
        accs = []

        for tr, va in skf.split(x, y):
            scaler, clf = _fit_one(x[tr], y[tr], c)
            accs.append(float(clf.score(scaler.transform(x[va]), y[va])))

        results[str(c)] = {"mean": float(np.mean(accs)), "std": float(np.std(accs))}

    best_c = max(c_grid, key=lambda c: results[str(c)]["mean"])

    return results, best_c


def fit_fold_models(x, y, c, n_folds, seed):
    # the deployed ensemble is these fold models; oof gives an honest accuracy
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    n_classes = len(np.unique(y))
    fold_models = []
    oof = np.zeros((len(y), n_classes), dtype=float)

    for tr, va in skf.split(x, y):
        scaler, clf = _fit_one(x[tr], y[tr], c)
        fold_models.append((scaler, clf))
        oof[va] = clf.predict_proba(scaler.transform(x[va]))

    return fold_models, oof


def oof_accuracy(oof, y):
    return float((oof.argmax(axis=1) == y).mean())


def predict_proba(fold_models, x):
    # softmax-average across the fold models
    n_classes = len(fold_models[0][1].classes_)
    probs = np.zeros((x.shape[0], n_classes), dtype=float)

    for scaler, clf in fold_models:
        probs += clf.predict_proba(scaler.transform(x))

    return probs / len(fold_models)
