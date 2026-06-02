import numpy as np


def sinkhorn_balanced(probs, col_target, n_iter=50, eps=1e-8):
    # project a row-stochastic probability matrix so each class column sums to
    # col_target, honoring the known uniform test prior. alternates row-normalize
    # (rows sum to 1) and column-rescale (columns sum to col_target).
    p = np.clip(probs.astype(np.float64), eps, None)

    for _ in range(n_iter):
        p = p / p.sum(axis=1, keepdims=True)
        p = p * (col_target / p.sum(axis=0, keepdims=True))

    return p / p.sum(axis=1, keepdims=True)
