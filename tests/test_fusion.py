import numpy as np

from src.deprecated.fusion import blend, tune_weights


def test_blend_equal_weights_is_the_mean():
    a = np.array([[0.7, 0.3], [0.2, 0.8]])
    b = np.array([[0.1, 0.9], [0.6, 0.4]])
    out = blend([a, b], [0.5, 0.5])
    assert np.allclose(out, (a + b) / 2)


def test_blend_shifts_toward_upweighted_matrix():
    a = np.array([[0.9, 0.1]])
    b = np.array([[0.1, 0.9]])
    out = blend([a, b], [0.8, 0.2])
    assert out[0, 0] > 0.5


def test_tune_weights_upweights_strong_and_flags_used():
    # bb0 is informative (correct class highest); bb1 is confidently wrong, so the
    # equal blend mispredicts and up-weighting bb0 strictly helps
    c = 3
    y = np.array([0, 1, 2] * 4)
    n = len(y)
    strong = np.full((n, c), 0.25)
    strong[np.arange(n), y] = 0.5
    harmful = np.full((n, c), 0.1)
    harmful[np.arange(n), (y + 1) % c] = 0.8

    w, used, equal_acc, tuned_acc = tune_weights([strong, harmful], y, step=0.1, margin=0.003)
    assert used is True
    assert w[0] > w[1]
    assert tuned_acc > equal_acc


def test_tune_weights_falls_back_to_equal_when_no_real_gain():
    # two identical backbones: no weighting can beat the equal blend
    c = 3
    y = np.array([0, 1, 2] * 3)
    n = len(y)
    p = np.full((n, c), 0.2)
    p[np.arange(n), y] = 0.6

    w, used, equal_acc, tuned_acc = tune_weights([p, p.copy()], y, step=0.1, margin=0.003)
    assert used is False
    assert np.allclose(w, [0.5, 0.5])


def test_tune_weights_three_identical_backbones_fall_back_to_equal():
    # production uses 3 backbones; three identical backbones cannot be beaten by any
    # weighting (every blend equals the shared matrix), so it must fall back to equal
    c = 3
    y = np.array([0, 1, 2] * 4)
    n = len(y)
    p = np.full((n, c), 0.2)
    p[np.arange(n), y] = 0.6

    w, used, _, _ = tune_weights([p, p.copy(), p.copy()], y, step=0.1, margin=0.003)
    assert used is False
    assert np.allclose(w, [1 / 3, 1 / 3, 1 / 3])


def test_tune_weights_is_deterministic():
    y = np.array([0, 1, 0, 1, 0, 1])
    rng = np.random.default_rng(0)
    a = rng.random((6, 2))
    a /= a.sum(1, keepdims=True)
    b = rng.random((6, 2))
    b /= b.sum(1, keepdims=True)
    r1 = tune_weights([a, b], y)
    r2 = tune_weights([a, b], y)
    assert r1[0] == r2[0] and r1[1:] == r2[1:]
