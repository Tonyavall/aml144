import numpy as np

from src.models.balance import sinkhorn_balanced


def test_columns_approach_target_and_rows_sum_to_one():
    # exercises the default n_iter; columns converge to target at machine precision
    rng = np.random.default_rng(0)
    probs = rng.random((20, 4))
    probs = probs / probs.sum(axis=1, keepdims=True)

    out = sinkhorn_balanced(probs, col_target=5.0)

    assert out.shape == (20, 4)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.allclose(out.sum(axis=0), 5.0, atol=1e-4)
    preds = out.argmax(axis=1)
    assert preds.min() >= 0 and preds.max() <= 3


def test_skewed_input_is_rebalanced_toward_target():
    # a head biased toward class 0 must be pushed toward balanced column sums;
    # guards against a silent no-op in the column-rescale step
    n, k = 40, 4
    rng = np.random.default_rng(1)
    probs = rng.random((n, k))
    probs[:, 0] += 3.0
    probs = probs / probs.sum(axis=1, keepdims=True)

    before = np.bincount(probs.argmax(axis=1), minlength=k)
    out = sinkhorn_balanced(probs, col_target=n / k)

    assert np.allclose(out.sum(axis=0), n / k, atol=1e-4)
    after = np.bincount(out.argmax(axis=1), minlength=k)
    assert before[0] > after[0]
    assert after.max() - after.min() < before.max() - before.min()


def test_uniform_input_stays_uniform():
    probs = np.full((8, 4), 0.25)
    out = sinkhorn_balanced(probs, col_target=2.0, n_iter=50)
    assert np.allclose(out, 0.25, atol=1e-6)
