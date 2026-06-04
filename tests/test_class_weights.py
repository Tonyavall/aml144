import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.models.lora_train import balanced_class_weights


def test_balanced_weights_match_sklearn():
    y = np.array([0, 0, 0, 0, 1, 1, 2])
    got = balanced_class_weights(y, 3)
    expected = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=y)
    assert np.allclose(got, expected)


def test_balanced_weights_rarer_class_gets_more_weight():
    y = np.array([0, 0, 0, 0, 1, 1, 2])
    w = balanced_class_weights(y, 3)
    assert w[2] > w[1] > w[0]


def test_balanced_weights_handles_class_absent_in_fold():
    # class 2 missing from this fold -> clamp count to 1, no division by zero
    y = np.array([0, 0, 1, 1])
    w = balanced_class_weights(y, 3)
    assert np.isfinite(w).all()
