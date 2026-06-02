import numpy as np


def blend(prob_list, weights):
    # weighted average of per-backbone probability matrices; weights sum to 1
    out = np.zeros_like(prob_list[0], dtype=float)
    for p, w in zip(prob_list, weights):
        out = out + w * p
    return out


def _simplex_grid(n, step):
    # all length-n weight tuples on the given grid that sum to 1 (integer compositions).
    # grid size grows combinatorially, so keep n small and step coarse (n=3, step=0.1
    # is 66 points)
    k = round(1.0 / step)
    assert abs(k - 1.0 / step) < 1e-9, f"step must divide 1 evenly, got {step}"

    def compositions(parts, total):
        if parts == 1:
            yield (total,)
            return
        for i in range(total + 1):
            for rest in compositions(parts - 1, total - i):
                yield (i,) + rest

    return [tuple(c / k for c in comp) for comp in compositions(n, k)]


def tune_weights(oof_list, y, step=0.1, margin=0.003):
    # grid-search blend weights on oof accuracy; adopt tuned weights only if they beat
    # the equal-weight blend by margin, else fall back to equal weights
    n = len(oof_list)

    def acc(weights):
        return float((blend(oof_list, weights).argmax(axis=1) == y).mean())

    equal = [1.0 / n] * n
    equal_acc = acc(equal)
    best_w, tuned_acc = equal, equal_acc

    for w in _simplex_grid(n, step):
        a = acc(w)
        if a > tuned_acc:
            best_w, tuned_acc = list(w), a

    if tuned_acc > equal_acc + margin:
        return best_w, True, equal_acc, tuned_acc
    return equal, False, equal_acc, tuned_acc
