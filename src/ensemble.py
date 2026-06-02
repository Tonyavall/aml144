import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch

from src.data.images import list_all_test_images, list_train_images
from src.data.labels import build_class_to_idx, idx_to_class
from src.models.backbone import (
    extract_features,
    extract_multiview_features,
    load_backbone,
)
from src.models.balance import sinkhorn_balanced
from src.models.fusion import blend, tune_weights
from src.models.head import (
    cross_validate,
    fit_fold_models,
    oof_accuracy,
    predict_proba,
)
from src.models.multiview import (
    cross_validate_views,
    fit_view_fold_models,
    stack_views,
)
from src.submission import write_submission
from src.utils import get_device, load_config, set_seed


def _bb_cfg(cfg, bb):
    # shallow cfg copy with model set to this backbone; extract_* read cfg["model"]
    c = dict(cfg)
    c["model"] = {"name": bb["name"], "img_size": bb["img_size"]}
    return c


def _predict_test(model, fold_models, bb, test_paths, c, mean, std, device, out):
    # average per-view probabilities over the test-time views (identity + hflip);
    # tta is applied uniformly to every backbone, including single-view-trained ones
    tag = re.sub(r"[^0-9a-zA-Z]+", "_", bb["name"])
    probs = None

    for view in c["features"]["views_test"]:
        cache = out / "cache" / f"test__{tag}__{view}__{bb['img_size']}.npz"
        x = extract_features(
            model, test_paths, view, c, mean, std, device, cache, bb["pool"]
        )
        p = predict_proba(fold_models, x)
        probs = p if probs is None else probs + p

    return probs / len(c["features"]["views_test"])


def run_backbone(bb, paths, y, test_paths, cfg, device, out):
    # train this backbone's probe (oof) and predict test probs; one model in memory
    c = _bb_cfg(cfg, bb)
    model, mean, std = load_backbone(bb["name"], bb["img_size"], device)
    c_grid = cfg["cv"]["c_grid"]
    n_repeats = cfg["cv"]["n_repeats"]
    seed = cfg["seed"]
    min_class = int(np.bincount(y).min())
    n_folds = min(cfg["cv"]["n_folds"], min_class)
    tag = re.sub(r"[^0-9a-zA-Z]+", "_", bb["name"])

    if bb["aug_views"] > 1:
        views = extract_multiview_features(
            model,
            paths,
            bb["aug_views"],
            c,
            mean,
            std,
            device,
            out / "cache",
            seed,
            bb["pool"],
        )
        x_aug, y_aug, groups, x_eval = stack_views(views, y)
        _, best_c, oof = cross_validate_views(
            x_aug, y_aug, groups, x_eval, y, c_grid, n_folds, n_repeats, seed
        )
        fold_models = fit_view_fold_models(
            x_aug, y_aug, groups, x_eval, y, best_c, n_folds, n_repeats, seed
        )
    else:
        cache = out / "cache" / f"train__{tag}__identity__{bb['img_size']}.npz"
        x = extract_features(
            model, paths, "identity", c, mean, std, device, cache, bb["pool"]
        )
        _, best_c = cross_validate(x, y, c_grid, n_folds, n_repeats, seed)
        fold_models, oof = fit_fold_models(x, y, best_c, n_folds, n_repeats, seed)

    test_probs = _predict_test(
        model, fold_models, bb, test_paths, c, mean, std, device, out
    )
    del model
    torch.cuda.empty_cache()

    return {
        "oof": oof,
        "test": test_probs,
        "best_c": best_c,
        "fold_models": fold_models,
    }


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    out = Path(cfg["output_dir"])
    (out / "ensemble").mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(parents=True, exist_ok=True)

    class_to_idx = build_class_to_idx(cfg["data"]["train_dir"])
    paths, labels = list_train_images(cfg["data"]["train_dir"], class_to_idx)
    y = np.array(labels)
    test_ids, test_paths = list_all_test_images(cfg["data"]["test_dir"])

    results = [
        run_backbone(bb, paths, y, test_paths, cfg, device, out)
        for bb in cfg["backbones"]
    ]
    oof_list = [r["oof"] for r in results]
    test_list = [r["test"] for r in results]

    ecfg = cfg["ensemble"]
    weights, used_tuned, equal_acc, tuned_acc = tune_weights(
        oof_list, y, ecfg["weight_step"], ecfg["weight_margin"]
    )

    test_blend = blend(test_list, weights)
    if cfg.get("inference", {}).get("sinkhorn", False):
        col_target = len(test_ids) / test_blend.shape[1]
        test_blend = sinkhorn_balanced(test_blend, col_target)

    inv = idx_to_class(class_to_idx)
    preds = test_blend.argmax(axis=1)
    id_to_label = {rid: inv[int(p)] for rid, p in zip(test_ids, preds)}
    valid_labels = [inv[i] for i in range(len(inv))]
    write_submission(
        id_to_label, test_ids, out / ecfg["output_submission"], valid_labels
    )

    per_bb = [
        {
            "name": bb["name"],
            "best_c": r["best_c"],
            "oof_acc": oof_accuracy(r["oof"], y),
        }
        for bb, r in zip(cfg["backbones"], results)
    ]
    blend_oof_acc = oof_accuracy(blend(oof_list, weights), y)
    metrics = {
        "per_backbone": per_bb,
        "equal_acc": equal_acc,
        "tuned_acc": tuned_acc,
        "used_tuned": used_tuned,
        "weights": list(weights),
        "blend_oof_acc": blend_oof_acc,
    }
    with open(out / "ensemble" / "ensemble_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    bundle = {
        "backbones": cfg["backbones"],
        "weights": list(weights),
        "fold_models": [r["fold_models"] for r in results],
        "class_to_idx": class_to_idx,
    }
    with open(out / "ensemble" / "ensemble_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)

    for bb, r in zip(cfg["backbones"], results):
        print(
            f"  {bb['name']}: oof={oof_accuracy(r['oof'], y):.4f} best_c={r['best_c']}"
        )
    print(
        f"equal={equal_acc:.4f} tuned={tuned_acc:.4f} used_tuned={used_tuned} "
        f"weights={[round(w, 2) for w in weights]} blend_oof={blend_oof_acc:.4f}"
    )
    print(f"wrote {out / ecfg['output_submission']} ({len(test_ids)} rows)")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
