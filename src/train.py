import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

from src.data import list_train_images
from src.features import extract_features, load_backbone
from src.head import cross_validate, fit_fold_models, oof_accuracy
from src.labels import build_class_to_idx
from src.utils import collect_metadata, get_device, load_config, set_seed


def _save_figures(results, c_grid, oof, y, fig_dir):
    # best-effort report figures. never let plotting break training
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir.mkdir(parents=True, exist_ok=True)
        means = [results[str(c)]["mean"] for c in c_grid]
        plt.figure()
        plt.semilogx(c_grid, means, marker="o")
        plt.xlabel("C")
        plt.ylabel("cv accuracy")
        plt.title("cv accuracy vs l2 strength")
        plt.savefig(fig_dir / "cv_curve.png", bbox_inches="tight")
        plt.close()

        n = oof.shape[1]
        cm = np.zeros((n, n), dtype=int)

        for true, pred in zip(y, oof.argmax(axis=1)):
            cm[true, pred] += 1

        plt.figure(figsize=(8, 8))
        plt.imshow(cm)
        plt.xlabel("predicted")
        plt.ylabel("true")
        plt.title("oof confusion matrix")
        plt.savefig(fig_dir / "confusion_matrix.png", bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"figure generation skipped: {exc}")


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    out = Path(cfg["output_dir"])
    (out / "model").mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(parents=True, exist_ok=True)

    class_to_idx = build_class_to_idx(cfg["data"]["train_dir"])
    paths, labels = list_train_images(cfg["data"]["train_dir"], class_to_idx)
    y = np.array(labels)

    model, mean, std = load_backbone(
        cfg["model"]["name"], cfg["model"]["img_size"], device
    )

    view = cfg["features"]["views_train"][0]
    tag = re.sub(r"[^0-9a-zA-Z]+", "_", cfg["model"]["name"])
    cache = out / "cache" / f"train__{tag}__{view}__{cfg['model']['img_size']}.npz"

    x = extract_features(model, paths, view, cfg, mean, std, device, cache)

    c_grid = cfg["cv"]["c_grid"]
    # clamp folds to the smallest class so stratified k-fold is always feasible
    min_class = int(np.bincount(y).min())
    n_folds = min(cfg["cv"]["n_folds"], min_class)

    if n_folds < cfg["cv"]["n_folds"]:
        print(f"clamped n_folds to {n_folds} (smallest class has {min_class} images)")

    results, best_c = cross_validate(x, y, c_grid, n_folds, cfg["seed"])
    fold_models, oof = fit_fold_models(x, y, best_c, n_folds, cfg["seed"])
    acc = oof_accuracy(oof, y)

    bundle = {
        "fold_models": fold_models,
        "best_c": best_c,
        "class_to_idx": class_to_idx,
        "model_name": cfg["model"]["name"],
        "img_size": cfg["model"]["img_size"],
        "mean": mean,
        "std": std,
    }

    bundle_path = out / "model" / "bundle.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    weights_sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    with open(out / "model" / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)
    with open(out / "model" / "metrics.json", "w") as f:
        json.dump(
            {"cv": results, "best_c": best_c, "oof_accuracy": acc, "n_folds": n_folds},
            f,
            indent=2,
        )
    extra = {
        "oof_accuracy": acc,
        "best_c": best_c,
        "weights_sha256": weights_sha,
        "n_folds": n_folds,
    }
    with open(out / "model" / "metadata.json", "w") as f:
        json.dump(collect_metadata(cfg, extra), f, indent=2)

    _save_figures(results, c_grid, oof, y, out / "figures")
    print(f"best C={best_c}  oof accuracy={acc:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
