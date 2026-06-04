import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from src.data.images import list_all_test_images, list_train_images
from src.data.labels import build_class_to_idx, idx_to_class
from src.deprecated.fusion import blend, tune_weights
from src.models.backbone import load_backbone
from src.models.balance import sinkhorn_balanced
from src.models.head import oof_accuracy
from src.models.lora_members import member_oof_and_bundle, member_test_probs
from src.submission import write_submission
from src.utils import collect_metadata, get_device, load_config, set_seed


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    out = Path(cfg["output_dir"])
    (out / "ensemble_ft").mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(parents=True, exist_ok=True)

    class_to_idx = build_class_to_idx(cfg["data"]["train_dir"])
    paths, labels = list_train_images(cfg["data"]["train_dir"], class_to_idx)
    y = np.array(labels)
    test_ids, test_paths = list_all_test_images(cfg["data"]["test_dir"])

    ecfg = cfg["ensemble_ft"]
    members = ecfg["members"]
    deploy_seed = ecfg["deploy_seed"]
    oof_seeds = ecfg["oof_seeds"]

    seed_oof = {s: [] for s in oof_seeds}  # seed -> [per-member oof matrix]
    test_list = []
    member_bundles = []
    per_member_meta = []
    member_histories = {}  # deploy-seed per-fold per-epoch history, by member name

    for member in members:
        # load_backbone only supplies this member's normalization stats; free the
        # frozen model from vram before training so it does not sit alongside the
        # lora model for the whole member run
        backbone_for_stats, mean, std = load_backbone(
            member["name"], member["img_size"], device
        )
        del backbone_for_stats
        torch.cuda.empty_cache()
        deploy_folds, n_classes = None, None
        member_oofs = {}

        for s in oof_seeds:
            oof, folds, n_classes, fold_histories = member_oof_and_bundle(
                member, cfg, paths, y, mean, std, device, s,
                capture_history=(s == deploy_seed),
            )
            seed_oof[s].append(oof)
            member_oofs[s] = oof
            if s == deploy_seed:
                deploy_folds = folds
                member_histories[member["name"]] = fold_histories

        test_probs = member_test_probs(
            member, deploy_folds, n_classes, mean, std, test_paths, cfg, device
        )
        test_list.append(test_probs)
        member_bundles.append(
            {
                "model_name": member["name"],
                "img_size": member["img_size"],
                "pool_mode": member["pool_mode"],
                "n_classes": n_classes,
                "mean": mean,
                "std": std,
                "folds": deploy_folds,
            }
        )
        per_member_meta.append(
            {
                "name": member["name"],
                "oof_acc": oof_accuracy(member_oofs[deploy_seed], y),
                "seed_oof_acc": {
                    str(s): oof_accuracy(member_oofs[s], y) for s in oof_seeds
                },
            }
        )

    oof_list = seed_oof[deploy_seed]
    weights, used_tuned, equal_acc, tuned_acc = tune_weights(
        oof_list, y, ecfg["weight_step"], ecfg["weight_margin"]
    )
    oof_blend = blend(oof_list, weights)
    blend_oof_acc = oof_accuracy(oof_blend, y)
    seed_blend_acc = {
        str(s): oof_accuracy(blend(seed_oof[s], weights), y) for s in oof_seeds
    }

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

    bundle = {
        "members": member_bundles,
        "weights": list(weights),
        "class_to_idx": class_to_idx,
    }
    with open(out / "ensemble_ft" / "ensemble_ft_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)

    # report artifacts (local/gitignored): blended deploy-seed oof preds for the
    # confusion matrix, and the deploy-seed per-epoch history for the training curves
    np.savez(out / "ensemble_ft" / "oof_blend.npz", probs=oof_blend, y=y)
    with open(out / "ensemble_ft" / "history.json", "w") as f:
        json.dump(member_histories, f, indent=2)

    blend_vals = list(seed_blend_acc.values())
    metrics = {
        "per_member": per_member_meta,
        "equal_acc": equal_acc,
        "tuned_acc": tuned_acc,
        "used_tuned": used_tuned,
        "weights": list(weights),
        "blend_oof_acc": blend_oof_acc,
        "seed_blend_acc": seed_blend_acc,
        "blend_oof_mean": float(np.mean(blend_vals)),
        "blend_oof_std": float(np.std(blend_vals)),
        "deploy_seed": deploy_seed,
        "oof_seeds": oof_seeds,
    }
    with open(out / "ensemble_ft" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out / "ensemble_ft" / "metadata.json", "w") as f:
        json.dump(
            collect_metadata(
                cfg, {"blend_oof_acc": blend_oof_acc, "weights": list(weights)}
            ),
            f,
            indent=2,
        )

    for m in per_member_meta:
        print(f"  {m['name']}: oof={m['oof_acc']:.4f}")
    print(
        f"equal={equal_acc:.4f} tuned={tuned_acc:.4f} used_tuned={used_tuned} "
        f"weights={[round(w, 2) for w in weights]} blend_oof={blend_oof_acc:.4f}"
    )
    print(
        f"blend oof across seeds={metrics['blend_oof_mean']:.4f} "
        f"+/- {metrics['blend_oof_std']:.4f}"
    )
    print(f"wrote {out / ecfg['output_submission']} ({len(test_ids)} rows)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
