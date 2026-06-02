import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from peft import set_peft_model_state_dict
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.data.images import list_all_test_images, list_train_images
from src.data.labels import build_class_to_idx, idx_to_class
from src.models.backbone import load_backbone
from src.models.balance import sinkhorn_balanced
from src.models.fusion import blend, tune_weights
from src.models.head import oof_accuracy
from src.models.lora_train import (
    LoraImageDataset,
    _train_lora_on_split,
    build_lora_model,
    build_lora_transforms,
)
from src.submission import write_submission
from src.utils import collect_metadata, get_device, load_config, set_seed


def _member_cfg(cfg, member):
    # per-member cfg: set the active backbone and merge this member's lora overrides on top
    # of the base lora config, forcing the member's img_size and pool_mode. does not mutate cfg.
    c = dict(cfg)
    c["model"] = {"name": member["name"], "img_size": member["img_size"]}
    lcfg = dict(cfg["lora"])
    lcfg.update(member.get("lora", {}))
    lcfg["img_size"] = member["img_size"]
    lcfg["pool_mode"] = member["pool_mode"]
    c["lora"] = lcfg

    return c


def shared_folds(y, n_folds, seed):
    # the single fold structure every member shares so per-member oof rows stay aligned.
    # stratifiedkfold uses only y + random_state for the split, so a length-only x is fine.
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    return list(skf.split(np.zeros((len(y), 1)), y))


def member_oof_and_bundle(member, cfg, paths, y, mean, std, device, seed):
    # one shared-fold pass for a member: returns the full oof matrix plus each fold's
    # best-epoch state (lora + head). the deploy-seed states are reused as the bundle.
    set_seed(seed)
    n_classes = int(len(np.unique(y)))
    n_folds = cfg["cv"]["n_folds"]
    mc = _member_cfg(cfg, member)
    oof = np.zeros((len(y), n_classes))
    folds = []

    for k, (tr, va) in enumerate(shared_folds(y, n_folds, seed)):
        _, probs, state = _train_lora_on_split(
            tr, va, mc, paths, y, mean, std, device, n_classes,
            f"{member['name']} seed {seed} fold {k}",
        )
        oof[va] = probs
        folds.append(state)

    return oof, folds, n_classes


def member_test_probs(member, folds, n_classes, mean, std, test_paths, cfg, device):
    # softmax-ensemble the member's fold-models over the test set at its deploy resolution
    mc = _member_cfg(cfg, member)
    lcfg = mc["lora"]
    model = build_lora_model(
        member["name"], member["img_size"], n_classes, lcfg, device, member["pool_mode"]
    )
    _, eval_tf = build_lora_transforms(member["img_size"], mean, std)
    test_ds = LoraImageDataset(test_paths, np.zeros(len(test_paths), dtype=int), eval_tf)
    test_dl = DataLoader(test_ds, batch_size=lcfg["batch_size"], shuffle=False, num_workers=0)

    probs = None
    for state in folds:
        set_peft_model_state_dict(model.backbone, state["lora"])
        model.head.load_state_dict(state["head"])
        model.eval()
        chunks = []

        with torch.no_grad():
            for imgs, _ in test_dl:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(imgs.to(device))
                chunks.append(logits.softmax(dim=1).float().cpu().numpy())

        p = np.concatenate(chunks)
        probs = p if probs is None else probs + p

    del model
    torch.cuda.empty_cache()

    return probs / len(folds)


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
            oof, folds, n_classes = member_oof_and_bundle(
                member, cfg, paths, y, mean, std, device, s
            )
            seed_oof[s].append(oof)
            member_oofs[s] = oof
            if s == deploy_seed:
                deploy_folds = folds

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
    blend_oof_acc = oof_accuracy(blend(oof_list, weights), y)
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
