import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from peft import set_peft_model_state_dict
from torch.utils.data import DataLoader

from src.data.images import list_all_test_images, list_train_images
from src.data.labels import build_class_to_idx, idx_to_class
from src.models.backbone import load_backbone
from src.models.balance import sinkhorn_balanced
from src.models.head import oof_accuracy
from src.models.lora_members import (
    _member_cfg,
    member_oof_and_bundle,
    member_test_probs,
    shared_folds,
)
from src.models.lora_train import (
    LoraImageDataset,
    build_lora_model,
    build_lora_transforms,
    eval_probs,
)
from src.submission import write_submission
from src.utils import collect_metadata, get_device, load_config, set_seed

# reference numbers for the decision (from the fine-tuned ensemble run / experiments.md).
# the ensemble is the incumbent deploy; the baseline is the no-lever siglip-2 member.
ENSEMBLE_REF = {"mean": 0.9404, "std": 0.0024, "deploy": 0.9416}
BASELINE_REF = {"deploy": 0.9435, "mean": 0.9401}


def decision_outcome(single_mean, single_std, ens_mean, ens_std):
    # tie-is-enough rule: single wins outright at or above the ensemble mean; it ties
    # (and is preferred on simplicity/cost/reproducibility) when within one combined
    # std; otherwise the ensemble keeps the deploy.
    combined_std = (single_std**2 + ens_std**2) ** 0.5
    if single_mean >= ens_mean:
        return "win"
    if ens_mean - single_mean <= combined_std:
        return "tie"
    return "loss"


def member_val_oof(member, folds, n_classes, mean, std, paths, y, seed, cfg, device, tta_views):
    # reassemble an oof matrix from stored fold snapshots by evaluating each fold's
    # val partition with the given tta views. lets us read the tta-off oof without
    # retraining (training already produced the tta-on oof). folds are appended in
    # shared_folds order, so zipping against shared_folds(...) re-pairs them.
    mc = _member_cfg(cfg, member)
    lcfg = mc["lora"]
    model = build_lora_model(
        member["name"], member["img_size"], n_classes, lcfg, device, member["pool_mode"]
    )
    _, eval_tf = build_lora_transforms(member["img_size"], mean, std)
    n_folds = cfg["cv"]["n_folds"]
    oof = np.zeros((len(y), n_classes))

    for state, (_, va) in zip(folds, shared_folds(y, n_folds, seed)):
        set_peft_model_state_dict(model.backbone, state["lora"])
        model.head.load_state_dict(state["head"])
        ds = LoraImageDataset([paths[i] for i in va], y[va], eval_tf)
        dl = DataLoader(ds, batch_size=lcfg["batch_size"], shuffle=False, num_workers=0)
        oof[va] = eval_probs(model, dl, device, tta_views)

    del model
    torch.cuda.empty_cache()

    return oof


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    out = Path(cfg["output_dir"])
    (out / "single_ft").mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(parents=True, exist_ok=True)

    class_to_idx = build_class_to_idx(cfg["data"]["train_dir"])
    paths, labels = list_train_images(cfg["data"]["train_dir"], class_to_idx)
    y = np.array(labels)
    test_ids, test_paths = list_all_test_images(cfg["data"]["test_dir"])

    scfg = cfg["single_ft"]
    member = scfg["member"]
    deploy_seed = scfg["deploy_seed"]
    oof_seeds = scfg["oof_seeds"]
    tta_views = member.get("lora", {}).get("tta_views", ["identity"])

    # load_backbone only supplies normalization stats; free the frozen model from vram
    backbone_for_stats, mean, std = load_backbone(member["name"], member["img_size"], device)
    del backbone_for_stats
    torch.cuda.empty_cache()

    seed_oof = {}
    deploy_folds, n_classes = None, None

    for s in oof_seeds:
        oof, folds, n_classes, _ = member_oof_and_bundle(
            member, cfg, paths, y, mean, std, device, s, capture_history=False
        )
        seed_oof[s] = oof
        if s == deploy_seed:
            deploy_folds = folds

    seed_acc = {str(s): oof_accuracy(seed_oof[s], y) for s in oof_seeds}
    acc_vals = list(seed_acc.values())
    single_mean = float(np.mean(acc_vals))
    single_std = float(np.std(acc_vals))

    # tta attribution on the deploy seed (no retrain): identity-only vs tta oof
    tta_oof_acc = oof_accuracy(seed_oof[deploy_seed], y)
    identity_oof = member_val_oof(
        member, deploy_folds, n_classes, mean, std, paths, y, deploy_seed, cfg, device, ["identity"]
    )
    identity_oof_acc = oof_accuracy(identity_oof, y)

    outcome = decision_outcome(single_mean, single_std, ENSEMBLE_REF["mean"], ENSEMBLE_REF["std"])
    recommended = "single" if outcome in ("win", "tie") else "ensemble"

    test_probs = member_test_probs(
        member, deploy_folds, n_classes, mean, std, test_paths, cfg, device
    )
    if cfg.get("inference", {}).get("sinkhorn", False):
        col_target = len(test_ids) / test_probs.shape[1]
        test_probs = sinkhorn_balanced(test_probs, col_target)

    inv = idx_to_class(class_to_idx)
    preds = test_probs.argmax(axis=1)
    id_to_label = {rid: inv[int(p)] for rid, p in zip(test_ids, preds)}
    valid_labels = [inv[i] for i in range(len(inv))]
    write_submission(id_to_label, test_ids, out / scfg["output_submission"], valid_labels)

    bundle = {
        "member": member,
        "n_classes": n_classes,
        "mean": mean,
        "std": std,
        "folds": deploy_folds,
        "class_to_idx": class_to_idx,
        "deploy_seed": deploy_seed,
    }
    with open(out / "single_ft" / "single_ft_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)

    metrics = {
        "seed_oof_acc": seed_acc,
        "oof_mean": single_mean,
        "oof_std": single_std,
        "deploy_seed": deploy_seed,
        "tta_oof_acc": tta_oof_acc,
        "identity_oof_acc": identity_oof_acc,
        "tta_delta": tta_oof_acc - identity_oof_acc,
        "tta_views": tta_views,
        "baseline_ref": BASELINE_REF,
        "ensemble_ref": ENSEMBLE_REF,
        "decision_outcome": outcome,
        "recommended_deploy": recommended,
    }
    with open(out / "single_ft" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out / "single_ft" / "metadata.json", "w") as f:
        json.dump(collect_metadata(cfg, {"oof_mean": single_mean, "decision": outcome}), f, indent=2)

    print(f"single siglip-2 oof per seed: {seed_acc}")
    print(f"  mean={single_mean:.4f} +/- {single_std:.4f}")
    print(
        f"  tta_oof={tta_oof_acc:.4f} identity_oof={identity_oof_acc:.4f} "
        f"(tta delta {tta_oof_acc - identity_oof_acc:+.4f})"
    )
    print(
        f"  ensemble ref mean={ENSEMBLE_REF['mean']:.4f} -> outcome={outcome} "
        f"(recommended deploy: {recommended})"
    )
    print(f"wrote {out / scfg['output_submission']} ({len(test_ids)} rows)")


def predict_main(config_path="config.yaml"):
    # inference-only: load the trained deploy bundle (outputs/single_ft/
    # single_ft_bundle.pkl, e.g. downloaded from the readme's google drive link)
    # and write the submission without any retraining. eval settings (identity-only
    # tta) come from the current config's single_ft member, matching the deploy.
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    out = Path(cfg["output_dir"])

    bundle_path = out / "single_ft" / "single_ft_bundle.pkl"
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    scfg = cfg["single_ft"]
    member = scfg["member"]
    test_ids, test_paths = list_all_test_images(cfg["data"]["test_dir"])

    test_probs = member_test_probs(
        member,
        bundle["folds"],
        bundle["n_classes"],
        bundle["mean"],
        bundle["std"],
        test_paths,
        cfg,
        device,
    )
    if cfg.get("inference", {}).get("sinkhorn", False):
        col_target = len(test_ids) / test_probs.shape[1]
        test_probs = sinkhorn_balanced(test_probs, col_target)

    inv = idx_to_class(bundle["class_to_idx"])
    preds = test_probs.argmax(axis=1)
    id_to_label = {rid: inv[int(p)] for rid, p in zip(test_ids, preds)}
    valid_labels = [inv[i] for i in range(len(inv))]
    write_submission(id_to_label, test_ids, out / scfg["output_submission"], valid_labels)
    print(f"wrote {out / scfg['output_submission']} ({len(test_ids)} rows) from {bundle_path}")


def _fold_series(fold_histories):
    # turn {fold_index: [ {epoch, train_loss, ...}, ... ]} into the report's series shape
    # {"fold k": {metric: [values over epochs]}} so plot_training_curves draws one line per fold.
    metrics = ("train_loss", "train_acc", "val_loss", "val_acc")
    series = {}
    for k in sorted(fold_histories):
        epochs = fold_histories[k]
        series[f"fold {k}"] = {m: [e[m] for e in epochs] for m in metrics}
    return series


def curves_main(config_path="config.yaml"):
    # additive: train the deploy-seed folds with per-epoch history capture and plot the
    # training curves. does not write the submission/bundle/metrics (those stay the
    # lb-validated deploy artifacts) - only writes history.json and training_curves.png.
    from src.report import plot_training_curves

    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    out = Path(cfg["output_dir"])
    (out / "single_ft").mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(parents=True, exist_ok=True)

    class_to_idx = build_class_to_idx(cfg["data"]["train_dir"])
    paths, labels = list_train_images(cfg["data"]["train_dir"], class_to_idx)
    y = np.array(labels)

    scfg = cfg["single_ft"]
    member = scfg["member"]
    deploy_seed = scfg["deploy_seed"]

    backbone_for_stats, mean, std = load_backbone(member["name"], member["img_size"], device)
    del backbone_for_stats
    torch.cuda.empty_cache()

    _, _, _, fold_histories = member_oof_and_bundle(
        member, cfg, paths, y, mean, std, device, deploy_seed, capture_history=True
    )

    with open(out / "single_ft" / "history.json", "w") as f:
        json.dump({member["name"]: fold_histories}, f, indent=2)

    fig_path = out / "single_ft" / "training_curves.png"
    plot_training_curves(_fold_series(fold_histories), fig_path)
    print(f"wrote {out / 'single_ft' / 'history.json'} and {fig_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "curves":
        curves_main(args[1] if len(args) > 1 else "config.yaml")
    elif args and args[0] == "predict":
        predict_main(args[1] if len(args) > 1 else "config.yaml")
    else:
        main(args[0] if args else "config.yaml")
