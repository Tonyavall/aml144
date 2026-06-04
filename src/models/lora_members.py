"""shared lora-member fold helpers used by the deployed single_ft model and the
deprecated ensemble_ft orchestrator. these build the shared cv folds, run a member's
per-fold lora training into an aligned oof matrix, and softmax-ensemble a member's
fold-models over the test set. kept here (not in an entry-point module) so the
deployed path does not import a deprecated orchestrator."""

import numpy as np
import torch
from peft import set_peft_model_state_dict
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.models.lora_train import (
    LoraImageDataset,
    _train_lora_on_split,
    build_lora_model,
    build_lora_transforms,
    eval_probs,
)
from src.utils import set_seed


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


def member_oof_and_bundle(
    member, cfg, paths, y, mean, std, device, seed, capture_history=False
):
    # one shared-fold pass for a member: returns the full oof matrix, each fold's
    # best-epoch state (lora + head), n_classes, and (when capture_history) the
    # per-fold per-epoch training history. deploy-seed states are reused as the bundle.
    set_seed(seed)
    n_classes = int(len(np.unique(y)))
    n_folds = cfg["cv"]["n_folds"]
    mc = _member_cfg(cfg, member)
    oof = np.zeros((len(y), n_classes))
    folds = []
    fold_histories = {}

    for k, (tr, va) in enumerate(shared_folds(y, n_folds, seed)):
        history = [] if capture_history else None
        _, probs, state = _train_lora_on_split(
            tr, va, mc, paths, y, mean, std, device, n_classes,
            f"{member['name']} seed {seed} fold {k}",
            history,
        )
        oof[va] = probs
        folds.append(state)
        if capture_history:
            fold_histories[k] = history

    return oof, folds, n_classes, fold_histories


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

    tta_views = lcfg.get("tta_views", ["identity"])
    probs = None
    for state in folds:
        set_peft_model_state_dict(model.backbone, state["lora"])
        model.head.load_state_dict(state["head"])
        p = eval_probs(model, test_dl, device, tta_views)
        probs = p if probs is None else probs + p

    del model
    torch.cuda.empty_cache()

    return probs / len(folds)
