import json
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.data.images import build_transform, list_all_test_images, list_train_images
from src.data.labels import build_class_to_idx, idx_to_class
from src.models.backbone import extract_features, load_backbone
from src.models.head import _fit_one, l2_normalize, oof_accuracy
from src.submission import write_submission
from src.utils import get_device, load_config, set_seed


def stratified_split(y, val_frac, seed):
    # stratified train/val index split; every class appears on both sides
    idx = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        idx, test_size=val_frac, stratify=y, random_state=seed
    )

    return np.sort(train_idx), np.sort(val_idx)


def lora_target_modules(total_blocks, n_last):
    # full module names for attn.qkv + attn.proj on the last n_last blocks
    mods = []

    for i in range(total_blocks - n_last, total_blocks):
        mods.append(f"blocks.{i}.attn.qkv")

        mods.append(f"blocks.{i}.attn.proj")
    return mods


def make_lr_lambda(total_steps, warmup_steps):
    # linear warmup to 1.0 over warmup_steps, then cosine decay to 0 by total_steps
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return lr_lambda


def build_lora_transforms(img_size, mean, std):
    # eval = the deployment identity transform; train = light augmentation
    bicubic = T.InterpolationMode.BICUBIC
    eval_tf = build_transform(img_size, mean, std, "identity")

    train_tf = T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0), interpolation=bicubic),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    return train_tf, eval_tf


class LoraImageDataset(Dataset):
    # maps (paths, labels) to (transformed image tensor, int label)
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), int(self.labels[i])


def pool_tokens(tokens, num_prefix, pool_mode):
    # cls_meanpatch: cls token (index 0) concatenated with the mean of patch tokens
    # (everything after the prefix tokens). avg: mean over all tokens, for gap backbones
    # that have no cls token (num_prefix=0).
    if pool_mode == "cls_meanpatch":
        cls = tokens[:, 0]
        patches = tokens[:, num_prefix:].mean(dim=1)
        return torch.cat([cls, patches], dim=1)
    if pool_mode == "avg":
        return tokens.mean(dim=1)
    raise ValueError(f"unknown pool_mode: {pool_mode}")


def head_input_dim(pool_mode, feat_dim):
    # cls_meanpatch concatenates two feat-dim vectors; avg keeps a single feat-dim vector
    if pool_mode == "cls_meanpatch":
        return 2 * feat_dim
    if pool_mode == "avg":
        return feat_dim
    raise ValueError(f"unknown pool_mode: {pool_mode}")


class LoraClassifier(nn.Module):
    # frozen+lora backbone feeding a trainable head over a pooled token representation
    def __init__(self, backbone, head, num_prefix, pool_mode="cls_meanpatch"):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.num_prefix = num_prefix
        self.pool_mode = pool_mode

    def forward(self, x):
        tokens = self.backbone.forward_features(x)
        feat = pool_tokens(tokens, self.num_prefix, self.pool_mode)

        return self.head(feat)


def build_lora_model(model_name, img_size, n_classes, lcfg, device, pool_mode="cls_meanpatch"):
    # frozen backbone with lora on the last n blocks' attention + a fresh pooled-token head.
    # pool_mode selects the token pooling and head input dim (cls_meanpatch=2*feat, avg=feat).
    base = timm.create_model(
        model_name, pretrained=True, num_classes=0, img_size=img_size
    )
    feat_dim = base.num_features
    num_prefix = base.num_prefix_tokens
    total_blocks = len(base.blocks)

    targets = lora_target_modules(total_blocks, lcfg["blocks"])
    peft_cfg = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        target_modules=targets,
        bias="none",
    )

    backbone = get_peft_model(base, peft_cfg)  # freezes base, trains lora only
    head = nn.Linear(head_input_dim(pool_mode, feat_dim), n_classes)
    model = LoraClassifier(backbone, head, num_prefix, pool_mode).to(device)

    return model


def linear_probe_val_acc(features, y, train_idx, val_idx, c):
    # paired frozen-probe baseline using the deployed head recipe (l2-norm + balanced)
    clf = _fit_one(features[train_idx], y[train_idx], c)
    pred = clf.predict(l2_normalize(features[val_idx]))

    return float((pred == y[val_idx]).mean())


def _train_lora_on_split(
    train_idx, val_idx, cfg, paths, y, mean, std, device, n_classes, label
):
    # train lora on train_idx, early-stop on val_idx accuracy
    # return (best_val_acc, best_val_probs) where probs are aligned to val_idx order
    lcfg = cfg["lora"]
    train_tf, eval_tf = build_lora_transforms(lcfg["img_size"], mean, std)

    train_ds = LoraImageDataset([paths[i] for i in train_idx], y[train_idx], train_tf)
    val_ds = LoraImageDataset([paths[i] for i in val_idx], y[val_idx], eval_tf)

    # num_workers=0: safe on windows, backbone forward is the bottleneck anyway
    train_dl = DataLoader(
        train_ds, batch_size=lcfg["batch_size"], shuffle=True, num_workers=0
    )
    val_dl = DataLoader(
        val_ds, batch_size=lcfg["batch_size"], shuffle=False, num_workers=0
    )

    model = build_lora_model(
        cfg["model"]["name"],
        lcfg["img_size"],
        n_classes,
        lcfg,
        device,
        lcfg.get("pool_mode", "cls_meanpatch"),
    )

    lora_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and n.startswith("backbone")
    ]

    head_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and n.startswith("head")
    ]

    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": lcfg["lr_lora"]},
            {"params": head_params, "lr": lcfg["lr_head"]},
        ],
        weight_decay=lcfg["weight_decay"],
    )

    steps_per_epoch = max(1, len(train_dl))
    total_steps = steps_per_epoch * lcfg["epochs"]
    warmup_steps = int(lcfg["warmup_frac"] * total_steps)

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, make_lr_lambda(total_steps, warmup_steps)
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=lcfg["label_smoothing"])

    best_val, best_probs, best_state, bad_epochs = 0.0, None, None, 0

    for epoch in range(lcfg["epochs"]):
        model.train()

        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(imgs)
                loss = loss_fn(logits, labels)

            loss.backward()
            opt.step()
            sched.step()

        model.eval()
        probs_chunks = []
        correct = total = 0

        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs = imgs.to(device)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(imgs)

                probs_chunks.append(logits.softmax(dim=1).float().cpu().numpy())
                pred = logits.argmax(dim=1).cpu()
                correct += int((pred == labels).sum())
                total += len(labels)

        val_acc = correct / total
        print(f"  {label} epoch {epoch}: val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val, best_probs, bad_epochs = val_acc, np.concatenate(probs_chunks), 0
            # snapshot only the trainable params (lora + head) at the best epoch
            best_state = {
                "lora": {
                    k: v.detach().cpu().clone()
                    for k, v in get_peft_model_state_dict(model.backbone).items()
                },
                "head": {
                    k: v.detach().cpu().clone()
                    for k, v in model.head.state_dict().items()
                },
            }
        else:
            bad_epochs += 1

            if bad_epochs >= lcfg["patience"]:
                print(f"  {label}: early stop at epoch {epoch}")
                break

    del model
    torch.cuda.empty_cache()

    return best_val, best_probs, best_state


def train_one_seed(seed, cfg, paths, y, features, mean, std, device):
    # one stratified 80/20 split: train lora, fit the paired linear probe, return both val accs
    set_seed(seed)
    n_classes = int(len(np.unique(y)))

    train_idx, val_idx = stratified_split(y, cfg["lora"]["val_frac"], seed)

    probe_val = linear_probe_val_acc(
        features, y, train_idx, val_idx, cfg["lora"]["probe_c"]
    )

    best_val, _, _ = _train_lora_on_split(
        train_idx, val_idx, cfg, paths, y, mean, std, device, n_classes, f"seed {seed}"
    )

    return {"seed": seed, "lora_val": best_val, "probe_val": probe_val}


def kfold_oof_seed(seed, cfg, paths, y, features, mean, std, device):
    # one k-fold pass: lora oof vs the paired frozen-probe oof on the SAME folds
    set_seed(seed)

    n_classes = int(len(np.unique(y)))
    n_folds = cfg["cv"]["n_folds"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), n_classes))
    probe_oof = np.zeros((len(y), n_classes))

    for k, (tr, va) in enumerate(skf.split(features, y)):
        _, probs, _ = _train_lora_on_split(
            tr, va, cfg, paths, y, mean, std, device, n_classes, f"seed {seed} fold {k}"
        )
        oof[va] = probs

        clf = _fit_one(features[tr], y[tr], cfg["lora"]["probe_c"])
        probe_oof[va] = clf.predict_proba(l2_normalize(features[va]))

    lora_oof = float((oof.argmax(axis=1) == y).mean())

    return {"seed": seed, "lora_oof": lora_oof, "probe_oof": oof_accuracy(probe_oof, y)}


def _load_data_and_probe_features(cfg, device):
    # paths/labels + frozen 2048-d identity features at the lora resolution (paired probe)
    out = Path(cfg["output_dir"])
    (out / "cache").mkdir(parents=True, exist_ok=True)

    class_to_idx = build_class_to_idx(cfg["data"]["train_dir"])
    paths, labels = list_train_images(cfg["data"]["train_dir"], class_to_idx)

    y = np.array(labels)
    img_size = cfg["lora"]["img_size"]
    plain, mean, std = load_backbone(cfg["model"]["name"], img_size, device)
    tag = re.sub(r"[^0-9a-zA-Z]+", "_", cfg["model"]["name"])
    cache = out / "cache" / f"train__{tag}__identity__{img_size}.npz"
    cfg["model"]["img_size"] = img_size  # extract_features reads this for its transform
    features = extract_features(plain, paths, "identity", cfg, mean, std, device, cache)

    del plain

    torch.cuda.empty_cache()

    return paths, y, features, mean, std


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    device = get_device()
    out = Path(cfg["output_dir"])
    (out / "lora").mkdir(parents=True, exist_ok=True)
    paths, y, features, mean, std = _load_data_and_probe_features(cfg, device)

    results = [
        train_one_seed(s, cfg, paths, y, features, mean, std, device)
        for s in cfg["lora"]["seeds"]
    ]

    deltas = [r["lora_val"] - r["probe_val"] for r in results]
    summary = {
        "results": results,
        "mean_lora_val": float(np.mean([r["lora_val"] for r in results])),
        "mean_probe_val": float(np.mean([r["probe_val"] for r in results])),
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas)),
    }

    with open(out / "lora" / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    for r in results:
        print(
            f"seed {r['seed']}: lora={r['lora_val']:.4f} "
            f"probe={r['probe_val']:.4f} delta={r['lora_val'] - r['probe_val']:+.4f}"
        )

    print(
        f"mean delta={summary['mean_delta']:+.4f} +/- {summary['std_delta']:.4f} "
        f"(lora {summary['mean_lora_val']:.4f} vs probe {summary['mean_probe_val']:.4f})"
    )


def kfold_main(config_path="config.yaml"):
    # full k-fold lora oof (per seed) vs the paired frozen-probe oof on the same folds
    cfg = load_config(config_path)
    device = get_device()
    out = Path(cfg["output_dir"])
    (out / "lora").mkdir(parents=True, exist_ok=True)
    paths, y, features, mean, std = _load_data_and_probe_features(cfg, device)

    results = [
        kfold_oof_seed(s, cfg, paths, y, features, mean, std, device)
        for s in cfg["lora"]["seeds"]
    ]

    deltas = [r["lora_oof"] - r["probe_oof"] for r in results]

    summary = {
        "results": results,
        "mean_lora_oof": float(np.mean([r["lora_oof"] for r in results])),
        "mean_probe_oof": float(np.mean([r["probe_oof"] for r in results])),
        "mean_delta": float(np.mean(deltas)),
        "std_delta": float(np.std(deltas)),
    }

    with open(out / "lora" / "kfold_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    for r in results:
        print(
            f"seed {r['seed']}: lora_oof={r['lora_oof']:.4f} "
            f"probe_oof={r['probe_oof']:.4f} delta={r['lora_oof'] - r['probe_oof']:+.4f}"
        )

    print(
        f"mean oof delta={summary['mean_delta']:+.4f} +/- {summary['std_delta']:.4f} "
        f"(lora {summary['mean_lora_oof']:.4f} vs probe {summary['mean_probe_oof']:.4f})"
    )


def train_fold_bundle(seed, cfg, paths, y, mean, std, device):
    # train the k fold-models for one seed; snapshot each fold's best-epoch lora+head state
    set_seed(seed)
    n_classes = int(len(np.unique(y)))
    n_folds = cfg["cv"]["n_folds"]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []

    for k, (tr, va) in enumerate(skf.split(np.zeros((len(y), 1)), y)):
        best_val, _, state = _train_lora_on_split(
            tr, va, cfg, paths, y, mean, std, device, n_classes, f"deploy fold {k}"
        )
        print(f"  deploy fold {k} best val_acc={best_val:.4f}")
        folds.append(state)

    return folds, n_classes


def _predict_test_probs(bundle, cfg, device):
    # softmax-ensemble the fold-models over the test set at the deploy resolution (identity)
    lcfg = cfg["lora"]
    model = build_lora_model(
        bundle["model_name"],
        bundle["img_size"],
        bundle["n_classes"],
        lcfg,
        device,
        bundle.get("pool_mode", "cls_meanpatch"),
    )

    _, eval_tf = build_lora_transforms(
        bundle["img_size"], bundle["mean"], bundle["std"]
    )

    ids, test_paths = list_all_test_images(cfg["data"]["test_dir"])
    test_ds = LoraImageDataset(
        test_paths, np.zeros(len(test_paths), dtype=int), eval_tf
    )

    test_dl = DataLoader(
        test_ds, batch_size=lcfg["batch_size"], shuffle=False, num_workers=0
    )

    probs = None
    for state in bundle["folds"]:
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

    return ids, probs / len(bundle["folds"])


def submission_main(config_path="config.yaml"):
    # train the deploy-seed fold ensemble, save the bundle, write a lora submission
    cfg = load_config(config_path)
    device = get_device()
    out = Path(cfg["output_dir"])
    (out / "lora").mkdir(parents=True, exist_ok=True)

    class_to_idx = build_class_to_idx(cfg["data"]["train_dir"])
    paths, labels = list_train_images(cfg["data"]["train_dir"], class_to_idx)
    y = np.array(labels)
    deploy_seed = cfg["lora"]["seeds"][0]
    img_size = cfg["lora"]["img_size"]
    _, mean, std = load_backbone(cfg["model"]["name"], img_size, device)

    folds, n_classes = train_fold_bundle(deploy_seed, cfg, paths, y, mean, std, device)
    bundle = {
        "folds": folds,
        "n_classes": n_classes,
        "model_name": cfg["model"]["name"],
        "img_size": img_size,
        "class_to_idx": class_to_idx,
        "mean": mean,
        "std": std,
        "deploy_seed": deploy_seed,
        "pool_mode": cfg["lora"].get("pool_mode", "cls_meanpatch"),
    }
    with open(out / "lora" / "lora_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)

    ids, probs = _predict_test_probs(bundle, cfg, device)
    inv = idx_to_class(class_to_idx)
    preds = probs.argmax(axis=1)
    id_to_label = {rid: inv[int(p)] for rid, p in zip(ids, preds)}
    valid_labels = [inv[i] for i in range(len(inv))]

    # non-destructive: keep the frozen baseline submission.csv intact
    write_submission(id_to_label, ids, out / "submission_lora.csv", valid_labels)
    print(
        f"wrote {out / 'submission_lora.csv'} ({len(ids)} rows) "
        f"from lora seed-{deploy_seed} {len(folds)}-fold ensemble"
    )


if __name__ == "__main__":
    args = sys.argv[1:]

    if args and args[0] == "kfold":
        kfold_main(args[1] if len(args) > 1 else "config.yaml")
    elif args and args[0] == "submit":
        submission_main(args[1] if len(args) > 1 else "config.yaml")
    else:
        main(args[0] if args else "config.yaml")
