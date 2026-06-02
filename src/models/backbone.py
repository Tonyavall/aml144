import re
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image

from src.data.images import build_aug_transform, build_transform
from src.utils import set_seed


def load_backbone(model_name, img_size, device):
    # frozen dinov3 feature extractor; resolve the model's own normalization
    model = timm.create_model(
        model_name, pretrained=True, num_classes=0, img_size=img_size
    )

    model.eval().to(device)
    dc = timm.data.resolve_model_data_config(model)

    return model, list(dc["mean"]), list(dc["std"])


@torch.no_grad()
def _embed(model, x, num_prefix, pool="cls_meanpatch"):
    # dinov3: cls token concatenated with mean-pooled patch tokens.
    # default: the model's own pooled embedding (contrastive backbones with no usable cls)
    if pool == "default":
        return model(x)

    tokens = model.forward_features(x)
    cls = tokens[:, 0]
    patches = tokens[:, num_prefix:].mean(dim=1)

    return torch.cat([cls, patches], dim=1)


def extract_features(model, paths, view, cfg, mean, std, device, cache_path, pool="cls_meanpatch"):
    # return a (n, d) float32 array aligned to paths, caching by exact path list
    # (d depends on the backbone and pool: dinov3 cls+meanpatch=2048, siglip2=1152, aimv2=1024)
    cache_path = Path(cache_path)
    key = [str(p) for p in paths]

    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if list(cached["ids"]) == key:
            return cached["features"]

    transform = build_transform(cfg["model"]["img_size"], mean, std, view)

    num_prefix = model.num_prefix_tokens
    batch_size = cfg["batch_size"]
    feats = []

    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]

        batch = torch.stack(
            [transform(Image.open(p).convert("RGB")) for p in chunk]
        ).to(device)

        feats.append(_embed(model, batch, num_prefix, pool).cpu().numpy())

    features = np.concatenate(feats, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(cache_path, features=features, ids=np.array(key))

    return features


def extract_multiview_features(model, paths, n_views, cfg, mean, std, device, cache_dir, seed, pool="cls_meanpatch"):
    # view 0 is the deterministic identity view; views 1..n-1 are seeded random crops.
    # each view is cached separately and reproducible; returns a list of (n, d) arrays.
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = re.sub(r"[^0-9a-zA-Z]+", "_", cfg["model"]["name"])
    img_size = cfg["model"]["img_size"]
    num_prefix = model.num_prefix_tokens
    batch_size = cfg["batch_size"]
    key = [str(p) for p in paths]

    views = []

    for v in range(n_views):
        cache_path = cache_dir / f"trainaug__{tag}__v{v}__{img_size}.npz"

        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            if list(cached["ids"]) == key:
                views.append(cached["features"])
                continue

        if v == 0:
            transform = build_transform(img_size, mean, std, "identity")
        else:
            # deterministic per-view seed so the random crops are reproducible
            set_seed(seed + v)
            transform = build_aug_transform(img_size, mean, std)

        feats = []

        # single-threaded loop keeps the seeded crop sequence reproducible
        for start in range(0, len(paths), batch_size):
            chunk = paths[start : start + batch_size]
            batch = torch.stack(
                [transform(Image.open(p).convert("RGB")) for p in chunk]
            ).to(device)
            feats.append(_embed(model, batch, num_prefix, pool).cpu().numpy())

        features = np.concatenate(feats, axis=0).astype(np.float32)
        np.savez(cache_path, features=features, ids=np.array(key))
        views.append(features)

    return views
