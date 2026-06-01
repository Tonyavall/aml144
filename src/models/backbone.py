from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image

from src.data.images import build_transform


def load_backbone(model_name, img_size, device):
    # frozen dinov3 feature extractor; resolve the model's own normalization
    model = timm.create_model(
        model_name, pretrained=True, num_classes=0, img_size=img_size
    )

    model.eval().to(device)
    dc = timm.data.resolve_model_data_config(model)

    return model, list(dc["mean"]), list(dc["std"])


@torch.no_grad()
def _embed(model, x, num_prefix):
    # cls token (index 0) concatenated with mean-pooled patch tokens
    tokens = model.forward_features(x)
    cls = tokens[:, 0]
    patches = tokens[:, num_prefix:].mean(dim=1)

    return torch.cat([cls, patches], dim=1)


def extract_features(model, paths, view, cfg, mean, std, device, cache_path):
    # return a (n, 2048) float32 array aligned to paths, caching by exact path list
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

        feats.append(_embed(model, batch, num_prefix).cpu().numpy())

    features = np.concatenate(feats, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(cache_path, features=features, ids=np.array(key))

    return features
