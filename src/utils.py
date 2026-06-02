import os
import random
import subprocess

import numpy as np
import torch
import yaml


def set_seed(seed):
    # seed every rng and force deterministic cudnn for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_device():
    # cuda when available, cpu fallback keeps the pipeline runnable anywhere
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def collect_metadata(cfg, extra=None):
    # provenance for reproducibility: git sha plus exact library versions
    import sklearn
    import timm

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = "unknown"
    meta = {
        "git_sha": sha,
        "torch": torch.__version__,
        "timm": timm.__version__,
        "sklearn": sklearn.__version__,
        "cuda": torch.version.cuda,
        "config": cfg,
    }
    if extra:
        meta.update(extra)
    return meta
