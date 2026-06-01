import pickle
import re
import sys
from pathlib import Path

from src.data import list_all_test_images
from src.features import extract_features, load_backbone
from src.head import predict_proba
from src.labels import idx_to_class
from src.submission import write_submission
from src.utils import get_device, load_config, set_seed


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    out = Path(cfg["output_dir"])

    with open(out / "model" / "bundle.pkl", "rb") as f:
        bundle = pickle.load(f)

    inv = idx_to_class(bundle["class_to_idx"])

    # kaggle scores every image in test/ (the sample_submission.csv template is stale),
    # so we predict all test images, not just the template ids
    required_ids, paths = list_all_test_images(cfg["data"]["test_dir"])

    model, mean, std = load_backbone(bundle["model_name"], bundle["img_size"], device)
    tag = re.sub(r"[^0-9a-zA-Z]+", "_", bundle["model_name"])
    probs = None

    for view in cfg["features"]["views_test"]:
        cache = out / "cache" / f"test__{tag}__{view}__{bundle['img_size']}.npz"

        x = extract_features(model, paths, view, cfg, mean, std, device, cache)
        p = predict_proba(bundle["fold_models"], x)

        probs = p if probs is None else probs + p

    probs /= len(cfg["features"]["views_test"])

    preds = probs.argmax(axis=1)

    id_to_label = {rid: inv[int(pred)] for rid, pred in zip(required_ids, preds)}

    valid_labels = [inv[i] for i in range(len(inv))]

    write_submission(id_to_label, required_ids, out / "submission.csv", valid_labels)

    print(f"wrote {out / 'submission.csv'} ({len(required_ids)} rows)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
