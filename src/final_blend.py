"""two-branch final blend: frozen hf siglip-2 members + the deployed fine-tune.

branch a holds logistic probes and text zero-shot members on frozen siglip-2
giant-opt-384 and so400m-512 embeddings; branch b is the deployed single_ft
fold ensemble reloaded from its bundle with no retraining. the five oof
probability matrices are blended with margin-gated grid-search weights tuned
on the shared seed-42 folds and applied to the test set.

typical usage example:

  python -m src.final_blend
"""

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from src.data.images import list_all_test_images, list_train_images
from src.data.labels import build_class_to_idx, idx_to_class
from src.deprecated.fusion import blend, tune_weights
from src.models.head import _fit_one, l2_normalize, oof_accuracy
from src.models.hf_siglip2 import backbone_pack, zero_shot_probs
from src.models.lora_members import member_test_probs, shared_folds
from src.submission import write_submission
from src.utils import collect_metadata, get_device, load_config, set_seed


def probe_member(x, y, folds, c_grid):
    """fits per-fold logistic probes on shared folds and picks c by oof accuracy.

    returns (oof, fold_models, best_c, best_acc) where oof is the (n, k)
    out-of-fold probability matrix at the winning c and fold_models are that
    c's per-fold models in shared-fold order.
    """
    n_classes = len(np.unique(y))
    best = None
    for c in c_grid:
        oof = np.zeros((len(y), n_classes))
        models = []
        for tr, va in folds:
            clf = _fit_one(x[tr], y[tr], c)
            oof[va] = clf.predict_proba(l2_normalize(x[va]))
            models.append(clf)
        acc = oof_accuracy(oof, y)
        if best is None or acc > best[3]:
            best = (oof, models, c, acc)
    return best


def load_class_names(csv_path, class_to_idx):
    """returns class names ordered by model index from the checked-in csv.

    Raises:
        ValueError: if any class id from class_to_idx has no row in the csv.
    """
    rows = {}
    with open(csv_path, newline="", encoding="ascii") as f:
        for row in csv.DictReader(f):
            rows[row["class_id"]] = row["name"].strip()
    inv = idx_to_class(class_to_idx)
    missing = [inv[i] for i in range(len(inv)) if inv[i] not in rows]
    if missing:
        raise ValueError(f"class_names.csv missing ids: {missing[:5]}")
    return [rows[inv[i]] for i in range(len(inv))]


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device()
    out = Path(cfg["output_dir"])
    (out / "final_blend").mkdir(parents=True, exist_ok=True)
    (out / "cache").mkdir(parents=True, exist_ok=True)

    class_to_idx = build_class_to_idx(cfg["data"]["train_dir"])
    paths, labels = list_train_images(cfg["data"]["train_dir"], class_to_idx)
    y = np.array(labels)
    test_ids, test_paths = list_all_test_images(cfg["data"]["test_dir"])

    fcfg = cfg["final_blend"]
    names = load_class_names(fcfg["class_names"], class_to_idx)
    prompts = [fcfg["prompt_template"].format(name=n) for n in names]

    deploy_seed = cfg["single_ft"]["deploy_seed"]
    folds = shared_folds(y, cfg["cv"]["n_folds"], deploy_seed)

    # branch a: per-backbone frozen probe + text zero-shot members
    members = {}
    probe_meta = {}
    for b in fcfg["backbones"]:
        pack = backbone_pack(
            b["model_id"], b["tag"], paths, test_paths, prompts,
            b["batch_size"], device, out / "cache",
        )
        oof, models, best_c, acc = probe_member(
            pack["train"], y, folds, cfg["cv"]["c_grid"]
        )
        test_probe = np.mean(
            [m.predict_proba(l2_normalize(pack["test"])) for m in models], axis=0
        )
        members[f"probe_{b['tag']}"] = {"oof": oof, "test": test_probe}
        probe_meta[b["tag"]] = {"best_c": best_c, "oof_acc": acc, "models": models}
        members[f"text_{b['tag']}"] = {
            "oof": zero_shot_probs(
                pack["train"], pack["text"], pack["scale_exp"], pack["bias"]
            ),
            "test": zero_shot_probs(
                pack["test"], pack["text"], pack["scale_exp"], pack["bias"]
            ),
        }

    # branch b: reload the deployed fold ensemble, no retraining. the import is
    # deferred so the unit tests for this module never pull torch-heavy code.
    from src.single_ft import member_val_oof

    with open(out / "single_ft" / "single_ft_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    member = cfg["single_ft"]["member"]
    ft_oof = member_val_oof(
        member, bundle["folds"], bundle["n_classes"], bundle["mean"],
        bundle["std"], paths, y, bundle["deploy_seed"], cfg, device, ["identity"],
    )
    ft_test = member_test_probs(
        member, bundle["folds"], bundle["n_classes"], bundle["mean"],
        bundle["std"], test_paths, cfg, device,
    )
    members["finetuned_siglip2"] = {"oof": ft_oof, "test": ft_test}

    order = list(members)
    oof_list = [members[k]["oof"] for k in order]
    member_acc = {k: oof_accuracy(members[k]["oof"], y) for k in order}
    weights, tuned, equal_acc, tuned_acc = tune_weights(
        oof_list, y, step=fcfg["weight_step"], margin=fcfg["weight_margin"]
    )
    blend_oof = blend(oof_list, weights)
    blend_acc = oof_accuracy(blend_oof, y)

    test_blend = blend([members[k]["test"] for k in order], weights)
    inv = idx_to_class(class_to_idx)
    preds = test_blend.argmax(axis=1)
    id_to_label = {rid: inv[int(p)] for rid, p in zip(test_ids, preds)}
    valid_labels = [inv[i] for i in range(len(inv))]
    write_submission(
        id_to_label, test_ids, out / fcfg["output_submission"], valid_labels
    )

    # deploy gate from the spec: must beat the single fine-tune by > 0.005 oof
    gate = blend_acc > 0.9484 + 0.005

    metrics = {
        "member_oof_acc": member_acc,
        "probe_best_c": {t: probe_meta[t]["best_c"] for t in probe_meta},
        "weights": dict(zip(order, weights)),
        "weights_tuned": tuned,
        "equal_weight_acc": equal_acc,
        "tuned_weight_acc": tuned_acc,
        "blend_oof_acc": blend_acc,
        "deploy_gate_passed": gate,
    }
    with open(out / "final_blend" / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out / "final_blend" / "metadata.json", "w") as f:
        json.dump(collect_metadata(cfg, {"blend_oof_acc": blend_acc}), f, indent=2)
    bundle_out = {
        "order": order,
        "weights": weights,
        "probe_models": {t: probe_meta[t]["models"] for t in probe_meta},
        "class_names": names,
        "prompts": prompts,
        "deploy_seed": deploy_seed,
    }
    with open(out / "final_blend" / "blend_bundle.pkl", "wb") as f:
        pickle.dump(bundle_out, f)

    for k in order:
        print(f"  {k}: oof {member_acc[k]:.4f} weight {dict(zip(order, weights))[k]:.2f}")
    print(
        f"blend oof {blend_acc:.4f}"
        f" (equal {equal_acc:.4f}, tuned {tuned_acc:.4f}, tuned_used={tuned})"
    )
    print(f"deploy gate (> 0.9534): {'PASS' if gate else 'FAIL'}")
    print(f"wrote {out / fcfg['output_submission']} ({len(test_ids)} rows)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
