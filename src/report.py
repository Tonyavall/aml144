import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT.parent.parent / "docs" / "report"  # cse144/docs/report
DEFAULT_DATA = REPO_ROOT / "outputs" / "ensemble_ft"

# short display names for the long timm backbone ids, for figure legends
SHORT_NAMES = {
    "vit_large_patch16_dinov3.lvd1689m": "dinov3",
    "vit_so400m_patch14_siglip_gap_378.v2_webli": "siglip2",
    "aimv2_large_patch14_336.apple_pt_dist": "aimv2",
}

TABLE_COLUMNS = [
    "phase",
    "approach",
    "pipeline",
    "key_hyperparams",
    "oof",
    "public_lb",
    "status",
]


def load_ledger(path):
    # read the results ledger csv into a list of dict rows
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def render_table(rows):
    # render the ledger rows as a github-flavored markdown table (a readable column subset)
    header = "| " + " | ".join(TABLE_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |"
    lines = [header, sep]
    for r in rows:
        vals = [(r.get(c) or "-") for c in TABLE_COLUMNS]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def fold0_series(history_json):
    # history_json: {member_name: {fold_index_str: [ {epoch, train_loss, ...}, ... ]}}
    # return {member_name: {metric: [values over epochs]}} for fold 0
    metrics = ("train_loss", "train_acc", "val_loss", "val_acc")
    out = {}
    for member, folds in history_json.items():
        epochs = folds.get("0", folds.get(0, []))
        out[member] = {m: [e[m] for e in epochs] for m in metrics}
    return out


def confusion_counts(preds, y, n):
    # n x n confusion matrix; rows are true labels, columns are predictions
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y, preds):
        cm[int(t), int(p)] += 1
    return cm


def _short(name):
    return SHORT_NAMES.get(name, name)


def plot_training_curves(series, out_path):
    # 2-panel hw2-style figure: loss and accuracy vs epoch, all members,
    # train dashed and val solid. one color per member.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for member, s in series.items():
        epochs = list(range(len(s["train_loss"])))
        line = ax1.plot(
            epochs, s["train_loss"], linestyle="--", label=f"{_short(member)} train"
        )[0]
        ax1.plot(
            epochs,
            s["val_loss"],
            linestyle="-",
            color=line.get_color(),
            label=f"{_short(member)} val",
        )
        line2 = ax2.plot(
            epochs, s["train_acc"], linestyle="--", label=f"{_short(member)} train"
        )[0]
        ax2.plot(
            epochs,
            s["val_acc"],
            linestyle="-",
            color=line2.get_color(),
            label=f"{_short(member)} val",
        )

    ax1.set_title("loss over epochs")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(fontsize=7)
    ax2.set_title("accuracy over epochs")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_results_comparison(rows, out_path):
    # grouped bar chart: oof for every approach, public lb where available
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["approach"] for r in rows]
    oof = [float(r["oof"]) if r.get("oof") else np.nan for r in rows]
    lb = [float(r["public_lb"]) if r.get("public_lb") else np.nan for r in rows]
    x = np.arange(len(labels))
    w = 0.4

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, oof, w, label="oof")
    ax.bar(x + w / 2, lb, w, label="public lb")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("accuracy")
    ax.set_title("results by approach: oof vs public lb")
    vals = [v for v in oof + lb if not np.isnan(v)]
    lo = max(0.0, min(vals) - 0.02) if vals else 0.0
    hi = min(1.0, max(vals) + 0.01) if vals else 1.0
    ax.set_ylim(lo, hi)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(npz_path, out_path):
    # confusion matrix heatmap from the fine-tuned ensemble deploy-seed oof predictions
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(npz_path)
    preds = d["probs"].argmax(axis=1)
    y = d["y"]
    n = int(d["probs"].shape[1])
    cm = confusion_counts(preds, y, n)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("fine-tuned ensemble oof confusion matrix")
    fig.colorbar(im, ax=ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _crosscheck_oof(rows, data_dir):
    # rigor check: warn if the ledger's fine-tuned-ensemble oof disagrees with the
    # regenerated metrics.json blend oof by more than ~0.005
    metrics_path = Path(data_dir) / "metrics.json"
    if not metrics_path.exists():
        return
    with open(metrics_path) as f:
        metrics = json.load(f)
    measured = metrics.get("blend_oof_acc")
    for r in rows:
        if r.get("pipeline") == "lora ensemble" and r.get("oof"):
            if measured is not None and abs(float(r["oof"]) - measured) > 0.005:
                print(
                    f"WARNING: ledger oof {r['oof']} for '{r['approach']}' disagrees "
                    f"with metrics.json blend_oof_acc {measured:.4f}"
                )


def main(out_dir=DEFAULT_OUT, data_dir=DEFAULT_DATA):
    out_dir = Path(out_dir)
    data_dir = Path(data_dir)
    figs = out_dir / "figures"

    rows = load_ledger(out_dir / "results_ledger.csv")
    (out_dir / "results_table.md").write_text(render_table(rows), encoding="ascii")
    _crosscheck_oof(rows, data_dir)
    plot_results_comparison(rows, figs / "results_comparison.png")

    history_path = data_dir / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            series = fold0_series(json.load(f))
        plot_training_curves(series, figs / "training_curves.png")
    else:
        print(f"skipped training_curves: {history_path} not found")

    oof_path = data_dir / "oof_blend.npz"
    if oof_path.exists():
        plot_confusion_matrix(oof_path, figs / "confusion_matrix_ft_ensemble.png")
    else:
        print(f"skipped confusion_matrix: {oof_path} not found")

    print(f"wrote report artifacts to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    args = ap.parse_args()
    main(out_dir=args.out, data_dir=args.data)
