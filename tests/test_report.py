import json

import numpy as np

from src.report import (
    confusion_counts,
    fold0_series,
    load_ledger,
    render_table,
)


def test_load_ledger_reads_rows(tmp_path):
    p = tmp_path / "ledger.csv"
    p.write_text(
        "phase,approach,backbones,pipeline,key_hyperparams,oof,public_lb,status,notes\n"
        "D,ft ens,a+b,lora ensemble,r8,0.9444,,best,notes here\n",
        encoding="ascii",
    )
    rows = load_ledger(p)
    assert len(rows) == 1
    assert rows[0]["approach"] == "ft ens"
    assert rows[0]["public_lb"] == ""


def test_render_table_header_rows_and_empty_lb():
    rows = [
        {"phase": "D", "approach": "ft ens", "pipeline": "lora ensemble",
         "key_hyperparams": "r8", "oof": "0.9444", "public_lb": "", "status": "best"},
    ]
    md = render_table(rows)
    lines = md.strip().splitlines()
    assert lines[0].startswith("| phase | approach |")
    assert lines[1].count("---") >= 5
    assert "ft ens" in md
    assert "0.9444" in md
    assert "| - |" in md
    assert md.endswith("\n")


def test_fold0_series_extracts_metrics():
    hist = {
        "dinov3": {
            "0": [
                {"epoch": 0, "train_loss": 1.0, "train_acc": 0.5, "val_loss": 0.9, "val_acc": 0.6},
                {"epoch": 1, "train_loss": 0.5, "train_acc": 0.7, "val_loss": 0.6, "val_acc": 0.8},
            ],
            "1": [{"epoch": 0, "train_loss": 9, "train_acc": 0, "val_loss": 9, "val_acc": 0}],
        }
    }
    s = fold0_series(hist)
    assert s["dinov3"]["val_acc"] == [0.6, 0.8]
    assert s["dinov3"]["train_loss"] == [1.0, 0.5]
    hist_int_key = {"dinov3": {0: hist["dinov3"]["0"]}}
    s2 = fold0_series(hist_int_key)
    assert s2["dinov3"]["val_acc"] == [0.6, 0.8]


def test_confusion_counts_basic():
    preds = np.array([0, 1, 1, 2])
    y = np.array([0, 1, 2, 2])
    cm = confusion_counts(preds, y, 3)
    assert cm[0, 0] == 1
    assert cm[1, 1] == 1
    assert cm[2, 1] == 1
    assert cm[2, 2] == 1
    assert cm.sum() == 4


def test_main_writes_table_and_figures(tmp_path):
    out_dir = tmp_path / "report"
    data_dir = tmp_path / "data"
    out_dir.mkdir()
    data_dir.mkdir()

    (out_dir / "results_ledger.csv").write_text(
        "phase,approach,backbones,pipeline,key_hyperparams,oof,public_lb,status,notes\n"
        "D,ft ens,a+b,lora ensemble,r8,0.9444,,best,n\n"
        "A,multiview,a,frozen probe,k8,0.9129,0.90000,deployed,n\n",
        encoding="ascii",
    )
    hist = {
        "dinov3": {"0": [{"epoch": 0, "train_loss": 1.0, "train_acc": 0.5, "val_loss": 0.9, "val_acc": 0.6}]},
        "siglip2": {"0": [{"epoch": 0, "train_loss": 1.1, "train_acc": 0.4, "val_loss": 1.0, "val_acc": 0.55}]},
    }
    (data_dir / "history.json").write_text(json.dumps(hist), encoding="ascii")
    rng = np.random.default_rng(0)
    probs = rng.random((20, 5))
    y = rng.integers(0, 5, size=20)
    np.savez(data_dir / "oof_blend.npz", probs=probs, y=y)

    from src.report import main

    main(out_dir=out_dir, data_dir=data_dir)

    assert (out_dir / "results_table.md").exists()
    assert (out_dir / "figures" / "training_curves.png").exists()
    assert (out_dir / "figures" / "results_comparison.png").exists()
    assert (out_dir / "figures" / "confusion_matrix_ft_ensemble.png").exists()
