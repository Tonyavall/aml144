# Fine-tuned SigLIP-2 Classifier

100-class image classification (~10 training images/class, imbalanced: 4-41 per class) for a
Kaggle transfer-learning task. The deployed model is a single LoRA-fine-tuned SigLIP-2 SO400M
backbone with an L2-normalized cosine head and a class-balanced loss. It reaches OOF 0.9484
(3-seed cross-validation) / public LB 0.93636, matching our best ensemble's leaderboard score
at one third of the cost.

This started as a frozen-feature probe and grew through several approaches (frozen multi-view,
a frozen cross-backbone ensemble, a fine-tuned LoRA ensemble) before landing on the single
fine-tuned backbone. The earlier pipelines are preserved under `src/deprecated/` as a record of
the journey; see `docs/flow/experiments.md` (phases A-E) for the full story.

See `docs/spec.md` for the assignment and `src/README.md` for the module layout.

## Acronyms

- OOF - out-of-fold (cross-validation) accuracy: each training image is scored only by the
  fold models that did not see it during training.
- LB - the Kaggle public leaderboard score (estimated on about 10% of the test set).
- CV - cross-validation.
- TTA - test-time augmentation: average the prediction over several views of each test image
  (identity + horizontal flip). Tested and dropped for the deploy (it cost ~1 image).
- LoRA - low-rank adaptation: fine-tune a few low-rank matrices inside the frozen backbone.
- probe - a lightweight linear classifier trained on frozen features (the earlier approach).

## Deployed model

A single SigLIP-2 SO400M (`vit_so400m_patch14_siglip_gap_378.v2_webli`) at 378 px, average-pooled,
LoRA-fine-tuned (r=8, last 4 transformer blocks' attention) with a trainable L2-normalized cosine
head and a class-balanced cross-entropy loss. Evaluated identity-only (no TTA). Deployed as the
seed-42 4-fold softmax ensemble.

| metric | value |
| --- | --- |
| OOF (3-seed shared-fold mean) | 0.9484 +/- 0.0004 |
| public LB | 0.93636 |
| fine-tunes to deploy | 12 (4 folds x 3 seeds for OOF; seed-42 4-fold deployed) |

Design and rationale: `docs/flow/specs/2026-06-03-single-siglip2-design.md`.

## Run training + inference (the deployed model)

One command trains and writes the submission:

```bash
python -m src.single_ft
```

Reads `config.yaml` (the `single_ft` block), runs the 3-seed shared-fold OOF, trains the
seed-42 4-fold deploy ensemble, and writes:

- `outputs/submission_siglip2.csv` - the deployed predictions (columns `ID,Label`, one row per
  test image). Copy it to `outputs/submission.csv` to upload to Kaggle.
- `outputs/single_ft/single_ft_bundle.pkl` - the LoRA adapters + cosine heads (the trained
  weights).
- `outputs/single_ft/metrics.json`, `metadata.json` - OOF scores, the TTA-vs-identity
  comparison, and provenance (git SHA + library versions).

## Earlier approaches (deprecated/)

Preserved for documentation, not the deployed model. Run from the repo root:

| approach | entry point | OOF / LB |
| --- | --- | --- |
| Single-backbone frozen probe | `python -m src.deprecated.train` (with `train_aug_views: 1`) then `python -m src.deprecated.predict` | 0.8851 / ~0.88 |
| Multi-view K=8 frozen probe | `python -m src.deprecated.train` (default config) then `python -m src.deprecated.predict` | 0.9129 / 0.90000 |
| Frozen cross-backbone ensemble | `python -m src.deprecated.ensemble` | 0.9314 / 0.91818 |
| Fine-tuned LoRA ensemble (3 backbones) | `python -m src.deprecated.ensemble_ft` | 0.9404 / 0.93636 |

The single fine-tuned SigLIP-2 matches the LoRA ensemble's LB and beats its OOF while training
3x fewer models and running single-backbone inference at test time.

## Setup

Requires a CUDA 12.8 PyTorch build for the Blackwell GPU (RTX 50-series). Verified working with
Python 3.14.0 and torch 2.9.0+cu128 on an RTX 5070 Ti (16 GB). CPU works but is slow.

```bash
python -m venv .venv
.venv\Scripts\activate   # windows. use source .venv/bin/activate on linux/mac
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # expect True on gpu
```

Run everything from the repo root. Do not run `pytest` from inside `data/` (a stray
`data/train/.pytest_cache` would be read as a class folder).

## Data layout

```
data/
  train/<class>/<n>.jpg   # 100 classes, 1079 images total (imbalanced, 4-41 per class)
  test/<id>.jpg           # 1036 test images, every one is a scored row
  sample_submission.csv   # template for the ID,Label format
```

## Reproducibility

- Fixed seed (`config.yaml: seed`) across `random`, NumPy, and PyTorch; deterministic cuDNN
  and `torch.use_deterministic_algorithms(True, warn_only=True)`.
- The deploy reports a 3-seed (42/43/44) shared-fold OOF mean +/- std. Some bf16 attention ops
  are non-deterministic, so per-seed OOF can drift ~0.2-0.3 pt and may drift more across
  different torch/CUDA/GPU versions.
- `outputs/single_ft/metadata.json` records the git SHA and exact torch/timm/scikit-learn/CUDA
  versions.

## Tests

```bash
pytest -q
```

67 tests cover labels, data listing, submission validation, the probe head, LoRA building blocks
(cosine head, class-balanced weights, TTA eval), the shared fold helpers, the single_ft
orchestrator, and the deprecated pipelines (multi-view grouped CV, Sinkhorn, fusion, ensemble).
The LoRA tests need `timm`/`peft`, so use the project venv.

## Diagrams

`docs/diagrams/deploy.svg` is a high-level view of the deployed model:

![deployed model](docs/diagrams/deploy.svg)

The earlier, denser d2 pipeline diagrams are archived under `docs/archive/diagrams/` as a
historical record; they document the now-deprecated pipelines. The deployed model's full design
is in `docs/flow/specs/2026-06-03-single-siglip2-design.md`.

## Trained weights

The deployed model bundle (`outputs/single_ft/single_ft_bundle.pkl`) is available at:
<GOOGLE_DRIVE_LINK>.
