# DINOv3 Frozen-Feature Classifier

100-class image classification (~10 training images/class, imbalanced: 4-41 per class) for a
Kaggle transfer-learning task. The approach is a frozen vision backbone (no fine-tuning) that
produces feature vectors, plus a scikit-learn multinomial logistic-regression "probe" that
classifies them. Three increasingly capable pipelines share the same core (see below).

See `docs/spec.md` for the assignment and `docs/architecture.md` for the design and diagrams.

## Acronyms

- OOF - out-of-fold (cross-validation) accuracy: each training image is scored only by the
  fold models that did not see it during training.
- LB - the Kaggle public leaderboard score (estimated on about 10% of the test set).
- CV - cross-validation.
- TTA - test-time augmentation: average the prediction over several views of each test image
  (here identity + horizontal flip).
- CLS - the transformer "class" token, a per-image summary token.
- LoRA - low-rank adaptation, the experimental fine-tuning path.
- probe - the lightweight linear classifier (logistic regression) trained on frozen features.
- C - the inverse L2 regularization strength of the logistic regression (smaller C = stronger
  regularization).

## Approach

- Backbone: `vit_large_patch16_dinov3.lvd1689m` (via `timm`, no gating), frozen - no
  fine-tuning. The cross-backbone ensemble adds frozen SigLIP-2 and AIMv2 as well.
- Features: for DINOv3, the CLS token concatenated with mean-pooled patch tokens (2048-d),
  extracted once and cached to disk. Feature dim is backbone-specific: DINOv3 2048-d,
  SigLIP-2 1152-d, AIMv2 1024-d.
- Head: a cosine logistic probe - features are L2-normalized per sample (turning the logistic
  head into a cosine classifier), then a multinomial `LogisticRegression` with
  `class_weight="balanced"`. The L2 strength C is chosen by RepeatedStratifiedKFold
  (4 folds x 3 repeats); the per-fold models form a softmax-averaged ensemble.
- Inference: test-time augmentation (identity + horizontal flip), averaged over views and
  fold models. Optional Sinkhorn class-balancing toward the uniform test prior (off by
  default).

## The three pipelines

| Pipeline | Entry point | What it does | OOF / LB |
| --- | --- | --- | --- |
| 1. Single-backbone probe | `python -m src.train` (with `train_aug_views: 1`) then `python -m src.predict` | one frozen backbone -> cosine probe -> submission | 0.8851 / ~0.88 |
| 2. Multi-view DINOv3 (deployed) | `python -m src.train` (default config) then `python -m src.predict` | 8 augmented feature views + leak-free grouped CV | 0.9129 / 0.90000 |
| 3. Cross-backbone ensemble | `python -m src.ensemble` | DINOv3 + SigLIP-2 + AIMv2, late fusion of probabilities | 0.9314 / not yet measured |

`train.py` automatically runs pipeline 2 when `config.yaml: features.train_aug_views > 1`
(it is 8 by default), and pipeline 1 otherwise. See `outputs/model/metrics.json` and
`outputs/ensemble/ensemble_metrics.json` for exact per-run numbers.

Module layout (arrows mean "imports from"):

![module dependency graph](docs/diagrams/modules.svg)

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

## Train (pipeline 1 / 2)

```bash
python -m src.train
```

Reads `config.yaml`, extracts and caches features, selects the L2 strength by
RepeatedStratifiedKFold (folds clamped to the smallest class), fits the fold ensemble, and
writes `outputs/model/` (`bundle.pkl`, `class_to_idx.json`, `metrics.json`, `metadata.json`)
plus `outputs/figures/`. With the default `train_aug_views: 8` this is the deployed multi-view
pipeline; set it to `1` for the single-view baseline.

## Predict / make a submission

```bash
python -m src.predict
```

Loads `outputs/model/bundle.pkl`, applies TTA (and optional Sinkhorn), and writes
`outputs/submission.csv` (columns `ID,Label`), one row per test image. Upload it to the
Kaggle competition page.

## Ensemble (pipeline 3)

```bash
python -m src.ensemble
```

Trains a frozen probe per backbone listed under `config.yaml: backbones`, blends their
out-of-fold probabilities with gated simplex-search weights (tuned weights are adopted only
if they beat the equal-weight blend by `ensemble.weight_margin`), and writes
`outputs/submission_ensemble.csv` plus `outputs/ensemble/`. This is non-destructive: it does
not overwrite the pipeline 1/2 `submission.csv` or `bundle.pkl`.

## LoRA fine-tuning (experimental)

```bash
python -m src.models.lora_train          # paired val: lora vs frozen probe over seeds
python -m src.models.lora_train kfold     # full k-fold oof comparison
python -m src.models.lora_train submit     # train deploy folds, write submission_lora.csv
```

An optional path that adapts the last few transformer blocks with LoRA plus a trainable head.
It is measured against the paired frozen-probe baseline on the same folds; the frozen probe
remains the deployed model.

## Reproducibility

- Fixed seed (`config.yaml: seed`) across `random`, NumPy, and PyTorch; deterministic cuDNN
  and `torch.use_deterministic_algorithms(True, warn_only=True)`.
- The cosine logistic head is convex/deterministic; CV reports mean +/- std over the folds.
- `outputs/model/metadata.json` records the git SHA and exact torch/timm/scikit-learn/CUDA
  versions. The sklearn probe paths are deterministic; GPU feature extraction has ~0.0-0.2 pt
  seed drift because some bf16 ops are not deterministic, and may drift more across different
  torch/CUDA/GPU versions.

## Tests

```bash
pytest -q
```

37 tests cover labels, data listing, submission validation, the probe head, multi-view
grouped CV, Sinkhorn, backbone pooling, fusion, ensemble orchestration, and LoRA (the LoRA
tests need `timm`/`peft`, so use the project venv).

## Diagrams

D2 sources and rendered SVGs live in `docs/diagrams/` (see `docs/diagrams/README.md` for how
to re-render with the `d2` CLI):

- `modules` - module dependency graph (the layering).
- `pipeline1` - single-backbone train + predict data flow.
- `pipeline2` - multi-view DINOv3 with leak-free grouped CV.
- `pipeline3` - cross-backbone ensemble fan-in and the weight-tuning gate.
- `probe` - the cosine logistic probe and its cross-validation internals.

## Trained weights

The deployed model bundle (`outputs/model/`) is available at: <GOOGLE_DRIVE_LINK>.
