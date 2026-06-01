# DINOv3 Frozen-Feature Classifier

100-class image classification (~10 training images/class, imbalanced) using a frozen
DINOv3 ViT-L/16 feature extractor with a stratified-fold logistic-regression ensemble.

See `docs/spec.md` for the task.

## Approach

- Backbone: `vit_large_patch16_dinov3.lvd1689m` (via `timm`, no gating), frozen - no fine-tuning.
- Features: CLS token concatenated with mean-pooled patch tokens (2048-d), extracted once at
  256x256 and cached to disk.
- Head: per-fold StandardScaler + multinomial logistic regression. The L2 strength C is chosen
  by stratified k-fold cross-validation; the k fold-models form a softmax-averaged ensemble.
- Inference: test-time augmentation (identity + horizontal flip), averaged over views and folds.

## Results

- Out-of-fold cross-validation accuracy: **~88.5%** (best C = 0.1, 4 folds). See
  `outputs/model/metrics.json` for the per-C curve and exact numbers.

## Setup

Requires a CUDA 12.8 PyTorch build for the Blackwell GPU (RTX 50-series). Verified working with
Python 3.14.0 and torch 2.9.0+cu128 on an RTX 5070 Ti. CPU works but is slow.

```bash
python -m venv .venv
.venv\Scripts\activate # windows. use source .venv/bin/activate on linux/mac
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())" # expect True on gpu
```

## Data layout

```
data/
  train/<class>/<n>.jpg # 100 classes, ~10 images each (imbalanced; 1079 total)
  test/<id>.jpg         # test images
  sample_submission.csv # defines the scored ids and the ID,Label format
```

## Train

```bash
python -m src.train
```

Reads `config.yaml`, extracts and caches DINOv3 features, selects the L2 strength by stratified
k-fold CV (folds clamped to the smallest class; 4 here), fits the fold ensemble, and writes
`outputs/model/` (`bundle.pkl`, `class_to_idx.json`, `metrics.json`, `metadata.json`) plus
`outputs/figures/`.

## Predict / make a submission

```bash
python -m src.predict
```

Writes `outputs/submission.csv` (columns `ID,Label`), matching `data/sample_submission.csv`
row for row. Upload it to the Kaggle competition page.

## Reproducibility

- Fixed seed (`config.yaml: seed`) across `random`, NumPy, and PyTorch; deterministic cuDNN.
- The logistic-regression head is convex/deterministic; CV reports mean +/- std over the folds.
- `outputs/model/metadata.json` records the git SHA and exact torch/timm/scikit-learn/CUDA
  versions. Results reproduce on the same library and GPU versions; feature extraction is the
  only nondeterministic step and may drift slightly across torch/CUDA/GPU versions.

## Tests

```bash
pytest -v
```

## Trained weights

The model bundle (`outputs/model/`) is available at: <GOOGLE_DRIVE_LINK>.

## Fallback (if accuracy disappoints)

If CV accuracy is near the 60% baseline, the domain likely shifts from DINOv3's training data.
The planned escalation is LoRA (rank 8) on the last 4-8 transformer blocks plus the head,
trained end to end (still fits 16 GB). Not implemented in this version.
