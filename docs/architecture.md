# Architecture

This document describes the design behind the deployed image classifier and the journey that
produced it. The deployed model is a two-branch blend (`src/final_blend.py`): a frozen branch
of SigLIP-2 probes and text zero-shot members over two larger HuggingFace checkpoints, blended
with the single LoRA-fine-tuned SigLIP-2 backbone (which is branch B). The module guide is in
`src/README.md`; the full experimental write-up is in the project report
(`docs/CSE144_Final_Report.pdf`). The previous frozen-probe-era version of this document is
archived at `docs/archive/architecture-v1-2026-06-03.md`.

## Acronyms

- OOF - out-of-fold (cross-validation) accuracy: each training image is scored only by the
  fold models that did not see it during training.
- LB - the Kaggle public leaderboard score (estimated on about 10% of the test set).
- CV - cross-validation.
- TTA - test-time augmentation: average the prediction over several views of each test image
  (identity + horizontal flip). Tested and dropped for the deploy (it cost about 1 image).
- CLS - the transformer "class" token, a per-image summary token.
- gap - global average pooling (how SigLIP-2 produces its pooled embedding).
- LoRA - low-rank adaptation: fine-tune a few low-rank matrices inserted into the frozen
  backbone's attention, leaving the pretrained weights frozen.
- probe - a lightweight linear classifier (logistic regression) on frozen features; the earlier
  (now deprecated) approach.

## Overview

The task is 100-class classification with only ~10 (4-41) training images per class. Full
fine-tuning of a large vision transformer would overfit and is slow. The project worked through
a series of approaches (a frozen linear probe, multi-view feature augmentation, a frozen
cross-backbone ensemble, a fine-tuned LoRA ensemble) and found that a single LoRA-fine-tuned
SigLIP-2 backbone matched the best ensemble's leaderboard score at a fraction of the cost. That
single backbone was the deploy until the final evening, when a frozen branch (probes + text
zero-shot on two larger SigLIP-2 checkpoints) was blended with it; the two-branch blend is the
deployed model. The earlier pipelines are preserved under `src/deprecated/` as a record of the
journey.

Module layout:

- `src/final_blend.py` - the deployed model's orchestrator (the two-branch blend).
- `src/models/hf_siglip2.py` - HuggingFace SigLIP-2 feature extraction (image and text towers)
  for branch A; unwraps the transformers 5.x model-output object via `pooler_output`.
- `src/data/class_names.csv` - the 100 class names (derived by viewing 1-2 training images per
  class); required by the text members.
- `src/single_ft.py` - branch B's trainer (the fine-tuned SigLIP-2); also runs standalone.
- `src/models/lora_train.py` - LoRA fine-tuning primitives: the backbone+LoRA model, the
  cosine/linear head, class-balanced loss, TTA eval, and per-split training.
- `src/models/lora_members.py` - per-member fold helpers (shared CV folds, per-member OOF
  training, test softmax-ensembling), shared by `single_ft` and the deprecated `ensemble_ft`.
- `src/models/{backbone,head,balance}.py`, `src/data/`, `src/submission.py`, `src/utils.py` -
  the shared foundation: feature extraction, the probe head, Sinkhorn balancing, IO, and
  config/seeding.
- `src/deprecated/` - the earlier pipelines (`train`, `predict`, `ensemble`, `ensemble_ft`,
  `multiview`, `fusion`).

The deployed flow is shown in `diagrams/deploy.svg` (below). The earlier, denser d2 pipeline
diagrams are archived under `docs/archive/diagrams/` as a historical record of the journey; they
document the now-deprecated pipelines.

## Deployed model: two-branch blend

The deployed model blends a frozen branch (A) with the fine-tuned backbone (B).

Branch A (frozen) runs two HuggingFace SigLIP-2 checkpoints,
`google/siglip2-giant-opt-patch16-384` ("gopt-384") and
`google/siglip2-so400m-patch16-512` ("so400m-512"). Each checkpoint contributes two members:

- a logistic-regression probe (scikit-learn) on L2-normalized shared-space image embeddings,
  trained per-fold on the SAME shared seed-42 `StratifiedKFold(4)` folds as branch B. Best C
  from the existing grid: 10.0 for gopt-384, 30.0 for so400m-512.
- a text zero-shot member: class-name prompts ("a photo of a {name}.") through the checkpoint's
  text tower, softmax over the 100 classes of scaled image-text cosine logits. No training.

Branch B (existing) is the LoRA fine-tuned SigLIP-2 fold ensemble, reloaded from its saved
bundle with no retraining (identity-only eval). Its recomputed OOF (0.9490) exactly reproduces
the deployed identity-only OOF, confirming fold alignment between branches.

Fusion is a margin-gated simplex grid search (step 0.1, margin 0.003, equal-weight fallback)
over the five OOF probability matrices; the tuned weights are applied to the test probability
matrices and argmaxed.

| member | OOF | weight |
|--------|-----|--------|
| probe_gopt384 | 0.9509 | 0.6 |
| text_gopt384 | 0.8693 | 0.1 |
| probe_so400m512 | 0.9472 | 0.1 |
| text_so400m512 | 0.8981 | 0.0 |
| finetuned_siglip2 | 0.9490 | 0.2 |

Equal-weight blend OOF is 0.9500; the tuned blend reaches OOF 0.9676. Deployed scores:
OOF 0.9676 / public LB 0.96363 (prior best 0.93636; +2.73 points).

Data flow: HF image/text embeddings -> probes + text zero-shot (branch A); bundle reload and
identity eval (branch B); weighted blend of the five probability matrices -> argmax. The class
names came from manual visual inspection of 1-2 training images per class (checked into
`src/data/class_names.csv`), which also identified the dataset as quarters of Food-101,
Oxford Flowers-102, Stanford Cars, and FGVC-Aircraft.

Run it with `python -m src.final_blend` (prerequisite: `outputs/single_ft/single_ft_bundle.pkl`).
It writes `outputs/submission_blend.csv` (copied to `outputs/submission.csv`), the probe fold
models / weights / class names in `outputs/final_blend/blend_bundle.pkl`, metrics and metadata
under `outputs/final_blend/`, and cached embeddings at `outputs/cache/hf_gopt384.npz` +
`hf_so400m512.npz` (keyed by a prompt hash; reruns skip the HF models). No test images are used
for training anything (the spec restricts training to the train directory; in particular we did
not use test-set pseudo-labeling).

## Branch B: the fine-tuned SigLIP-2

![branch B: test image to prediction](diagrams/deploy.svg)

This is branch B of the deployed blend; it also remains trainable standalone via
`python -m src.single_ft`. The backbone is SigLIP-2 SO400M
(`vit_so400m_patch14_siglip_gap_378.v2_webli`) at 378 px. LoRA
(r=8, alpha=16, dropout=0.05) adapts the attention projections (`attn.qkv` + `attn.proj`) of the
last 4 transformer blocks; the pretrained weights stay frozen. A trainable head sits on the
average-pooled token embedding (SigLIP-2 is a gap model with no CLS token, so it mean-pools).

Data flow per fold:

- image -> LoRA backbone (`forward_features`) -> average-pooled feature (1152-d) -> head ->
  softmax probabilities.

The head is an L2-normalized cosine head with a learnable scale, trained with a class-balanced
cross-entropy loss (plus label smoothing 0.1). This ports the frozen probe's cosine + balanced
recipe into the fine-tuned head; on the imbalanced 4-41/class data it contributed the bulk of
the gain over a plain linear head.

Training and evaluation:

- A single shared `StratifiedKFold(4)` per seed (`lora_members.shared_folds`); each fold trains
  LoRA + head with AdamW (lr 1e-4 LoRA / 1e-3 head), cosine schedule + 10% warmup, bf16, up to
  30 epochs, early stop on validation accuracy (patience 5). The best-epoch validation softmax
  forms that fold's OOF rows.
- Reported over seeds {42, 43, 44} as a 3-seed OOF mean +/- std. Served as the seed-42 4-fold
  softmax ensemble over the test set.
- Evaluation is identity-only. hflip TTA was measured and cost about 1 image (consistent with
  the earlier multi-view TTA finding), so it is off.

Standalone scores: OOF 0.9484 +/- 0.0004 (3-seed) / public LB 0.93636. This matches the
fine-tuned 3-backbone ensemble's LB (0.93636) while beating its OOF (0.9404) and training only
12 fine-tunes instead of 36.

Run it with `python -m src.single_ft`; it writes `outputs/submission_siglip2.csv`, the bundle at
`outputs/single_ft/single_ft_bundle.pkl`, and metrics/metadata under `outputs/single_ft/`.

## How we got here (deprecated pipelines)

Preserved under `src/deprecated/`:

| approach | entry point | OOF / LB |
|----------|-------------|----------|
| Single-backbone frozen probe | `src.deprecated.train` (aug_views 1) + `src.deprecated.predict` | 0.8851 / ~0.88 |
| Multi-view K=8 frozen probe | `src.deprecated.train` (default) + `src.deprecated.predict` | 0.9129 / 0.90000 |
| Frozen cross-backbone ensemble | `src.deprecated.ensemble` | 0.9314 / 0.91818 |
| Fine-tuned LoRA ensemble (3 backbones) | `src.deprecated.ensemble_ft` | 0.9404 / 0.93636 |
| Single fine-tuned SigLIP-2 (branch B; was the deploy until tonight) | `src.single_ft` | 0.9484 / 0.93636 |

The frozen probe L2-normalizes features and uses a `class_weight="balanced"` logistic regression,
cross-validated with `RepeatedStratifiedKFold` to pick C. Multi-view augments features (8 seeded
views per image) under grouped CV so no augmented view of a held-out image leaks into its training
fold. The ensembles blend per-backbone OOF/test probability matrices (late fusion on the 100-d
probabilities) with gated simplex-search weights. The `ensemble_ft` path generalized the LoRA code
to all three backbones; this work then showed that one fine-tuned backbone sufficed.

## Key design decisions and non-obvious details

- LoRA over full fine-tuning. ~1000 images would overfit a fully fine-tuned 400M-parameter
  backbone; LoRA on the last 4 blocks plus a small head keeps the trainable parameter count tiny.
- The deployed head is a cosine classifier. Pooled features are L2-normalized and matched against
  L2-normalized class weights with a learnable scale; the loss is class-balanced to offset the
  4-41 per-class imbalance. The default `head_type` is still a plain linear head, so the cosine
  head is opt-in per member.
- Levers are config-gated and default off. `head_type`, `class_balanced`, and `tta_views` live in
  a member's `lora` config override; with none set, the deprecated `ensemble_ft` and LoRA paths
  are byte-for-byte unchanged.
- Pooling depends on the backbone. SigLIP-2 and AIMv2 are gap models with no CLS token
  (`num_prefix_tokens=0`), so they mean-pool tokens; DINOv3 has a CLS token and uses CLS +
  mean-patch. Applying cls+meanpatch to a gap model would silently grab a patch token as the CLS.
- Fold alignment is shared and free. `shared_folds` builds one `StratifiedKFold` per seed that
  every member and the identity-OOF attribution reuse, so OOF rows stay aligned and comparisons
  (single vs ensemble member) are honest.
- TTA is not free. hflip averaging cost about 1 image on OOF, matching the earlier 512/multi-view
  finding, so the deploy is identity-only. The comparison is recorded in `metrics.json`
  (`tta_delta`).
- Trust OOF over the public LB. The ~110-image public slice cannot separate configs within ~1 OOF
  point: single SigLIP-2 and the 3-backbone ensemble tie at LB 0.93636 even though the 1079-image
  OOF separates them by +0.80. Decisions follow the multi-seed OOF.
- Outputs are non-destructive. `final_blend` writes `submission_blend.csv`; the deployed
  `submission.csv` is a manual copy of it. `single_ft` (branch B standalone) writes
  `submission_siglip2.csv`. Earlier submissions and model dirs are archived under
  `outputs/deprecated/` (nothing is deleted).
- Sinkhorn balancing is approximate and off by default. The test set is 1036 images (~10.36 per
  class), not the spec's idealized 1000, so the "exactly 10 per class" premise only approximately
  holds; `inference.sinkhorn` defaults to false.
- Frozen probes on bigger checkpoints beat fine-tuning the smaller one. The gopt-384 probe
  (0.9509) outscored the entire fine-tuned model (0.9484); backbone scale dominated.
- Text zero-shot members are weak alone but blend-positive. Individually 0.87-0.90, yet the
  gopt-384 text member earned weight 0.1 in the blend.
- The margin-gated weight search guards against useless members. The so400m-512 text member got
  weight 0.0 (the margin gate, 0.003, rejected it for not clearing the equal-weight fallback).
- Prompt-hash npz caching. Branch A's HF embeddings are cached at `outputs/cache/*.npz` keyed by
  a prompt hash, so reruns skip the HF models entirely (~13 min first run -> ~3 min cached).
- transformers 5.x returns a model-output object. `get_image_features` / `get_text_features`
  are unwrapped via `pooler_output` in `hf_siglip2.py`.
- No test-image training. The spec restricts training to the train directory; in particular we
  did not use test-set pseudo-labeling. Class names came from viewing train images only.

## Reproducibility

`set_seed` seeds `random`, NumPy, and PyTorch, sets cuDNN deterministic, and calls
`torch.use_deterministic_algorithms(True, warn_only=True)`. The scikit-learn probe paths are
deterministic. Branch B's LoRA path uses memory-efficient attention, which is
non-deterministic, so per-seed OOF drifts roughly 0.2-0.3 point (about 2-3 images per fold) and
can drift more across torch/CUDA/GPU versions - this is why branch B reports a 3-seed OOF mean
+/- std rather than a single number. `collect_metadata` records the git SHA and exact library
versions in `outputs/single_ft/metadata.json`.

The blend adds no new seed variance beyond branch B: branch A's probes (sklearn lbfgs on frozen
embeddings) and text members (frozen embeddings, deterministic softmax) are deterministic, and
branch B is reused unchanged from its saved bundle.
