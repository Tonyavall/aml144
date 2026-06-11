# src layout

## Deployed model

`final_blend.py` is the deployed model: a two-branch blend. Branch A is a frozen
branch over two HuggingFace SigLIP-2 checkpoints (`google/siglip2-giant-opt-patch16-384`
and `google/siglip2-so400m-patch16-512`), each contributing a logistic-regression
probe on L2-normalized shared-space embeddings and a text zero-shot member
(class-name prompts through the text tower). Branch B is the LoRA fine-tuned
SigLIP-2 fold ensemble (`single_ft.py`), reloaded from its saved bundle with no
retraining. A margin-gated simplex grid search fuses the five OOF probability
matrices; the tuned weights are applied to the test matrices. Run it from the
repo root:

```
.venv\Scripts\python.exe -m src.final_blend
```

It requires `outputs/single_ft/single_ft_bundle.pkl` to exist (train it with
`python -m src.single_ft` or download from the Drive link in the README). It
writes the deployed submission to `outputs/submission_blend.csv` (copied to
`outputs/submission.csv` for the Kaggle upload) plus `outputs/final_blend/`
(metrics, metadata, and `blend_bundle.pkl` with the probe fold models, weights,
and class names) and `outputs/cache/hf_gopt384.npz` + `hf_so400m512.npz` (cached
embeddings keyed by a prompt hash; reruns skip the HF models). Deployed scores:
OOF 0.9676 / public LB 0.96363. Full story in `docs/architecture.md` and the
project report (`docs/CSE144_Final_Report.pdf`).

- `models/hf_siglip2.py` - HuggingFace SigLIP-2 feature extraction (image and text
  towers) for branch A; unwraps the transformers 5.x model-output object via
  `pooler_output`.
- `data/class_names.csv` - the 100 class names, derived by viewing 1-2 training
  images per class; required by the text members. Identifies the dataset as
  quarters of Food-101, Oxford Flowers-102, Stanford Cars, and FGVC-Aircraft.

## Branch B: the fine-tuned SigLIP-2

`single_ft.py` is branch B of the deployed blend and runs standalone too: a single
LoRA-fine-tuned SigLIP-2 SO400M (@378, avg pool) with an L2-normalized cosine head
and class-balanced loss, identity-only eval. Run it from the repo root:

```
.venv\Scripts\python.exe -m src.single_ft
```

It writes the branch-B submission to `outputs/submission_siglip2.csv` plus the
model bundle and metrics to `outputs/single_ft/`. Standalone scores: OOF 0.9484
(3-seed shared-fold mean) / public LB 0.93636. Full story (including why hflip TTA
was dropped) in `docs/architecture.md` and the project report.

## Shared core (dependencies of the deployed model)

- `models/lora_train.py` - LoRA fine-tuning primitives: the backbone+LoRA model,
  the cosine/linear head, class-balanced loss, TTA eval, and per-split training.
- `models/lora_members.py` - per-member fold helpers (shared CV folds, per-member
  OOF training, test softmax-ensembling); shared by `single_ft` and the deprecated
  `ensemble_ft`.
- `models/backbone.py`, `models/head.py`, `models/balance.py` - feature extraction,
  the frozen probe head, and Sinkhorn inference balancing.
- `data/`, `submission.py`, `utils.py` - image/label IO, submission writing, config
  and seeding.
- `report.py` - journey-report figures and tables (documentation tooling).

## deprecated/ (earlier approaches, kept for the report)

Prior pipelines, superseded by `single_ft` but preserved as documentation of the
journey (see `docs/architecture.md`). These are NOT the deployed model:

- `deprecated/train.py` + `deprecated/predict.py` - frozen probe and multi-view
  K=8 (the earlier deploy, LB 0.900).
- `deprecated/ensemble.py` - frozen cross-backbone ensemble (OOF 0.9314 / LB 0.91818).
- `deprecated/ensemble_ft.py` - fine-tuned 3-backbone LoRA ensemble (OOF 0.9404 /
  LB 0.93636); `single_ft` matches its LB and beats its OOF at 1/3 the cost.
- `deprecated/multiview.py`, `deprecated/fusion.py` - feature augmentation and
  probability-blend helpers used by the pipelines above.
