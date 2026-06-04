# src layout

## Deployed model

`single_ft.py` is the deployed model: a single LoRA-fine-tuned SigLIP-2 SO400M
(@378, avg pool) with an L2-normalized cosine head and class-balanced loss,
identity-only eval. Run it from the repo root:

```
.venv\Scripts\python.exe -m src.single_ft
```

It writes the deployed submission to `outputs/submission_siglip2.csv` (copied to
`outputs/submission.csv` for the Kaggle upload) plus the model bundle and metrics
to `outputs/single_ft/`. Deployed scores: OOF 0.9484 (3-seed shared-fold mean) /
public LB 0.93636. Full story (including why hflip TTA was dropped) in
`docs/flow/specs/2026-06-03-single-siglip2-design.md` and `docs/flow/experiments.md`
(phase E).

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
journey (see `docs/flow/experiments.md` phases A-D). These are NOT the deployed
model:

- `deprecated/train.py` + `deprecated/predict.py` - frozen probe and multi-view
  K=8 (the earlier deploy, LB 0.900).
- `deprecated/ensemble.py` - frozen cross-backbone ensemble (OOF 0.9314 / LB 0.91818).
- `deprecated/ensemble_ft.py` - fine-tuned 3-backbone LoRA ensemble (OOF 0.9404 /
  LB 0.93636); `single_ft` matches its LB and beats its OOF at 1/3 the cost.
- `deprecated/multiview.py`, `deprecated/fusion.py` - feature augmentation and
  probability-blend helpers used by the pipelines above.
