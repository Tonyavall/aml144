# Diagrams

`deploy.svg` - a high-level diagram of the deployed model (a single fine-tuned SigLIP-2):
test image -> fine-tuned backbone + cosine head -> 100-class prediction. Hand-authored SVG,
kept deliberately simple for the talk. The presentation deck that embeds it is kept outside
this repo at `cse144/docs/slides/`.

The earlier, denser d2 diagrams (the module dependency graph, the three frozen/ensemble
pipelines, and the probe internals) are archived under `docs/archive/diagrams/` as a historical
record of the journey. They document the now-deprecated pipelines under `src/deprecated/`. To
re-render those, see the d2 commands in `docs/archive/diagrams/README.md` (requires the d2 CLI).
