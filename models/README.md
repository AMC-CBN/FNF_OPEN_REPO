# Model Artifacts

This folder stores trained checkpoints. The CLI defaults to writing outputs here
(`models/classification/` for multi-view classifiers, `models/detection/` for
Faster R-CNN models). Pretrained assets from the paper are also kept alongside
any new runs you produce locally.

- `classification/` – multi-view fracture classifiers (organised as `<backbone>/<task>/fold{n}_best.pt`).
- `detection/` – Faster R-CNN weights for locating the hip joint.

Cross-validated classification runs store:

- `metrics.json` – aggregate + per-fold statistics.
- `fold{n}_best.pt` – best validation checkpoint for fold `n` (loaded by `fnf-evaluate-classifier`).
