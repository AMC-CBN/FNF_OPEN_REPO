# FNF Open

**FNF Open** is a lightweight, modular Python library for preprocessing hip radiographs and
classifying femoral neck fractures (FNF) from AP and lateral views. It extracts the reusable
components from the original research notebooks and exposes them as a clear, tested API
together with simple command‑line tools.

> **What it does**
>
> - Preprocess DICOM/X‑ray images into square, normalized model inputs
> - Build multi‑view classification datasets (AP‑left, AP‑right, LAT)
> - Train and evaluate timm‑based backbones (e.g., EfficientNet, ResNet, RexNet) with 5‑fold stratified CV
> - Produce compact JSON reports with accuracy, precision, recall, F1, and confusion matrices

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

<details>
<summary>Optional: install with extras</summary>

```bash
pip install -e ".[dev]"
```
</details>

## Command‑line tools

After installation, the following console commands are available:

- `fnf-prepare-internal` – preprocess the internal dataset into detection/ classification pickles used by the paper
- `fnf-prepare-external` – preprocess an external dataset into pickles for evaluation
- `fnf-train-classifier` – train a multi-view classifier with cross-validation (supports `--task g12_vs_g34` or `--task g3_vs_g4`)
- `fnf-evaluate-classifier` – run inference with saved checkpoints (single model or ensemble) and export predictions
- `fnf-train-detection` – train Faster R-CNN hip detectors on AP/LAT crops (supports five-fold loops)
- `fnf-evaluate-detection` – report detection metrics for saved checkpoints (single fold or `{fold}` templates)
- `fnf-run-external-pipeline` – run the default external preprocessing workflow on a datasets workspace (see `datasets/README.md`)

Show full help for any command with `-h`.

## Pipeline overview

1. **Detection Stage (Hip-Joint ROI)** – Train a Faster R-CNN detector on the AP/LAT pickles to localise hip joints. Ground-truth boxes follow the study specification: draw a circle centred on the femoral-neck midpoint with radius to the femoral-head extremity, add a 50 px margin, and take the circumscribing square. AP views provide two ROIs (affected + contralateral sides), LAT views provide one, yielding three ROIs per patient. Evaluate detectors with precision, recall, IoU, and, when available, mAP at IoU > 0.5.
2. **Classification Stage (Garden Type)** – Run the classifier exclusively on the detected ROIs (never the full-frame X-ray). Each crop is resized to the backbone input (e.g., 256×256) with padding to preserve aspect, min/max normalised, and passed through an ImageNet-initialised backbone. The three branches share weights, their embeddings fuse in the classification head, and training uses a weighted `BCEWithLogitsLoss` with fold-specific class balancing. Tasks are staged: Garden I+II vs III+IV, then Garden III vs IV for displaced predictions. Optional ensembles can apply hard voting across top-K models. Supply `--views ap` to the CLI when you want an AP-only workflow; all LAT-specific arguments become optional in that mode.
3. **Labels and Ground Truth** – Use the 3D-CT derived Garden label per patient (select the more severe side if labels differ). Two expert readers assign labels, resolving disagreements by consensus. Detection annotations supervise ROI localisation only and are not class labels.
4. **Classification Evaluation Protocol** – Always evaluate on ROIs produced by the detection stage. Report accuracy, precision, recall, Dice/F1, and AUC where relevant. For confidence analysis, stratify metrics by probability thresholds (e.g., ≥95%).
5. **Cross-Validation & Splits** – Follow 5-fold patient-level splits stratified by Garden type with a 3:1:1 train/val/test ratio per fold (permuted across folds).
6. **Training Setup** – Detection trains up to 100 epochs (batch size 8), classification up to 200 epochs (batch size 32). Use AdamW (β₁=0.9, β₂=0.999) with initial LR 1e-3, and keep augmentations minimal (horizontal flip for classification).
7. **Implementation Notes** – The classification code must consume detector-produced ROIs. Keep detection and classification modules decoupled but pipelined, ensuring benchmarks mirror the detection→classification workflow. Maintain scripts to train/evaluate detection (reporting IoU/mAP/precision/recall), train/evaluate classification from detected crops, and optionally evaluate using saved detection checkpoints without retraining.

### Quick start: reproduce the paper workflow

1. Prepare the internal dataset (labels, raw DICOMs, XML annotations). Replace the paths below with your own data location—by default we expect the workspace described in `datasets/README.md`:

   ```bash
   fnf-prepare-internal datasets/internal/label.xlsx \
     datasets/internal/raw \
     datasets/internal/box_annotation \
     datasets/internal/box_annotation \
     datasets/internal/preprocessed
   ```

   The command writes four pickle files––`FNF_Paper_data.pkl`, `FNF_AP_Detection_data.pkl`, `FNF_LAT_Detection_data.pkl`, and `FNF_Classification_data.pkl`––mirroring the preprocessing stages described in the manuscript. Detection pickles feed the Faster R-CNN training scripts under `example/`, while `FNF_Classification_data.pkl` powers the multi-view classifier.

2. Train the hip-joint detectors (per-fold Faster R-CNN). Use the detection CLI to mirror the study splits:

   ```bash
   fnf-train-detection datasets/internal/preprocessed/FNF_AP_Detection_data.pkl \
     --view AP \
     --fold all \
     --epochs 100 \
     --batch-size 8 \
     --out models/detection

   fnf-train-detection datasets/internal/preprocessed/FNF_LAT_Detection_data.pkl \
     --view LAT \
     --fold all \
     --epochs 100 \
     --batch-size 8 \
     --out models/detection
   ```

   Each run prints precision, recall, mean IoU, and mAP per fold, averages them, and drops checkpoints such as `models/detection/detection_ap_fold{k}_best.pth`. Use `--view AP`/`LAT` and adjust hyperparameters as required.

3. Evaluate existing detection checkpoints (optional). When weights already exist—either from the previous step or provided artifacts—benchmark them against the prepared pickles without retraining:

   ```bash
   fnf-evaluate-detection datasets/internal/preprocessed/FNF_AP_Detection_data.pkl \
     --view AP \
     --fold all \
     --checkpoint-template models/detection/detection_ap_fold{fold}_best.pth
   ```

   The command loads each fold’s checkpoint, prints precision/recall/IoU/mAP, and reports their mean; training-specific flags such as `--epochs` are ignored automatically.

4. Train the cross-validated classifier (metrics plus per-fold checkpoints are stored under `models/`). Ensure detector crops are available before this step:

   ```bash
   fnf-train-classifier datasets/internal/preprocessed/FNF_Classification_data.pkl \
     --backbone efficientnet_b4 \
     --epochs 200 \
     --folds 5 \
     --ap-detection-pickle datasets/internal/preprocessed/FNF_AP_Detection_data.pkl \
     --lat-detection-pickle datasets/internal/preprocessed/FNF_LAT_Detection_data.pkl \
     --ap-detection-checkpoint-template "models/detection/detection_ap_fold{fold}_best.pth" \
     --lat-detection-checkpoint-template "models/detection/detection_lat_fold{fold}_best.pth" \
     --out models/classification
   ```

   Each fold writes a checkpoint to `models/classification/efficientnet_b4/g12_vs_g34/fold{n}_best.pt`, alongside `metrics.json`. Pass `--no-pretrained` only when the machine has no internet access. Use `--task g3_vs_g4` to fine-tune Garden III versus IV models (those checkpoints land under `models/classification/efficientnet_b4/g3_vs_g4/`).

   To run an AP-only experiment, add `--views ap` and omit the LAT-specific inputs:

   ```bash
   fnf-train-classifier datasets/internal/preprocessed/FNF_Classification_data.pkl \
     --views ap \
     --backbone efficientnet_b4 \
     --epochs 200 \
     --folds 5 \
     --ap-detection-pickle datasets/internal/preprocessed/FNF_AP_Detection_data.pkl \
     --ap-detection-checkpoint-template "models/detection/detection_ap_fold{fold}_best.pth" \
     --out models/classification
   ```

5. Evaluate saved folds (single model or ensemble) and export predictions:

   ```bash
   fnf-evaluate-classifier datasets/internal/preprocessed/FNF_Classification_data.pkl \
     --checkpoint-dir models/classification/efficientnet_b4/g12_vs_g34 \
     --backbone efficientnet_b4 \
     --task g12_vs_g34 \
     --ensemble mean \
     --ap-detection-pickle datasets/internal/preprocessed/FNF_AP_Detection_data.pkl \
     --lat-detection-pickle datasets/internal/preprocessed/FNF_LAT_Detection_data.pkl \
     --ap-detection-checkpoint-template "models/detection/detection_ap_fold{fold}_best.pth" \
     --lat-detection-checkpoint-template "models/detection/detection_lat_fold{fold}_best.pth" \
     --secondary-checkpoint-dir models/classification/efficientnet_b4/g3_vs_g4 \
     --save-predictions models/classification/efficientnet_b4/predictions.csv
   ```

   The command first evaluates the Garden I+II vs III+IV classifier, then (because of `--secondary-checkpoint-dir`) automatically runs the Garden III vs IV stage on displaced predictions. Per-model metrics, optional ensemble performance (hard or mean voting), and per-patient confidence scores are printed; metrics JSON is saved next to the checkpoints (override with `--save-metrics`). The CLI defaults to the first CUDA device; pass `--device cpu` to force CPU inference or supply a GPU index. Use `--views ap` and drop the LAT arguments when you only need an AP-only evaluation, or keep the default for AP+LAT metrics. Drop `--secondary-checkpoint-dir` when you only need the binary Garden I+II vs III+IV evaluation.

6. (Optional) Prepare an external evaluation set with `fnf-run-external-pipeline` or `fnf-prepare-external`, then repeat the detection→classification evaluation using the saved checkpoints to mirror study conditions.

## Project layout

```
.
├── CHANGELOG.md                    # Release history
├── CITATION.cff                    # Citation metadata for the project
├── datasets/                       # Sample/internal & external datasets (workspace)
├── docs/                           # Additional documentation (model card, etc.)
├── examples/                       # Research scripts for detection/classification studies
├── models/                         # Pretrained + newly trained checkpoints (default CLI output)
├── src/fnf_open/                   # Library source (CLI, data, models, training, ...)
├── scripts/                        # Legacy wrapper scripts kept for backwards compat
├── pyproject.toml                  # Package metadata and entry points
├── requirements.txt                # Optional runtime dependency mirror
└── LICENSE
```

## Python API

```python
from pathlib import Path
from fnf_open import (
    load_pickle_dataset,
    TrainingConfig,
    train_model_cross_validation,
    aggregate_reports,
)
from fnf_open.classification.detection_crops import build_detection_crops

dataset = load_pickle_dataset(Path("datasets/internal/preprocessed/FNF_Classification_data.pkl"))
detected_crops = build_detection_crops(
    ap_detection_pickle=Path("datasets/internal/preprocessed/FNF_AP_Detection_data.pkl"),
    lat_detection_pickle=Path("datasets/internal/preprocessed/FNF_LAT_Detection_data.pkl"),
    ap_checkpoint_template="models/detection/detection_ap_fold{fold}_best.pth",
    lat_checkpoint_template="models/detection/detection_lat_fold{fold}_best.pth",
    device="cuda",
)
config = TrainingConfig(
    epochs=200,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=1e-2,
    seed=42,
    model_name="efficientnet_b4",
    num_classes=1,
    task="g12_vs_g34",
    use_pretrained=True,
)
reports = train_model_cross_validation(
    dataset,
    config,
    num_folds=5,
    device="cuda",
    detected_crops=detected_crops,
)
print(aggregate_reports(reports))
```

Set `device="cpu"` if CUDA is unavailable. Only set `use_pretrained=False` when the environment cannot download ImageNet weights. Use `set_random_seed` when you need reproducible sampling outside of the provided training loop.

## Additional documentation

- `docs/MODEL_CARD.md` – model card describing intended use, metrics, and known limitations.
- `models/README.md` – explains how checkpoints are organised under `classification/` and `detection/`.
- `models/` – legacy research scripts/checkpoints retained for archival purposes (default CLI outputs also land here).
- `CHANGELOG.md` – highlights the differences between releases.
- `datasets/README.md` – expected folder layout for the raw/external data workspaces.

## Reproducibility notes

- The packaged code supports 5‑fold stratified cross‑validation (3:1:1 train/val/test).
- Backbones are provided via [`timm`](https://github.com/huggingface/pytorch-image-models) and can be selected with a flag.
- Metrics include accuracy, precision/recall/F1 and confusion matrix; ROC/AUC can be added from scikit‑learn.

## Citing

See [`CITATION.cff`](CITATION.cff) for the full citation entry.

## License

MIT. See [`LICENSE`](LICENSE).
