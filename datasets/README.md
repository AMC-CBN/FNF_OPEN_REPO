# Datasets workspace

The `datasets/` directory holds both the reference materials bundled with the
repository and any additional data you provide locally. Two subdirectories are
expected by the tooling:

- `datasets/internal/` – hospital cohort used to reproduce the paper.
- `datasets/external/` – optional cohort for external validation.

Feel free to mirror this structure elsewhere and point the CLI commands at your
preferred locations.

## Internal dataset (paper reproduction)

```
datasets/internal/
├── label.xlsx            # Excel sheet with patient IDs and Garden grades
├── raw/                  # Patient folders containing AP/LAT DICOM series
├── box_annotation/       # Bounding-box XML annotations (contains both AP/LAT files)
├── preprocessed/         # Pickles written by fnf-prepare-internal
```

Run the preprocessing pipeline (adjust the paths if you keep the data elsewhere):

```bash
fnf-prepare-internal datasets/internal/label.xlsx \
  datasets/internal/raw \
  datasets/internal/box_annotation \
  datasets/internal/box_annotation \
  datasets/internal/preprocessed
```

This produces four pickle files mirroring the stages described in the paper:

- `FNF_Paper_data.pkl` – metadata and original DICOM paths.
- `FNF_AP_Detection_data.pkl` / `FNF_LAT_Detection_data.pkl` – detection inputs and labels for Faster R-CNN.
- `FNF_Classification_data.pkl` – normalised 256×256 crops (left/right AP + LAT) with Garden labels.

Use the detection pickles with the Faster R-CNN helpers in `example/`, and feed
`FNF_Classification_data.pkl` to `fnf-train-classifier` for cross-validation.

## External dataset (evaluation only)

```
datasets/external/
├── metadata.xlsx         # Excel sheet with patient IDs and Garden grades
├── split/                # Patient-level folders with DICOM files
├── xml_ap/               # AP XML annotations (one per patient)
├── xml_lat/              # LAT XML annotations (one per patient)
└── processed/            # Generated pickle files will be written here
```

Run the helper to mirror the structure used in the original notebooks:

```bash
fnf-run-external-pipeline datasets/external
```

Three pickles will be written to `datasets/external/processed/`:

- `external_combined.pkl` – combined AP/LAT metadata.
- `external_ap.pkl` – preprocessed AP view tensors.
- `external_lat.pkl` – preprocessed LAT view tensors.

These pickles are intended for evaluation or inference. Do **not** train the
classifier on them; the learning pipeline relies on the crops derived from the
internal dataset.
