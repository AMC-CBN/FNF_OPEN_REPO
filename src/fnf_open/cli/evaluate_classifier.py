"""CLI for running inference with trained multi-view classifiers."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Iterable, List, MutableMapping, Optional, Sequence, Tuple

import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from ..classification.detection_crops import build_detection_crops
from ..data.classification_dataset import MultiViewDataset, prepare_classification_dataset
from ..data.loading import load_pickle_dataset
from ..models.multiview import LegacyMultiBranchClassifier, MultiBranchClassifier
from ..transforms.classification import create_transforms


LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _resolve_checkpoint_paths(
    directory: Path | None,
    explicit: Sequence[Path] | None,
) -> List[Path]:
    paths: List[Path] = []
    if directory is not None:
        directory = directory.expanduser().resolve()
        if not directory.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {directory}")
        candidates = []
        for pattern in ("*.pt", "*.pth"):
            candidates.extend(directory.glob(pattern))
        paths.extend(sorted(candidates))
    if explicit:
        paths.extend(Path(path).expanduser().resolve() for path in explicit)

    unique: List[Path] = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file does not exist: {path}")
        seen.add(path)
        unique.append(path)
    return unique


def _strip_module_prefix(state: MutableMapping[str, torch.Tensor]) -> MutableMapping[str, torch.Tensor]:
    if any(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return state


def _remap_legacy_branches(state: MutableMapping[str, torch.Tensor]) -> MutableMapping[str, torch.Tensor]:
    branch_map = {
        "input1": "branches.0",
        "input2": "branches.1",
        "input3": "branches.2",
    }

    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        for legacy, target in branch_map.items():
            prefix = f"{legacy}."
            if new_key.startswith(prefix):
                new_key = f"{target}{new_key[len(legacy):]}"
                break
        remapped[new_key] = value
    return remapped


def _merge_branch_state(state: MutableMapping[str, torch.Tensor]) -> MutableMapping[str, torch.Tensor]:
    if any(key.startswith("feature_extractor.") for key in state):
        return state

    has_branch = any(key.startswith("branches.") for key in state)
    if not has_branch:
        return state

    merged: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("branches.0."):
            suffix = key[len("branches.0.") :]
            if suffix.startswith("classifier"):
                continue  # drop per-branch classifiers
            merged[f"feature_extractor.{suffix}"] = value
        elif key.startswith("branches."):
            if ".classifier" in key:
                continue
            # ignore other branches (assumes branch0 representative)
            continue
        else:
            merged[key] = value
    return merged


def _remap_classifier_keys(state: MutableMapping[str, torch.Tensor]) -> MutableMapping[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        if key.startswith("classifier.fc."):
            new_key = "classifier.0." + key[len("classifier.fc.") :]
        elif key == "classifier.fc.weight":
            new_key = "classifier.0.weight"
        elif key == "classifier.fc.bias":
            new_key = "classifier.0.bias"
        elif key == "classifier.weight":
            new_key = "classifier.0.weight"
        elif key == "classifier.bias":
            new_key = "classifier.0.bias"
        remapped[new_key] = value
    return remapped


def _normalise_state_dict(
    state: MutableMapping[str, torch.Tensor], legacy: bool
) -> MutableMapping[str, torch.Tensor]:
    state = _strip_module_prefix(state)
    if any(key.startswith("input") for key in state):
        state = _remap_legacy_branches(state)
    if not legacy:
        state = _merge_branch_state(state)
        state = _remap_classifier_keys(state)
    return state


def _extract_include_lat(metadata: object) -> Optional[bool]:
    if not isinstance(metadata, dict):
        return None
    config = metadata.get("config")
    if isinstance(config, dict):
        value = config.get("include_lat")
        if isinstance(value, bool):
            return value
    direct = metadata.get("include_lat")
    if isinstance(direct, bool):
        return direct
    return None


def _unpack_checkpoint(
    checkpoint_path: Path,
) -> Tuple[MutableMapping[str, torch.Tensor], dict[str, object]]:
    state = torch.load(checkpoint_path, map_location="cpu")
    model_state: MutableMapping[str, torch.Tensor]
    metadata: dict[str, object]

    if isinstance(state, dict):
        if "model_state" in state:
            model_state = state["model_state"]  # type: ignore[assignment]
            metadata = state.get("metadata", {})  # type: ignore[assignment]
        elif "model" in state:
            model_state = state["model"]  # type: ignore[assignment]
            metadata = state.get("metadata", {})  # type: ignore[assignment]
        elif "state_dict" in state:
            model_state = state["state_dict"]  # type: ignore[assignment]
            metadata = state.get("metadata", {})  # type: ignore[assignment]
        else:
            if all(isinstance(k, str) for k in state.keys()):
                model_state = state  # type: ignore[assignment]
                metadata = {}
            else:
                raise KeyError(
                    f"Checkpoint {checkpoint_path} does not contain a recognised state dict"
                )
    else:
        model_state = state  # type: ignore[assignment]
        metadata = {}

    if not isinstance(metadata, dict):
        metadata = {}
    return model_state, metadata


def _infer_num_outputs(state: MutableMapping[str, torch.Tensor], default: int = 1) -> int:
    for key, value in state.items():
        if key.endswith("classifier.0.weight") and value.ndim == 2:
            return int(value.shape[0])
        if key.endswith("classifier.weight") and value.ndim == 2:
            return int(value.shape[0])
    return default


def _logits_to_positive_prob(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return torch.sigmoid(logits)
    if logits.size(1) == 1:
        return torch.sigmoid(logits.squeeze(1))
    probs = torch.softmax(logits, dim=1)
    return probs[:, 1]


def _load_logits(
    checkpoint_path: Path,
    backbone: str,
    loader: DataLoader,
    device: torch.device,
    *,
    num_views: int,
) -> Tuple[torch.Tensor, dict[str, object], int]:
    LOGGER.info("Evaluating checkpoint %s", checkpoint_path.name)
    model_state, metadata = _unpack_checkpoint(checkpoint_path)

    has_feature_extractor = any(key.startswith("feature_extractor") for key in model_state)
    is_legacy = not has_feature_extractor or any(
        key.startswith("input") or key.startswith("branches.") for key in model_state
    )
    if is_legacy and num_views != 3:
        raise ValueError(
            f"Legacy checkpoint {checkpoint_path} only supports AP+LAT evaluation (num_views=3)."
        )
    model_state = _normalise_state_dict(model_state, legacy=is_legacy)

    metadata_config = metadata.get("config") if isinstance(metadata, dict) else None
    config_num_classes: Optional[int] = None
    if isinstance(metadata_config, dict):
        maybe_value = metadata_config.get("num_classes")
        if isinstance(maybe_value, int):
            config_num_classes = maybe_value
    config_include_lat = _extract_include_lat(metadata)

    if config_include_lat is not None:
        expected_from_metadata = 3 if config_include_lat else 2
        if expected_from_metadata != num_views:
            raise ValueError(
                f"Checkpoint {checkpoint_path} expects {'AP+LAT' if config_include_lat else 'AP only'} views, "
                f"but evaluation was configured with {num_views}."
            )

    inferred_outputs = _infer_num_outputs(model_state, default=config_num_classes or 1)

    if is_legacy:
        model = LegacyMultiBranchClassifier(
            backbone=backbone,
            num_classes=inferred_outputs,
            pretrained=False,
        )
    else:
        model = MultiBranchClassifier(
            backbone=backbone,
            num_classes=inferred_outputs,
            pretrained=False,
            num_views=num_views,
        )

    load_result = model.load_state_dict(model_state, strict=False)
    if load_result.missing_keys:
        LOGGER.warning(
            "Checkpoint %s missing parameters: %s",
            checkpoint_path.name,
            ", ".join(sorted(load_result.missing_keys)),
        )
    if load_result.unexpected_keys:
        LOGGER.warning(
            "Checkpoint %s has unexpected parameters: %s",
            checkpoint_path.name,
            ", ".join(sorted(load_result.unexpected_keys)),
        )
    model.to(device)
    model.eval()

    logits_list: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            ap_right = batch["ap_right"].to(device)
            ap_left = batch["ap_left"].to(device)
            lat_tensor = batch.get("lat")
            lat = lat_tensor.to(device) if isinstance(lat_tensor, torch.Tensor) else None
            logits = model(ap_right, ap_left, lat)
            logits_list.append(logits.cpu())

    if not logits_list:
        raise ValueError("No samples were processed; check dataset and loader configuration")

    logits_tensor = torch.cat(logits_list, dim=0)
    return logits_tensor, metadata, inferred_outputs


def _metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[dict[str, object], torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_probs = _logits_to_positive_prob(logits)
    predictions = (pos_probs >= 0.5).long()

    accuracy = float((predictions == targets).float().mean().item())
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets.numpy(),
        predictions.numpy(),
        average="binary",
        zero_division=0,
    )
    conf_mat = confusion_matrix(targets.numpy(), predictions.numpy(), labels=[0, 1])

    metrics = {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": conf_mat.astype(int).tolist(),
    }
    confidence = torch.where(predictions == 1, pos_probs, 1 - pos_probs)
    return metrics, predictions, confidence, pos_probs


def _hard_vote(
    predictions: Iterable[torch.Tensor],
) -> torch.Tensor:
    votes = torch.stack(list(predictions))
    majority, _ = torch.mode(votes, dim=0)
    return majority


def _hard_vote_confidence(
    predictions: Iterable[torch.Tensor],
) -> torch.Tensor:
    votes = torch.stack(list(predictions))
    num_classes = int(votes.max().item()) + 1 if votes.numel() else 2
    vote_counts = torch.zeros(votes.size(1), num_classes, dtype=torch.float32)
    for vote in votes:
        indices = torch.arange(vote.numel(), dtype=torch.long)
        vote_counts[indices, vote] += 1
    confidence = vote_counts.max(dim=1).values / votes.size(0)
    return confidence


def _write_predictions(
    path: Path,
    serials: Sequence[str],
    labels: Sequence[int],
    predictions: Sequence[int],
    confidence: Sequence[float],
) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["serial", "label", "prediction", "confidence"])
        for serial, label, pred, conf in zip(serials, labels, predictions, confidence):
            writer.writerow([serial, int(label), int(pred), f"{float(conf):.6f}"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to the classification pickle dataset")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory containing per-fold checkpoints saved by fnf-train-classifier",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        help="Explicit checkpoint files to evaluate (overrides directory listing)",
    )
    parser.add_argument("--backbone", default="efficientnet_b0", help="Backbone name used during training")
    parser.add_argument(
        "--task",
        choices=("g12_vs_g34", "g3_vs_g4"),
        default="g12_vs_g34",
        help="Classification task the checkpoints were trained for",
    )
    parser.add_argument(
        "--views",
        choices=("auto", "ap", "ap_lat"),
        default="auto",
        help="Radiographic views to evaluate with (auto-detect, AP only, or AP+LAT)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run inference on (cuda[:index] or cpu); falls back to CPU if CUDA unavailable",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader workers")
    parser.add_argument(
        "--ensemble",
        choices=("mean", "hard", "none"),
        default="mean",
        help="Ensemble strategy when multiple checkpoints are provided",
    )
    parser.add_argument(
        "--secondary-checkpoint-dir",
        type=Path,
        help="Directory containing checkpoints for the secondary (g3_vs_g4) classifier",
    )
    parser.add_argument(
        "--secondary-checkpoints",
        type=Path,
        nargs="+",
        help="Explicit secondary-stage checkpoint files (overrides --secondary-checkpoint-dir)",
    )
    parser.add_argument(
        "--secondary-task",
        choices=("g3_vs_g4",),
        default="g3_vs_g4",
        help="Classification task used for the secondary stage",
    )
    parser.add_argument(
        "--save-predictions",
        type=Path,
        help="Optional CSV file to store per-sample predictions and confidence",
    )
    parser.add_argument(
        "--save-metrics",
        type=Path,
        help="Optional JSON file to store aggregated evaluation metrics (defaults to checkpoint dir)",
    )
    parser.add_argument(
        "--ap-detection-pickle",
        type=Path,
        help="Path to FNF_AP_Detection_data.pkl for detector-derived crops",
    )
    parser.add_argument(
        "--lat-detection-pickle",
        type=Path,
        help="Path to FNF_LAT_Detection_data.pkl for detector-derived crops",
    )
    parser.add_argument(
        "--ap-detection-checkpoint",
        type=Path,
        help="Checkpoint for the AP Faster R-CNN detector",
    )
    parser.add_argument(
        "--lat-detection-checkpoint",
        type=Path,
        help="Checkpoint for the LAT Faster R-CNN detector",
    )
    parser.add_argument(
        "--ap-detection-checkpoint-template",
        help="String template for AP detector checkpoints (use {fold})",
    )
    parser.add_argument(
        "--lat-detection-checkpoint-template",
        help="String template for LAT detector checkpoints (use {fold})",
    )
    parser.add_argument(
        "--detection-score-thr",
        type=float,
        default=0.3,
        help="Minimum detector score when selecting hip-joint boxes",
    )
    parser.add_argument(
        "--detection-device",
        help="Device string for detector inference (defaults to --device)",
    )
    parser.add_argument(
        "--use-ground-truth-crops",
        action="store_true",
        help="Reuse stored ground-truth crops instead of detector outputs",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve() if args.checkpoint_dir else None
    checkpoint_paths = _resolve_checkpoint_paths(checkpoint_dir, args.checkpoints)
    if not checkpoint_paths:
        parser.error("No checkpoints were provided. Use --checkpoint-dir or --checkpoints.")

    if args.save_predictions and args.ensemble == "none" and len(checkpoint_paths) > 1:
        parser.error("--save-predictions requires an ensemble method when multiple checkpoints are used")

    if args.views == "ap":
        include_lat = False
    elif args.views == "ap_lat":
        include_lat = True
    else:
        inferred_flags: List[bool] = []
        for path in checkpoint_paths:
            try:
                state_dict, metadata = _unpack_checkpoint(path)
            except Exception as exc:
                LOGGER.warning(
                    "Failed to inspect checkpoint %s for view inference: %s",
                    path,
                    exc,
                )
                continue
            flag = _extract_include_lat(metadata)
            if flag is not None:
                inferred_flags.append(flag)
            del state_dict
        if inferred_flags:
            unique_flags = set(inferred_flags)
            if len(unique_flags) > 1:
                parser.error(
                    "Checkpoints disagree on stored view configuration; specify --views explicitly."
                )
            include_lat = unique_flags.pop()
        else:
            LOGGER.warning(
                "Could not infer view configuration from checkpoints; defaulting to AP+LAT evaluation."
            )
            include_lat = True

    num_views = 3 if include_lat else 2
    LOGGER.info("Evaluating with %s views", "AP+LAT" if include_lat else "AP only")

    raw_dataset = load_pickle_dataset(args.dataset)

    def _resolve_device_string(device_str: str | None, purpose: str) -> str:
        if not device_str:
            return "cpu"
        resolved = device_str
        lowered = device_str.lower()
        if lowered.startswith("cuda"):
            if not torch.cuda.is_available():
                LOGGER.warning(
                    "CUDA requested for %s via device string '%s' but is unavailable; falling back to CPU",
                    purpose,
                    device_str,
                )
                resolved = "cpu"
            else:
                _, _, index_str = device_str.partition(":")
                if index_str:
                    try:
                        index = int(index_str)
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid CUDA device '{device_str}' for {purpose}; expected format cuda[:index]"
                        ) from exc
                    device_count = torch.cuda.device_count()
                    if index < 0 or index >= device_count:
                        raise ValueError(
                            f"Invalid CUDA device '{device_str}' for {purpose}. Available indices: 0..{device_count - 1}"
                        )
        return resolved

    classification_device_str = _resolve_device_string(str(args.device), "classification")
    detection_device_input = str(args.detection_device) if args.detection_device else classification_device_str
    detection_device_str = _resolve_device_string(detection_device_input, "detection")

    detected_crops = None
    if not args.use_ground_truth_crops:
        if args.ap_detection_pickle is None:
            parser.error(
                "Detector crops requested but AP detection pickle is missing. "
                "Provide --ap-detection-pickle or use --use-ground-truth-crops."
            )
        if include_lat and args.lat_detection_pickle is None:
            parser.error(
                "LAT detection pickle required for AP+LAT evaluation. "
                "Provide --lat-detection-pickle or select --views ap."
            )
        if args.ap_detection_checkpoint is None and args.ap_detection_checkpoint_template is None:
            parser.error(
                "Provide either --ap-detection-checkpoint or --ap-detection-checkpoint-template."
            )
        if include_lat and args.lat_detection_checkpoint is None and args.lat_detection_checkpoint_template is None:
            parser.error(
                "Provide either --lat-detection-checkpoint or --lat-detection-checkpoint-template."
            )

        detected_crops = build_detection_crops(
            ap_detection_pickle=args.ap_detection_pickle,
            lat_detection_pickle=args.lat_detection_pickle if include_lat else None,
            ap_checkpoint=args.ap_detection_checkpoint,
            lat_checkpoint=args.lat_detection_checkpoint,
            ap_checkpoint_template=args.ap_detection_checkpoint_template,
            lat_checkpoint_template=args.lat_detection_checkpoint_template,
            score_thr=args.detection_score_thr,
            device=detection_device_str,
            include_lat=include_lat,
        )

    prepared = prepare_classification_dataset(
        raw_dataset,
        task=args.task,
        detected_crops=detected_crops,
        include_lat=include_lat,
    )
    _, eval_transform = create_transforms(
        args.task,
        augment=False,
        include_lat=include_lat,
    )
    dataset = MultiViewDataset(prepared, eval_transform)

    device = torch.device(classification_device_str)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    targets = torch.tensor(dataset.labels, dtype=torch.long)

    logits_per_model: List[torch.Tensor] = []
    metadata_per_model: List[dict[str, object]] = []
    positive_probabilities: List[torch.Tensor] = []
    for path in checkpoint_paths:
        logits, metadata, _ = _load_logits(
            path,
            args.backbone,
            loader,
            device,
            num_views=num_views,
        )
        logits_per_model.append(logits)
        metadata_per_model.append(metadata)
        positive_probabilities.append(_logits_to_positive_prob(logits))

    results: dict[str, object] = {
        "device": str(device),
        "views": "ap_lat" if include_lat else "ap",
        "models": [],
    }
    for path, logits, metadata in zip(checkpoint_paths, logits_per_model, metadata_per_model):
        metrics, predictions, confidence, _ = _metrics_from_logits(logits, targets)
        model_entry = {
            "checkpoint": str(path),
            "metrics": metrics,
            "metadata": metadata,
        }
        results["models"].append(model_entry)
        LOGGER.info(
            "Checkpoint %s -> accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
            path.name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )

    ensemble_summary: dict[str, object] | None = None
    ensemble_predictions: torch.Tensor | None = None
    ensemble_confidence: torch.Tensor | None = None
    if len(logits_per_model) == 1 or args.ensemble != "none":
        if len(logits_per_model) == 1:
            ensemble_source = "single"
            combined_logits = logits_per_model[0]
            metrics, predictions, confidence, _ = _metrics_from_logits(combined_logits, targets)
            ensemble_predictions = predictions
            ensemble_confidence = confidence
        elif args.ensemble == "mean":
            ensemble_source = "mean"
            stacked = torch.stack(logits_per_model)
            combined_logits = stacked.mean(dim=0)
            metrics, predictions, confidence, _ = _metrics_from_logits(combined_logits, targets)
            ensemble_predictions = predictions
            ensemble_confidence = confidence
        else:  # hard voting
            ensemble_source = "hard"
            per_model_predictions = [(probs >= 0.5).long() for probs in positive_probabilities]
            ensemble_predictions = _hard_vote(per_model_predictions)
            ensemble_confidence = _hard_vote_confidence(per_model_predictions)
            accuracy = float((ensemble_predictions == targets).float().mean().item())
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets.numpy(),
                ensemble_predictions.numpy(),
                average="binary",
                zero_division=0,
            )
            conf_mat = confusion_matrix(targets.numpy(), ensemble_predictions.numpy(), labels=[0, 1])
            metrics = {
                "accuracy": accuracy,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "confusion_matrix": conf_mat.astype(int).tolist(),
            }

        ensemble_summary = {"strategy": ensemble_source, "metrics": metrics}
        LOGGER.info(
            "Ensemble (%s) -> accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
            ensemble_source,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )

        if args.save_predictions is not None:
            serials = prepared.serials
            _write_predictions(
                args.save_predictions,
                serials,
                prepared.labels,
                ensemble_predictions.numpy(),
                ensemble_confidence.numpy(),
            )
            LOGGER.info("Saved predictions to %s", args.save_predictions)

    if ensemble_summary is not None:
        results["ensemble"] = ensemble_summary

    secondary_dir = (
        args.secondary_checkpoint_dir.expanduser().resolve()
        if args.secondary_checkpoint_dir
        else None
    )
    secondary_paths = _resolve_checkpoint_paths(secondary_dir, args.secondary_checkpoints)
    if secondary_paths:
        if args.task != "g12_vs_g34":
            parser.error("Secondary checkpoints require the primary task to be g12_vs_g34")

        try:
            secondary_prepared = prepare_classification_dataset(
                raw_dataset,
                task=args.secondary_task,
                detected_crops=detected_crops,
                include_lat=include_lat,
            )
        except ValueError as exc:  # pragma: no cover - user input validation
            parser.error(f"Secondary dataset preparation failed: {exc}")
        _, secondary_eval_transform = create_transforms(
            args.secondary_task,
            augment=False,
            include_lat=include_lat,
        )
        secondary_dataset = MultiViewDataset(secondary_prepared, secondary_eval_transform)
        secondary_loader = DataLoader(
            secondary_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        secondary_targets = torch.tensor(secondary_dataset.labels, dtype=torch.long)

        sec_logits_per_model: List[torch.Tensor] = []
        sec_metadata_per_model: List[dict[str, object]] = []
        sec_positive_probabilities: List[torch.Tensor] = []
        for path in secondary_paths:
            logits, metadata, _ = _load_logits(
                path,
                args.backbone,
                secondary_loader,
                device,
                num_views=num_views,
            )
            sec_logits_per_model.append(logits)
            sec_metadata_per_model.append(metadata)
            sec_positive_probabilities.append(_logits_to_positive_prob(logits))

        secondary_results: dict[str, object] = {"device": str(device), "models": []}
        for path, logits, metadata in zip(secondary_paths, sec_logits_per_model, sec_metadata_per_model):
            metrics, predictions, confidence, _ = _metrics_from_logits(logits, secondary_targets)
            secondary_results["models"].append(
                {
                    "checkpoint": str(path),
                    "metrics": metrics,
                    "metadata": metadata,
                }
            )
            LOGGER.info(
                "Secondary %s -> accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
                path.name,
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
            )

        secondary_ensemble_summary: dict[str, object] | None = None
        if len(sec_logits_per_model) == 1 or args.ensemble != "none":
            if len(sec_logits_per_model) == 1:
                sec_source = "single"
                combined_logits = sec_logits_per_model[0]
                metrics, predictions, _, _ = _metrics_from_logits(combined_logits, secondary_targets)
            elif args.ensemble == "mean":
                sec_source = "mean"
                stacked = torch.stack(sec_logits_per_model)
                combined_logits = stacked.mean(dim=0)
                metrics, predictions, _, _ = _metrics_from_logits(combined_logits, secondary_targets)
            else:
                sec_source = "hard"
                per_model_predictions = [(probs >= 0.5).long() for probs in sec_positive_probabilities]
                predictions = _hard_vote(per_model_predictions)
                accuracy = float((predictions == secondary_targets).float().mean().item())
                precision, recall, f1, _ = precision_recall_fscore_support(
                    secondary_targets.numpy(),
                    predictions.numpy(),
                    average="binary",
                    zero_division=0,
                )
                conf_mat = confusion_matrix(secondary_targets.numpy(), predictions.numpy(), labels=[0, 1])
                metrics = {
                    "accuracy": accuracy,
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "confusion_matrix": conf_mat.astype(int).tolist(),
                }

            secondary_ensemble_summary = {"strategy": sec_source, "metrics": metrics}
            LOGGER.info(
                "Secondary ensemble (%s) -> accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
                sec_source,
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
            )

            secondary_results["ensemble"] = secondary_ensemble_summary
        results["secondary"] = secondary_results

    metrics_path = args.save_metrics
    if metrics_path is None:
        if checkpoint_dir is not None:
            metrics_path = checkpoint_dir / "evaluation_metrics.json"
        else:
            first_model_dir = checkpoint_paths[0].parent if checkpoint_paths else Path.cwd()
            metrics_path = first_model_dir / "evaluation_metrics.json"

    if metrics_path is not None:
        metrics_path = metrics_path.expanduser().resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(results, indent=2))
        LOGGER.info("Saved metrics to %s", metrics_path)

    print(json.dumps(results, indent=2))
    return 0


__all__ = ["build_parser", "main"]
