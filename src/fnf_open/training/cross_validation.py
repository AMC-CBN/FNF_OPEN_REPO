"""Cross-validation utilities for three-view fracture classifiers."""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import albumentations as A
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from ..data.classification_dataset import (
    MultiViewDataset,
    PreparedClassificationDataset,
    prepare_classification_dataset,
)
from ..models.multiview import MultiBranchClassifier
from ..transforms.classification import create_transforms
from .config import FoldReport, TrainingConfig


LOGGER = logging.getLogger(__name__)


def _build_loader(
    data: PreparedClassificationDataset,
    indices: Sequence[int],
    transform: A.BasicTransform,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = MultiViewDataset(data=data, indices=indices, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _split_train_val(
    indices: Sequence[int],
    val_fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    indices_arr = np.array(indices, dtype=int)
    rng.shuffle(indices_arr)
    val_size = max(1, int(len(indices_arr) * val_fraction))
    train_idx = indices_arr[val_size:]
    val_idx = indices_arr[:val_size]
    if len(train_idx) == 0:  # pragma: no cover - guard for tiny datasets
        raise ValueError("Training split is empty; reduce val_fraction or provide more samples")
    return train_idx, val_idx


def _compute_pos_weight(labels: Sequence[int]) -> Optional[torch.Tensor]:
    positives = int(np.sum(labels))
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return None
    ratio = negatives / positives
    return torch.tensor([ratio], dtype=torch.float32)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        ap_l = batch["ap_left"].to(device)
        ap_r = batch["ap_right"].to(device)
        lat = batch["lat"].to(device)
        y = batch["label"].to(device).float().unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        logits = model(ap_r, ap_l, lat)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in loader:
        ap_l = batch["ap_left"].to(device)
        ap_r = batch["ap_right"].to(device)
        lat = batch["lat"].to(device)
        y = batch["label"].to(device).float().unsqueeze(1)
        logits = model(ap_r, ap_l, lat)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long().squeeze(1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.squeeze(1).cpu().numpy())
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets).astype(int)
    accuracy = float((preds == targets).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, average="binary", zero_division=0
    )
    conf_mat = confusion_matrix(targets, preds, labels=[0, 1])
    return total_loss / len(loader.dataset), accuracy, float(precision), float(recall), float(f1), conf_mat


def train_model_cross_validation(
    raw_dataset: MutableMapping[str, Sequence[np.ndarray]],
    cfg: TrainingConfig,
    num_folds: int = 5,
    device: str | torch.device = "cpu",
    checkpoint_dir: Optional[Path] = None,
) -> List[FoldReport]:
    prepared = prepare_classification_dataset(raw_dataset, task=cfg.task)
    train_transform, eval_transform = create_transforms(cfg.task, augment=True)
    labels = prepared.labels

    fold_mapping = prepared.fold_mapping
    if fold_mapping:
        fold_names = sorted(fold_mapping.keys())
        if len(fold_names) < 3:
            raise ValueError("At least three folds are required to build 3:1:1 splits")
        num_folds = len(fold_names)
    else:
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=cfg.seed)
    reports: List[FoldReport] = []
    device_t = torch.device(device)

    checkpoint_dir_path: Optional[Path]
    if checkpoint_dir is None:
        checkpoint_dir_path = None
    else:
        checkpoint_dir_path = Path(checkpoint_dir).expanduser().resolve()
        checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Starting %d-fold cross-validation for task %s with %d samples",
        num_folds,
        cfg.task,
        len(labels),
    )

    if fold_mapping:
        # Rotate folds so each serves as the test split once; subsequent fold is validation
        fold_names = sorted(fold_mapping.keys())
        fold_count = len(fold_names)
        fold_sequence = list(range(fold_count))
        splits = []
        for idx in fold_sequence:
            test_name = fold_names[idx]
            val_name = fold_names[(idx + 1) % fold_count]
            train_names = [fold_names[i] for i in range(fold_count) if fold_names[i] not in {test_name, val_name}]
            train_indices = np.concatenate([fold_mapping[name] for name in train_names]).astype(int)
            val_indices = np.asarray(fold_mapping[val_name], dtype=int)
            test_indices = np.asarray(fold_mapping[test_name], dtype=int)
            splits.append((train_indices, val_indices, test_indices))
    else:
        splits = []
        for fold_index, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            rng = np.random.default_rng(cfg.seed + fold_index)
            train_idx, val_idx = _split_train_val(trainval_idx, val_fraction=0.2, rng=rng)
            splits.append((train_idx, val_idx, test_idx))

    for fold_index, (train_idx, val_idx, test_idx) in enumerate(splits):
        train_idx = np.asarray(train_idx, dtype=int)
        val_idx = np.asarray(val_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

        train_loader = _build_loader(
            prepared, train_idx, train_transform, cfg.batch_size, shuffle=True
        )
        val_loader = _build_loader(
            prepared, val_idx, eval_transform, cfg.batch_size, shuffle=False
        )
        test_loader = _build_loader(
            prepared, test_idx, eval_transform, cfg.batch_size, shuffle=False
        )

        fold_num = fold_index + 1
        LOGGER.info(
            "Fold %d/%d: train=%d, val=%d, test=%d, epochs=%d",
            fold_num,
            num_folds,
            len(train_idx),
            len(val_idx),
            len(test_idx),
            cfg.epochs,
        )

        model = MultiBranchClassifier(
            backbone=cfg.model_name,
            num_classes=cfg.num_classes,
            pretrained=cfg.use_pretrained,
        )
        model.to(device_t)
        pos_weight_value = _compute_pos_weight(prepared.labels[train_idx])
        if pos_weight_value is not None:
            pos_weight_tensor = pos_weight_value.to(device_t)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )

        best_val_acc = -1.0
        best_epoch = -1
        val_loss = float("nan")
        best_model_state: Optional[OrderedDict[str, torch.Tensor]] = None
        checkpoint_metadata: Optional[Dict[str, object]] = None
        fold_checkpoint_path: Optional[Path] = None

        if checkpoint_dir_path is not None:
            fold_checkpoint_path = checkpoint_dir_path / f"fold{fold_num}_best.pt"

        for epoch in range(cfg.epochs):
            _ = _train_one_epoch(model, train_loader, criterion, optimizer, device_t)
            val_loss, val_acc, *_ = _evaluate(model, val_loader, criterion, device_t)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_model_state = OrderedDict(
                    (name, tensor.detach().cpu()) for name, tensor in model.state_dict().items()
                )
                checkpoint_metadata = {
                    "fold": fold_num,
                    "config": asdict(cfg),
                    "task": cfg.task,
                    "best_epoch": epoch + 1,
                    "best_val_accuracy": float(val_acc),
                    "best_val_loss": float(val_loss),
                }
                if fold_checkpoint_path is not None:
                    torch.save(
                        {
                            "model_state": best_model_state,
                            "metadata": checkpoint_metadata,
                        },
                        fold_checkpoint_path,
                    )

            LOGGER.info(
                "Fold %d/%d - Epoch %d/%d: val_loss=%.4f val_acc=%.4f best_epoch=%d",
                fold_num,
                num_folds,
                epoch + 1,
                cfg.epochs,
                val_loss,
                val_acc,
                best_epoch + 1,
            )

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        test_loss, test_acc, precision, recall, f1, conf_mat = _evaluate(
            model, test_loader, criterion, device_t
        )

        if fold_checkpoint_path is not None and checkpoint_metadata is not None and best_model_state is not None:
            checkpoint_metadata.update(
                {
                    "test_loss": float(test_loss),
                    "test_accuracy": float(test_acc),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "confusion_matrix": conf_mat.astype(int).tolist(),
                }
            )
            torch.save(
                {
                    "model_state": best_model_state,
                    "metadata": checkpoint_metadata,
                },
                fold_checkpoint_path,
            )
        LOGGER.info(
            "Fold %d/%d finished: best_epoch=%d val_acc=%.4f test_acc=%.4f",
            fold_num,
            num_folds,
            best_epoch + 1,
            best_val_acc,
            test_acc,
        )
        reports.append(
            FoldReport(
                fold_index=fold_index,
                best_epoch=best_epoch,
                val_loss=val_loss,
                val_accuracy=best_val_acc,
                test_loss=test_loss,
                test_accuracy=test_acc,
                precision=precision,
                recall=recall,
                f1=f1,
                confusion_matrix=conf_mat,
            )
        )
    return reports


def aggregate_reports(reports: Iterable[FoldReport]) -> Dict[str, object]:
    """Compute aggregate metrics across folds."""

    reports_list = list(reports)
    if not reports_list:
        raise ValueError("No reports provided for aggregation")

    mean_accuracy = float(np.mean([r.test_accuracy for r in reports_list]))
    mean_precision = float(np.mean([r.precision for r in reports_list]))
    mean_recall = float(np.mean([r.recall for r in reports_list]))
    mean_f1 = float(np.mean([r.f1 for r in reports_list]))
    conf_matrix = np.add.reduce([r.confusion_matrix for r in reports_list])

    return {
        "accuracy": mean_accuracy,
        "precision": mean_precision,
        "recall": mean_recall,
        "f1": mean_f1,
        "confusion_matrix": conf_matrix.astype(int).tolist(),
    }
