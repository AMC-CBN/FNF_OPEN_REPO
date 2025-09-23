"""Training routines for Faster R-CNN hip detection."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..utils.paths import detection_checkpoint_path
from .metrics import evaluate_detection
from .models import build_faster_rcnn
from .utils import EarlyStopping, get_device, seed_everything

__all__ = ["TrainConfig", "train_detection", "evaluate_detection_model", "load_detection_checkpoint"]


@dataclass
class TrainConfig:
    """Configuration for Faster R-CNN training/evaluation."""

    num_classes: int = 3  # including background
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 5e-4
    batch_size: int = 4
    num_workers: int = 2
    seed: int = 42
    score_thr: float = 0.3
    iou_thr: float = 0.5
    amp: bool = True
    grad_clip_norm: float = 2.0
    out_dir: str = "models/detection"
    name: str = "fasterrcnn_resnet50"
    save_best_metric: str = "mean_iou"  # or "precision"
    patience: int = 8


def _make_loader(dataset: Dataset, cfg: TrainConfig, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=getattr(dataset, "collate_fn", None) or (lambda x: tuple(zip(*x))),
        pin_memory=True,
    )


def _run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    *,
    desc: str,
) -> Dict[str, float]:
    model.eval()
    all_preds: List[Dict[str, torch.Tensor]] = []
    all_tgts: List[Dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for _, images, targets in tqdm(loader, desc=desc, leave=False):
            images = [img.to(device) for img in images]
            preds = model(images)
            for pred, tgt in zip(preds, targets):
                all_preds.append({k: v.detach().cpu() for k, v in pred.items()})
                all_tgts.append({k: v.detach().cpu() for k, v in tgt.items()})
    return evaluate_detection(all_preds, all_tgts, iou_thr=cfg.iou_thr, score_thr=cfg.score_thr)


def train_detection(
    train_dataset: Dataset,
    val_dataset: Dataset,
    cfg: TrainConfig,
    *,
    save_checkpoint: bool = True,
) -> Dict[str, object]:
    """Train a Faster R-CNN detector and return training statistics."""

    seed_everything(cfg.seed)
    device = get_device()
    model = build_faster_rcnn(cfg.num_classes)
    model.to(device)

    run_id = cfg.name
    ckpt_path = detection_checkpoint_path(f"{run_id}_best.pth", directory=cfg.out_dir, ensure_parent=True)

    train_loader = _make_loader(train_dataset, cfg, shuffle=True)
    val_loader = _make_loader(val_dataset, cfg, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler(enabled=cfg.amp)
    early_stopping = EarlyStopping(patience=cfg.patience, mode="max")

    best_metric = -1.0
    log: Dict[str, object] = {"best_epoch": -1, "best_val": -1.0, "history": []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train]", leave=False)
        for _, images, targets in progress:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.amp):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            if cfg.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            loss_scalar = float(loss.item())
            epoch_loss += loss_scalar
            progress.set_postfix(loss=f"{loss_scalar:.3f}")

        scheduler.step()
        metrics = _run_validation(
            model,
            val_loader,
            device,
            cfg,
            desc=f"Epoch {epoch}/{cfg.epochs} [val]",
        )

        monitored = metrics.get(cfg.save_best_metric, 0.0)
        log["history"].append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(len(train_loader), 1),
                "metrics": metrics,
            }
        )
        log["last_metrics"] = metrics

        if monitored > best_metric:
            best_metric = monitored
            log["best_epoch"] = epoch
            log["best_val"] = monitored
            log["best_metrics"] = metrics
            if save_checkpoint:
                torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, ckpt_path)

        if early_stopping.step(monitored):
            break

    if "best_metrics" not in log:
        log["best_metrics"] = log.get("last_metrics", {})
    log["ckpt_path"] = str(ckpt_path) if ckpt_path.exists() else ""
    return log


def evaluate_detection_model(
    model: torch.nn.Module,
    dataset: Dataset,
    cfg: TrainConfig,
) -> Dict[str, float]:
    """Evaluate a Faster R-CNN detector without training."""

    seed_everything(cfg.seed)
    device = get_device()
    model.to(device)
    loader = _make_loader(dataset, cfg, shuffle=False)
    return _run_validation(model, loader, device, cfg, desc="Evaluation")


def load_detection_checkpoint(
    model: torch.nn.Module,
    checkpoint_name: str,
    *,
    checkpoint_dir: Optional[str] = None,
    map_location: Optional[str | torch.device] = None,
) -> Tuple[torch.nn.Module, Dict[str, object] | None]:
    """Load model weights saved by :func:`train_detection`."""

    ckpt_path = detection_checkpoint_path(checkpoint_name, directory=checkpoint_dir)
    state = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(state["model"])
    return model, state.get("cfg")
