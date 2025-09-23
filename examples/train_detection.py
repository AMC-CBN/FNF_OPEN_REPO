\
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .utils import seed_everything, get_device, EarlyStopping, timestamp
from .metrics import evaluate_detection
from fnf_open.utils.paths import DEFAULT_DETECTION_MODEL_DIR, detection_checkpoint_path


@dataclass
class TrainConfig:
    num_classes: int = 3           # including background
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
    out_dir: str = str(DEFAULT_DETECTION_MODEL_DIR)
    name: str = "fasterrcnn_resnet50_ap"
    save_best_metric: str = "mean_iou"  # or "precision"
    patience: int = 8  # early stopping


def _make_loader(dataset, cfg: TrainConfig, *, shuffle: bool) -> DataLoader:
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
    val_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    *,
    desc: str,
) -> Dict[str, float]:
    all_preds: List[Dict[str, torch.Tensor]] = []
    all_tgts: List[Dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for _, images, targets in tqdm(val_loader, desc=desc, leave=False):
            images = [img.to(device) for img in images]
            preds = model(images)
            for p, t in zip(preds, targets):
                p_cpu = {k: v.detach().cpu() for k, v in p.items()}
                t_cpu = {k: v.detach().cpu() for k, v in t.items()}
                all_preds.append(p_cpu)
                all_tgts.append(t_cpu)

    metrics = evaluate_detection(all_preds, all_tgts, iou_thr=cfg.iou_thr, score_thr=cfg.score_thr)
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    val_ds,
    cfg: TrainConfig,
) -> Dict[str, float]:
    """Run inference-only evaluation for a detection model."""

    seed_everything(cfg.seed)
    device = get_device()
    model.to(device)
    val_loader = _make_loader(val_ds, cfg, shuffle=False)
    model.eval()
    metrics = _run_validation(model, val_loader, device, cfg, desc="Evaluation")
    return metrics


def run_training(
    model: torch.nn.Module,
    train_ds,
    val_ds,
    cfg: TrainConfig,
    save_checkpoint: bool = True,
) -> Dict[str, float]:
    """
    Generic training loop for torchvision detection models.
    """
    seed_everything(cfg.seed)
    device = get_device()
    model.to(device)
    run_id = f"{cfg.name}_{timestamp()}"
    ckpt_path = detection_checkpoint_path(
        f"{run_id}_best.pth", directory=cfg.out_dir, ensure_parent=True
    )

    train_loader = _make_loader(train_ds, cfg, shuffle=True)
    val_loader = _make_loader(val_ds, cfg, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler(enabled=cfg.amp)
    es = EarlyStopping(patience=cfg.patience, mode="max")

    best_metric = -1.0
    log = {"best_epoch": -1, "best_val": -1.0}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train]", leave=False)
        train_loss = 0.0
        for _, images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.amp):
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

            scaler.scale(loss).backward()
            if cfg.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item())
            pbar.set_postfix(loss=f"{float(loss.item()):.3f}")

        scheduler.step()

        # Validation (prediction mode)
        model.eval()
        metrics = _run_validation(
            model,
            val_loader,
            device,
            cfg,
            desc=f"Epoch {epoch}/{cfg.epochs} [val]",
        )
        mean_iou = metrics.get("mean_iou", 0.0)
        precision = metrics.get("precision", 0.0)

        # choose monitored
        monitored = mean_iou if cfg.save_best_metric == "mean_iou" else precision
        log.setdefault("history", []).append({
            "epoch": epoch,
            "train_loss": train_loss / max(len(train_loader), 1),
            "metrics": metrics,
        })
        log["last_metrics"] = metrics
        if monitored > best_metric:
            best_metric = monitored
            log["best_epoch"] = epoch
            log["best_val"] = best_metric
            log["best_metrics"] = metrics
            if save_checkpoint:
                torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, ckpt_path)

        # early stopping
        if es.step(monitored):
            break

    if "best_metrics" not in log:
        log["best_metrics"] = log.get("last_metrics", {})
    log["ckpt_path"] = str(ckpt_path) if ckpt_path.exists() else ""
    return log


def load_detection_checkpoint(
    model: torch.nn.Module,
    checkpoint_name: str,
    *,
    checkpoint_dir: Optional[str] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.nn.Module, Dict[str, object] | None]:
    """Load a saved detection checkpoint from the default directory."""

    ckpt_path = detection_checkpoint_path(checkpoint_name, directory=checkpoint_dir)
    state = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(state["model"])
    return model, state.get("cfg")
