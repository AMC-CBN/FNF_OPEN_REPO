\
# A compact classification training loop (ResNet50) to support the classification notebooks.
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .utils import seed_everything, get_device, EarlyStopping, timestamp
import os
from fnf_open.utils.paths import (
    DEFAULT_CLASSIFICATION_MODEL_DIR,
    classification_checkpoint_path,
)


@dataclass
class ClsConfig:
    num_classes: int
    data_dir: str
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_workers: int = 2
    amp: bool = True
    patience: int = 5
    out_dir: str = str(DEFAULT_CLASSIFICATION_MODEL_DIR)
    name: str = "resnet50_cls"


def run_classification(cfg: ClsConfig) -> Dict[str, str]:
    seed_everything(42)
    device = get_device()
    run_id = f"{cfg.name}_{timestamp()}"
    ckpt_path = classification_checkpoint_path(
        f"{run_id}_best.pth", directory=cfg.out_dir, ensure_parent=True
    )

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(os.path.join(cfg.data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(cfg.data_dir, "val"), transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = models.resnet50(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler(enabled=cfg.amp)
    es = EarlyStopping(patience=cfg.patience, mode="max")

    best_acc = -1.0
    log = {"best_epoch": -1, "best_val": -1.0}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train]", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += float(loss.item())
        scheduler.step()

        # val
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.epochs} [val]", leave=False):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == y).sum().item())
                total += int(y.numel())
        acc = float(correct / max(1, total))

        if acc > best_acc:
            best_acc = acc
            log["best_epoch"] = epoch
            log["best_val"] = best_acc
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, ckpt_path)

        if es.step(acc):
            break

    log["ckpt_path"] = str(ckpt_path) if ckpt_path.exists() else ""
    return log


def load_classification_checkpoint(
    model: nn.Module,
    checkpoint_name: str,
    *,
    checkpoint_dir: Optional[str] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Tuple[nn.Module, Dict[str, object] | None]:
    """Load a saved classification checkpoint from the default directory."""

    ckpt_path = classification_checkpoint_path(checkpoint_name, directory=checkpoint_dir)
    state = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(state["model"])
    return model, state.get("cfg")
