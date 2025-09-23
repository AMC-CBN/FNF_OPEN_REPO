"""Detection metrics utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

import torch

__all__ = ["evaluate_detection", "box_iou"]


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes (format: [xmin, ymin, xmax, ymax])."""

    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.size(0), boxes2.size(0)), dtype=torch.float32, device=boxes1.device)

    x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)


def _greedy_match(
    preds: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    *,
    iou_thr: float,
    score_thr: float,
) -> Tuple[int, int, int, float, List[Tuple[int, float, bool]]]:
    """Greedy 1-to-1 matching by IoU for a single image.

    Returns ``(tp, fp, fn, mean_iou, matches)`` where ``matches`` is a list of
    ``(label, score, is_tp)`` for each prediction kept after score filtering.
    """

    pboxes = preds["boxes"]
    plabels = preds["labels"]
    pscores = preds.get("scores", torch.ones_like(plabels, dtype=torch.float32))

    keep = pscores >= score_thr
    pboxes = pboxes[keep]
    plabels = plabels[keep]
    pscores = pscores[keep]

    gboxes = target["boxes"]
    glabels = target["labels"]

    matches: List[Tuple[int, float, bool]] = []
    if gboxes.numel() == 0:
        for label, score in zip(plabels.tolist(), pscores.tolist()):
            matches.append((int(label), float(score), False))
        return 0, int(plabels.size(0)), 0, 0.0, matches

    if pboxes.numel() == 0:
        return 0, 0, int(gboxes.size(0)), 0.0, matches

    tp = 0
    fp = 0
    fn = int(gboxes.size(0))
    iou_list: List[float] = []
    matched_gt = set()

    order = torch.argsort(pscores, descending=True)
    for idx in order.tolist():
        pbox = pboxes[idx].unsqueeze(0)
        plab = plabels[idx].item()
        cand = torch.where(glabels == plab)[0]
        if cand.numel() == 0:
            fp += 1
            matches.append((int(plab), float(pscores[idx].item()), False))
            continue

        ious = box_iou(pbox, gboxes[cand]).squeeze(0)
        best_idx = int(torch.argmax(ious).item())
        iou = float(ious[best_idx].item())
        gt_index = int(cand[best_idx].item())
        if gt_index in matched_gt or iou < iou_thr:
            fp += 1
            matches.append((int(plab), float(pscores[idx].item()), False))
            continue

        matched_gt.add(gt_index)
        tp += 1
        fn -= 1
        iou_list.append(iou)
        matches.append((int(plab), float(pscores[idx].item()), True))

    mean_iou = float(sum(iou_list) / len(iou_list)) if iou_list else 0.0
    return tp, fp, fn, mean_iou, matches


def evaluate_detection(
    preds: Sequence[Dict[str, torch.Tensor]],
    targets: Sequence[Dict[str, torch.Tensor]],
    *,
    iou_thr: float = 0.5,
    score_thr: float = 0.3,
) -> Dict[str, float]:
    """Compute precision, recall, mean IoU, and mAP for detection results."""

    assert len(preds) == len(targets)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    mean_ious: List[float] = []

    matches_by_class: DefaultDict[int, List[Tuple[float, bool]]] = defaultdict(list)
    gt_count: DefaultDict[int, int] = defaultdict(int)

    for pred, tgt in zip(preds, targets):
        tp, fp, fn, mean_iou, matches = _greedy_match(pred, tgt, iou_thr=iou_thr, score_thr=score_thr)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        mean_ious.append(mean_iou)

        for label in tgt["labels"].tolist():
            gt_count[int(label)] += 1
        for label, score, is_tp in matches:
            matches_by_class[label].append((score, is_tp))

    precision = float(total_tp / (total_tp + total_fp + 1e-7))
    recall = float(total_tp / (total_tp + total_fn + 1e-7))
    mean_iou = float(sum(mean_ious) / len(mean_ious)) if mean_ious else 0.0

    aps: List[float] = []
    for label, num_gt in gt_count.items():
        matches = matches_by_class.get(label, [])
        if num_gt == 0:
            continue
        if not matches:
            aps.append(0.0)
            continue

        sorted_matches = sorted(matches, key=lambda x: x[0], reverse=True)
        tps = 0
        fps = 0
        precisions: List[float] = []
        recalls: List[float] = []
        for _, is_tp in sorted_matches:
            if is_tp:
                tps += 1
            else:
                fps += 1
            precisions.append(tps / (tps + fps + 1e-7))
            recalls.append(tps / (num_gt + 1e-7))

        precisions_tensor = torch.tensor([1.0] + precisions + [0.0])
        recalls_tensor = torch.tensor([0.0] + recalls + [1.0])
        for idx in range(precisions_tensor.numel() - 2, -1, -1):
            precisions_tensor[idx] = torch.maximum(precisions_tensor[idx], precisions_tensor[idx + 1])
        ap = torch.sum((recalls_tensor[1:] - recalls_tensor[:-1]) * precisions_tensor[1:]).item()
        aps.append(ap)

    map_score = float(sum(aps) / len(aps)) if aps else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
        "mAP": map_score,
    }
