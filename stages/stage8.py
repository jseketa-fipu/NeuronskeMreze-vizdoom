"""Stage 8 — Focal loss for class + objectness imbalance.

Replaces Stage 6's BCE+CE losses with focal-loss equivalents to address class
imbalance. Architecture, optimizer, augmentation, and per-map split unchanged.

See stages/stage_008_plan.md for design rationale.
"""
import time
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Reuse from earlier stages
from stage3 import (
    collate_fn, build_targets, evaluate_map,
    ANCHORS_PX, NUM_CLASSES, NUM_ANCHORS, GRID_SIZE, INPUT_SIZE,
    ENEMY_CLASSES, TRAIN_MAPS, VAL_MAPS, DATA_DIR,
    SEED, LAMBDA_BOX, LAMBDA_OBJ, LAMBDA_NOOBJ, LAMBDA_CLS,
)
from stage5 import FineTunedYOLO
from stage6 import FrameDetectionDatasetAugmented

# Stage-8 config
WEIGHTS_OUT = Path("stage8_best.pt")
BATCH_SIZE = 16
EPOCHS = 40
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
FOCAL_GAMMA = 2.0
FOCAL_ALPHA_OBJ = 0.25       # for objectness loss only — down-weights easy negatives


# ---- Focal loss components ----

def focal_bce(logits, targets, alpha=FOCAL_ALPHA_OBJ, gamma=FOCAL_GAMMA):
    """Focal binary cross-entropy. Used for objectness."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * ce).sum()


def focal_ce(logits, targets, gamma=FOCAL_GAMMA):
    """Focal multi-class cross-entropy. Used for class loss."""
    log_p = F.log_softmax(logits, dim=-1)
    log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    return (-((1 - pt) ** gamma) * log_pt).sum()


def yolo_loss_focal(predictions, box_target, obj_target, cls_target,
                    pos_mask, noobj_mask, num_anchors, num_classes):
    """Same as Stage 3's yolo_loss but with focal loss replacing BCE/CE."""
    B = predictions.size(0)
    A, OPA = num_anchors, 5 + num_classes
    pred = predictions.view(B, A, OPA, GRID_SIZE, GRID_SIZE)
    pred_tx = pred[:, :, 0]
    pred_ty = pred[:, :, 1]
    pred_tw = pred[:, :, 2]
    pred_th = pred[:, :, 3]
    pred_obj = pred[:, :, 4]
    pred_cls = pred[:, :, 5:5 + num_classes]

    zero = torch.zeros((), device=predictions.device)

    n_pos = pos_mask.sum().item()
    if n_pos > 0:
        # Box loss (unchanged from Stage 3)
        box_loss = (
            F.mse_loss(torch.sigmoid(pred_tx[pos_mask]), box_target[:, :, 0][pos_mask], reduction="sum") +
            F.mse_loss(torch.sigmoid(pred_ty[pos_mask]), box_target[:, :, 1][pos_mask], reduction="sum") +
            F.smooth_l1_loss(pred_tw[pos_mask], box_target[:, :, 2][pos_mask], reduction="sum") +
            F.smooth_l1_loss(pred_th[pos_mask], box_target[:, :, 3][pos_mask], reduction="sum")
        )
        # FOCAL objectness loss on positives.
        obj_loss = focal_bce(pred_obj[pos_mask], obj_target[pos_mask])
        # FOCAL class loss.
        pred_cls_perm = pred_cls.permute(0, 1, 3, 4, 2)
        cls_loss = focal_ce(pred_cls_perm[pos_mask], cls_target[pos_mask])
    else:
        box_loss = zero; obj_loss = zero; cls_loss = zero

    if noobj_mask.sum() > 0:
        # FOCAL objectness loss on negatives (most cells).
        noobj_loss = focal_bce(pred_obj[noobj_mask], obj_target[noobj_mask])
    else:
        noobj_loss = zero

    total = (LAMBDA_BOX * box_loss + LAMBDA_OBJ * obj_loss
             + LAMBDA_NOOBJ * noobj_loss + LAMBDA_CLS * cls_loss) / B
    return total, {
        "box":   float(box_loss.detach().cpu()) / B,
        "obj":   float(obj_loss.detach().cpu()) / B,
        "noobj": float(noobj_loss.detach().cpu()) / B,
        "cls":   float(cls_loss.detach().cpu()) / B,
    }


def train_one_epoch_focal(model, loader, optimizer, anchors_px, device):
    """Stage 6's training loop but with the focal-loss variant."""
    model.train()
    total, n = 0.0, 0
    components = {"box": 0.0, "obj": 0.0, "noobj": 0.0, "cls": 0.0}
    for images, gt_boxes_batch in loader:
        images = images.to(device, non_blocking=True)
        predictions = model(images)
        bt, ot, ct, pm, nm = build_targets(
            gt_boxes_batch, anchors_px, GRID_SIZE, INPUT_SIZE, NUM_CLASSES, device)
        loss, comp = yolo_loss_focal(predictions, bt, ot, ct, pm, nm, NUM_ANCHORS, NUM_CLASSES)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        total += loss.item()
        for k in components:
            components[k] += comp[k]
        n += 1
    return total / n, {k: v / n for k, v in components.items()}


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    anchors = ANCHORS_PX.to(device)
    train_ds = FrameDetectionDatasetAugmented(DATA_DIR, TRAIN_MAPS, augment=True)
    val_ds   = FrameDetectionDatasetAugmented(DATA_DIR, VAL_MAPS,   augment=False)
    print(f"Train frames: {len(train_ds):,} (augmented)")
    print(f"Val frames:   {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda"))

    model = FineTunedYOLO().to(device)
    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
        {"params": model.head.parameters(),     "lr": LR_HEAD},
    ])
    print(f"Focal: gamma={FOCAL_GAMMA}, alpha_obj={FOCAL_ALPHA_OBJ}")
    print(f"Backbone LR: {LR_BACKBONE}, Head LR: {LR_HEAD}")

    best_map = 0.0
    print("\nEpoch  total  box   obj   noobj cls    val_mAP   time")
    print("-" * 60)
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        avg_loss, comp = train_one_epoch_focal(model, train_loader, optimizer, anchors, device)
        t_train = time.time() - t0
        t0 = time.time()
        val_map, _ = evaluate_map(model, val_ds, device, anchors)
        t_eval = time.time() - t0

        marker = ""
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), WEIGHTS_OUT)
            marker = "← best"
        print(f" {epoch:2d}/{EPOCHS}  {avg_loss:6.2f}  {comp['box']:5.2f} "
              f"{comp['obj']:.2f} {comp['noobj']:.2f} {comp['cls']:.2f}  "
              f"{val_map*100:6.2f}%   {t_train:.0f}+{t_eval:.0f}s  {marker}", flush=True)

    print(f"\nBest val mAP: {best_map*100:.2f}%")
    print(f"Best weights saved to: {WEIGHTS_OUT}")

    model.load_state_dict(torch.load(WEIGHTS_OUT, map_location=device))
    val_map, per_class = evaluate_map(model, val_ds, device, anchors,
                                      sample_size=min(len(val_ds), 2000))
    print(f"\nFull val mAP (best weights, 2k sample): {val_map*100:.2f}%")
    print("\nPer-class AP:")
    for cls in range(NUM_CLASSES):
        ap = per_class[cls]
        if ap is None:
            print(f"  {ENEMY_CLASSES[cls]:18s}    --    (no gt in sample)")
        else:
            print(f"  {ENEMY_CLASSES[cls]:18s}  {ap*100:5.2f}%")


if __name__ == "__main__":
    main()
