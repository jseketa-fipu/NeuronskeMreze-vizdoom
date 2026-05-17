"""Stage 5 — Pretrained ResNet18 backbone unfrozen + fine-tuned.

The fix for Stage 4's frozen-BN failure: let the backbone train (including
BatchNorm running statistics adapting to Doom's pixel distribution) while
preserving pretrained features via a 10× lower backbone LR vs the head.

See stages/stage_005_plan.md for design rationale.
"""
import time
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

# Reuse everything from Stages 3/4
from stage3 import (
    collate_fn, build_targets, yolo_loss, evaluate_map,
    ANCHORS_PX, NUM_CLASSES, NUM_ANCHORS, GRID_SIZE, INPUT_SIZE,
    ENEMY_CLASSES, TRAIN_MAPS, VAL_MAPS, DATA_DIR,
    SEED, VAL_EVAL_SIZE,
)
from stage4 import FrameDetectionDatasetImageNet, IMAGENET_MEAN, IMAGENET_STD  # noqa: F401

# Stage-5-specific config
WEIGHTS_OUT = Path("stage5_best.pt")
BATCH_SIZE = 16
EPOCHS = 30
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4   # 10× lower — preserves pretrained features


class FineTunedYOLO(nn.Module):
    """ResNet18 + 1×1 head; both trainable, both go into train() during training
    (so BatchNorm running stats adapt to Doom)."""
    def __init__(self, num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.out_per_anchor = 5 + num_classes
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.head = nn.Conv2d(512, num_anchors * self.out_per_anchor, 1)
        # Note: NO freezing, NO eval() forcing, NO no_grad in forward.

    def forward(self, x):
        return self.head(self.backbone(x))


def train_one_epoch_finetune(model, loader, optimizer, anchors_px, device):
    """Standard training loop — both backbone and head receive gradients."""
    model.train()
    total, n = 0.0, 0
    components = {"box": 0.0, "obj": 0.0, "noobj": 0.0, "cls": 0.0}
    for images, gt_boxes_batch in loader:
        images = images.to(device, non_blocking=True)
        predictions = model(images)
        bt, ot, ct, pm, nm = build_targets(
            gt_boxes_batch, anchors_px, GRID_SIZE, INPUT_SIZE, NUM_CLASSES, device)
        loss, comp = yolo_loss(predictions, bt, ot, ct, pm, nm, NUM_ANCHORS, NUM_CLASSES)
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
    if device.type == "cpu":
        print("  WARNING: CPU is impractical; use Colab T4.")

    anchors = ANCHORS_PX.to(device)
    train_ds = FrameDetectionDatasetImageNet(DATA_DIR, TRAIN_MAPS)
    val_ds   = FrameDetectionDatasetImageNet(DATA_DIR, VAL_MAPS)
    print(f"Train frames: {len(train_ds):,}")
    print(f"Val frames:   {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda"))

    model = FineTunedYOLO().to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {n_total:,}")
    print(f"Trainable parameters: {n_trainable:,}   (full model)")

    # Discriminative LR: backbone gets 10× smaller LR than head.
    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
        {"params": model.head.parameters(),     "lr": LR_HEAD},
    ])
    print(f"Backbone LR: {LR_BACKBONE}, Head LR: {LR_HEAD}")

    best_map = 0.0
    print("\nEpoch  total  box   obj   noobj cls    val_mAP   time")
    print("-" * 60)
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        avg_loss, comp = train_one_epoch_finetune(model, train_loader, optimizer, anchors, device)
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

    # Final per-class breakdown with best weights, larger val sample.
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
