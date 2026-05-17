"""Stage 6 — Data augmentation on top of Stage 5's fine-tuned backbone.

Adds on-the-fly horizontal flip (50%) + modest color jitter to the training
set only. Val sees no augmentation. Otherwise identical to Stage 5.

See stages/stage_006_plan.md for design rationale.
"""
import time
import random
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Reuse from earlier stages
from stage3 import (
    collate_fn, build_targets, yolo_loss, evaluate_map,
    ANCHORS_PX, NUM_CLASSES, NUM_ANCHORS, GRID_SIZE, INPUT_SIZE,
    ENEMY_CLASSES, TRAIN_MAPS, VAL_MAPS, DATA_DIR,
    SEED, VAL_EVAL_SIZE,
)
from stage4 import IMAGENET_MEAN, IMAGENET_STD
from stage5 import FineTunedYOLO, train_one_epoch_finetune

# Stage-6 config
WEIGHTS_OUT = Path("stage6_best.pt")
BATCH_SIZE = 16
EPOCHS = 40                # augmentation slows convergence; need more epochs
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4


class FrameDetectionDatasetAugmented(Dataset):
    """Dataset with optional training-time augmentation.

    Augmentations (only when augment=True):
      - Horizontal flip at 50% probability (with bbox cx flip).
      - Modest color jitter (brightness/contrast/saturation; no hue change).

    Val/test always pass augment=False — they see raw normalized images.
    """
    def __init__(self, root: Path, allowed_maps: set, input_size: int = INPUT_SIZE,
                 augment: bool = False):
        self.items = []
        for map_name in sorted(allowed_maps):
            map_dir = root / map_name
            if not map_dir.is_dir():
                continue
            images = sorted(list(map_dir.glob("*.png")) + list(map_dir.glob("*.jpg")))
            for img_path in images:
                txt = img_path.with_suffix(".txt")
                if not txt.exists():
                    continue
                boxes = []
                with open(txt) as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) != 5:
                            continue
                        cls = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                        if 0 < w < 1 and 0 < h < 1 and 0 < cx < 1 and 0 < cy < 1:
                            boxes.append((cls, cx, cy, w, h))
                if boxes:
                    self.items.append((img_path, boxes))
        self.input_size = input_size
        self.augment = augment
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        # Color-jitter ranges intentionally modest. Hue is *not* perturbed
        # because Doom's class identities are partly palette-coded.
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.15)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, boxes = self.items[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        tensor = torch.from_numpy(img)

        if self.augment:
            # Horizontal flip with bbox transform
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=[2])
                boxes = [(cls, 1.0 - cx, cy, w, h) for cls, cx, cy, w, h in boxes]
            # Color jitter (photometric — bboxes unchanged)
            tensor = self.color_jitter(tensor)

        return (tensor - self.mean) / self.std, boxes


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    anchors = ANCHORS_PX.to(device)
    train_ds = FrameDetectionDatasetAugmented(DATA_DIR, TRAIN_MAPS, augment=True)
    val_ds   = FrameDetectionDatasetAugmented(DATA_DIR, VAL_MAPS,   augment=False)
    print(f"Train frames: {len(train_ds):,} (with augmentation)")
    print(f"Val frames:   {len(val_ds):,} (no augmentation)")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda"))

    model = FineTunedYOLO().to(device)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_total:,}")

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
