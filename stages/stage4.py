"""Stage 4 — Pretrained backbone (frozen) + Stage 3 detection head.

See stages/stage_004_plan.md for design rationale.

Only the head trains (~34k params). Backbone is ResNet18 pretrained on ImageNet,
frozen and kept in eval() mode (so BatchNorm doesn't update its running stats).

Reuses every other component from stage3.py: target builder, loss, decoding,
NMS, mAP eval, training loop. Minimal diff vs. Stage 3 so any mAP improvement
is attributable to the backbone swap.
"""
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# Import everything reusable from Stage 3
from stage3 import (
    collate_fn, build_targets, yolo_loss,
    decode_predictions, predictions_to_detections,
    nms_per_class, compute_ap_per_class, evaluate_map,
    ANCHORS_PX, NUM_CLASSES, NUM_ANCHORS, GRID_SIZE, INPUT_SIZE,
    ENEMY_CLASSES, TRAIN_MAPS, VAL_MAPS, DATA_DIR,
    SEED, CONF_THRESH, NMS_IOU, AP_IOU, VAL_EVAL_SIZE,
)
import random
import torch.nn.functional as F  # noqa: F401  (transitive use through stage3 functions)

# Stage-4-specific config
WEIGHTS_OUT = Path("stage4_best.pt")
BATCH_SIZE = 16
EPOCHS = 30                                # head converges faster than from-scratch
LR = 1e-3                                  # head only — can be aggressive

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class FrameDetectionDatasetImageNet(Dataset):
    """Same as Stage 3's FrameDetectionDataset but with ImageNet normalization
    (required for pretrained models)."""
    def __init__(self, root: Path, allowed_maps: set, input_size: int = INPUT_SIZE):
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
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

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
        return (tensor - self.mean) / self.std, boxes


class PretrainedYOLO(nn.Module):
    """ResNet18 (pretrained on ImageNet, frozen) + 1×1 detection head."""
    def __init__(self, num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.out_per_anchor = 5 + num_classes
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Strip avgpool + fc; keep through layer4 → (B, 512, 13, 13) for 416 input.
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # Freeze backbone weights and keep it in eval mode (no BN-running-stat updates).
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.head = nn.Conv2d(512, num_anchors * self.out_per_anchor, 1)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.head(features)

    def train(self, mode=True):
        """Override: head goes into train mode, backbone stays in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self


def train_one_epoch_head_only(model, loader, optimizer, anchors_px, device):
    """Same as stage3's train_one_epoch but the no_grad backbone makes it slightly faster."""
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
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=10.0)
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
        print("  WARNING: CPU is impractical for training. Use Colab T4.")

    anchors = ANCHORS_PX.to(device)
    train_ds = FrameDetectionDatasetImageNet(DATA_DIR, TRAIN_MAPS)
    val_ds   = FrameDetectionDatasetImageNet(DATA_DIR, VAL_MAPS)
    print(f"Train frames: {len(train_ds):,}")
    print(f"Val frames:   {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda"))

    model = PretrainedYOLO().to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {n_total:,}")
    print(f"Trainable parameters: {n_trainable:,}   (head only)")

    optimizer = torch.optim.Adam(model.head.parameters(), lr=LR)

    best_map = 0.0
    print("\nEpoch  total  box   obj   noobj cls    val_mAP   time")
    print("-" * 60)
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        avg_loss, comp = train_one_epoch_head_only(model, train_loader, optimizer, anchors, device)
        t_train = time.time() - t0

        t0 = time.time()
        val_map, _ = evaluate_map(model, val_ds, device, anchors)
        t_eval = time.time() - t0

        marker = ""
        if val_map > best_map:
            best_map = val_map
            torch.save(model.head.state_dict(), WEIGHTS_OUT)  # head only (small file)
            marker = "← best"
        print(f" {epoch:2d}/{EPOCHS}  {avg_loss:6.2f}  {comp['box']:5.2f} "
              f"{comp['obj']:.2f} {comp['noobj']:.2f} {comp['cls']:.2f}  "
              f"{val_map*100:6.2f}%   {t_train:.0f}+{t_eval:.0f}s  {marker}", flush=True)

    print(f"\nBest val mAP: {best_map*100:.2f}%")
    print(f"Best head weights saved to: {WEIGHTS_OUT}")

    # Final per-class breakdown with best weights, larger val sample.
    model.head.load_state_dict(torch.load(WEIGHTS_OUT, map_location=device))
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
