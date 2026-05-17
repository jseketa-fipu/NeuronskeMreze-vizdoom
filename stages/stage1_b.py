"""Stage 1 — Extended (Option B): regularization + augmentation ablation.

Trains four variants of the same architecture and compares their final val accuracy:
  baseline           — no regularization, no augmentation
  dropout            — Dropout(0.5) before the classifier head
  augment            — geometric + light photometric augmentation on train
  dropout+augment    — both

Each variant runs for EPOCHS epochs; best-by-val-accuracy weights are saved
per variant. A summary table is printed at the end.

Uses the same per-map split as stage1.py (TRAIN_MAPS / VAL_MAPS).
"""
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

CROPS_DIR = Path("crops")
EPOCHS = 15
BATCH_SIZE_TRAIN = 128
BATCH_SIZE_VAL = 256

# Class list and split — same as stage1.py
with open(Path("data") / "classes.txt") as f:
    ENEMY_CLASSES = [line.strip() for line in f
                     if line.strip() and not line.startswith("#")]
NUM_CLASSES = len(ENEMY_CLASSES)
TRAIN_MAPS = set(f"MAP{i:02d}" for i in range(1, 16)) | {"MAP31"}
VAL_MAPS   = set(f"MAP{i:02d}" for i in range(16, 26))


class CropDataset(Dataset):
    """Loads pre-generated 64x64 enemy crops filtered by allowed map names.
    Optionally applies a torchvision transform pipeline (used for augmentation)."""
    def __init__(self, root: Path, allowed_maps: set, transform=None):
        self.items = []
        for class_id, name in enumerate(ENEMY_CLASSES):
            class_dir = root / name
            if not class_dir.is_dir():
                continue
            for f in class_dir.iterdir():
                if f.suffix != ".png":
                    continue
                map_name = f.name.split("_", 1)[0]
                if map_name in allowed_maps:
                    self.items.append((f, class_id))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, class_id = self.items[idx]
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        tensor = torch.from_numpy(img)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, class_id


class SimpleCNN(nn.Module):
    """Same as stage1.py, optionally with a Dropout layer before the linear head."""
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),nn.ReLU(), nn.MaxPool2d(2),
        )
        head_layers = [nn.Flatten()]
        if dropout > 0:
            head_layers.append(nn.Dropout(dropout))
        head_layers.append(nn.Linear(128 * 8 * 8, num_classes))
        self.classifier = nn.Sequential(*head_layers)

    def forward(self, x):
        return self.classifier(self.features(x))


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, seen = 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        seen += labels.size(0)
    return correct / seen


def run_variant(name: str, dropout: float, train_transform, device, val_loader):
    """Train one variant from scratch; return its (best_val_acc, best_epoch, train_acc_at_best)."""
    print(f"\n{'=' * 70}\nVariant: {name}\n{'=' * 70}")
    train_ds = CropDataset(CROPS_DIR, TRAIN_MAPS, transform=train_transform)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    model = SimpleCNN(NUM_CLASSES, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val, best_epoch, best_train = 0.0, 0, 0.0
    ckpt_path = Path(f"stage1_b_{name}.pt")
    print("Epoch  train_loss  train_acc   val_acc")
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        marker = ""
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch
            best_train = tr_acc
            torch.save(model.state_dict(), ckpt_path)
            marker = "← best"
        print(f" {epoch:2d}/{EPOCHS}    {tr_loss:.3f}     {tr_acc*100:5.1f}%    {val_acc*100:5.1f}%   {marker}")
    print(f"  -> elapsed {(time.time()-t0)/60:.1f} min, best val {best_val*100:.2f}% at epoch {best_epoch}")
    return {"name": name, "best_val": best_val, "best_epoch": best_epoch,
            "best_train": best_train, "ckpt": str(ckpt_path)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Val dataset stays constant across variants (no augmentation on val).
    val_ds = CropDataset(CROPS_DIR, VAL_MAPS, transform=None)
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE_VAL, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )
    print(f"Val crops: {len(val_ds):,}")

    # Augmentation pipeline: applied only during training, only to the variants that opt in.
    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
    ])

    variants = [
        # (name, dropout, train_transform)
        ("baseline",        0.0, None),
        ("dropout",         0.5, None),
        ("augment",         0.0, aug),
        ("dropout+augment", 0.5, aug),
    ]
    results = []
    for name, dp, tr in variants:
        results.append(run_variant(name, dp, tr, device, val_loader))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<22} {'Best val':>10} {'@epoch':>8} {'train@best':>12}")
    print("-" * 56)
    for r in results:
        print(f"{r['name']:<22} {r['best_val']*100:>9.2f}% {r['best_epoch']:>8d} {r['best_train']*100:>11.1f}%")
    print("-" * 56)
    best_overall = max(results, key=lambda r: r["best_val"])
    print(f"Winner: {best_overall['name']} ({best_overall['best_val']*100:.2f}%)")
    print(f"Best checkpoint: {best_overall['ckpt']}")


if __name__ == "__main__":
    main()
