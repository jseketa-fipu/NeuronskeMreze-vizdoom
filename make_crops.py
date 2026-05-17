"""Generate cropped enemy images from captured frames + YOLO labels.

One-shot data-prep step for Stage 1. Reads every (frame.png, frame.txt) pair under
data/MAPxx/, crops each labelled bounding box, resizes to a fixed size, and saves
to crops/<ClassName>/<MAP>_<frame>_<box_idx>.png.

The map prefix in the filename is what enables per-map train/val/test splitting later.
"""
from pathlib import Path
import cv2
import numpy as np

DATA_DIR = Path("data")
CROPS_DIR = Path("crops")
CROP_SIZE = 64               # all crops resized to this square pixel size
PADDING_FACTOR = 0.05        # 5% padding around each bbox before cropping


def letterbox(crop, target=CROP_SIZE, fill=0):
    """Resize crop to fit in target×target preserving aspect ratio; pad rest with `fill`.
    Avoids the aspect-ratio distortion of plain cv2.resize on non-square inputs,
    and uses appropriate interpolation for up- vs down-scaling."""
    h, w = crop.shape[:2]
    scale = min(target / w, target / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)
    canvas = np.full((target, target, crop.shape[2]), fill, dtype=crop.dtype)
    y_off = (target - new_h) // 2
    x_off = (target - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas

# Read class names from auto-generated classes.txt; skip comment / blank lines.
with open(DATA_DIR / "classes.txt") as f:
    CLASS_NAMES = [line.strip() for line in f
                   if line.strip() and not line.startswith("#")]

# One subdirectory per class.
for name in CLASS_NAMES:
    (CROPS_DIR / name).mkdir(parents=True, exist_ok=True)

counts = {name: 0 for name in CLASS_NAMES}
skipped = 0  # crops we couldn't extract (degenerate bbox after clipping, missing image, ...)

for map_dir in sorted(DATA_DIR.glob("MAP*")):
    if not map_dir.is_dir():
        continue
    map_name = map_dir.name
    for txt_path in sorted(map_dir.glob("*.txt")):
        png_path = txt_path.with_suffix(".png")
        if not png_path.exists():
            continue
        img = cv2.imread(str(png_path))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]
        frame_stem = txt_path.stem  # e.g. "000339"
        with open(txt_path) as f:
            for box_idx, line in enumerate(f):
                parts = line.split()
                if len(parts) != 5:
                    continue
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                if not (0 <= cid < len(CLASS_NAMES)):
                    skipped += 1
                    continue
                # Expand bbox by padding factor before computing pixel coords.
                bw *= 1 + 2 * PADDING_FACTOR
                bh *= 1 + 2 * PADDING_FACTOR
                x1 = max(0, int((cx - bw / 2) * w))
                y1 = max(0, int((cy - bh / 2) * h))
                x2 = min(w, int((cx + bw / 2) * w))
                y2 = min(h, int((cy + bh / 2) * h))
                if x2 <= x1 or y2 <= y1:
                    skipped += 1
                    continue
                crop = img[y1:y2, x1:x2]
                crop = letterbox(crop)
                class_name = CLASS_NAMES[cid]
                out_name = f"{map_name}_{frame_stem}_{box_idx}.png"
                cv2.imwrite(str(CROPS_DIR / class_name / out_name), crop)
                counts[class_name] += 1
    nonzero = {k: v for k, v in counts.items() if v}
    print(f"{map_name}: cumulative {nonzero}")

print("\n=== Final per-class counts ===")
total = 0
for name in CLASS_NAMES:
    print(f"  {name:20s} {counts[name]:6d}")
    total += counts[name]
print(f"  {'TOTAL':20s} {total:6d}")
print(f"  {'skipped':20s} {skipped:6d}")
print(f"\nCrops written to: {CROPS_DIR.resolve()}")
