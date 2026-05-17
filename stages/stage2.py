"""Stage 2 — Sliding-Window Detector.

Reuses the trained Stage 1 classifier as a *detection* building block by sliding it
across full 640x480 val frames at multiple scales, running the classifier on each
window, then deduplicating overlapping detections with per-class non-maximum
suppression. Evaluates with per-class average precision (AP) at IoU 0.5 (VOC-style).

The result is intentionally bad — that's Stage 2's lesson. Sliding-window is the
naive detection baseline; numbers here motivate why purpose-built one-shot
detectors (Stages 3+) exist.

Sampling: to keep CPU runtime reasonable, we evaluate on a random subset of val
frames (default 1000). Bump SAMPLE_SIZE for more thorough eval at the cost of time.
"""
import time
import random
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
import torch.nn as nn

# ---- Constants ----
DATA_DIR = Path("data")
WEIGHTS = Path("stage1_best.pt")          # uses the baseline Stage 1 model
SAMPLE_SIZE = 1000                         # random val frames to evaluate
SCALES = [50, 90, 150, 220]                # window sizes in pixels (square)
STRIDE = 32                                # pixels between adjacent window positions
CONFIDENCE_THRESH = 0.4                    # min classifier softmax confidence
NMS_IOU = 0.45                             # IoU threshold for per-class NMS
AP_IOU = 0.5                               # IoU threshold counting a TP for AP
BATCH_SIZE = 256                           # classifier inference batch size
SEED = 42

# Class list (matches stage1.py / classes.txt)
with open(DATA_DIR / "classes.txt") as f:
    ENEMY_CLASSES = [line.strip() for line in f
                     if line.strip() and not line.startswith("#")]
NUM_CLASSES = len(ENEMY_CLASSES)
VAL_MAPS = [f"MAP{i:02d}" for i in range(16, 26)]


# ---- Model (must match stage1.py) ----
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---- Geometry helpers ----
def iou(box_a, box_b):
    """IoU of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def parse_ground_truth(txt_path, frame_w, frame_h):
    """Parse a YOLO-format .txt file into [(class_id, x1, y1, x2, y2), ...] in pixels."""
    gts = []
    with open(txt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            x1 = max(0, int((cx - bw / 2) * frame_w))
            y1 = max(0, int((cy - bh / 2) * frame_h))
            x2 = min(frame_w, int((cx + bw / 2) * frame_w))
            y2 = min(frame_h, int((cy + bh / 2) * frame_h))
            if x2 > x1 and y2 > y1:
                gts.append((cid, x1, y1, x2, y2))
    return gts


# ---- Sliding-window detection on a single frame ----
def sliding_windows(frame, scales, stride):
    """Yield (window_64x64_BGR, (x1, y1, x2, y2)) for every position+scale."""
    H, W = frame.shape[:2]
    for s in scales:
        if s > H or s > W:
            continue
        for y in range(0, H - s + 1, stride):
            for x in range(0, W - s + 1, stride):
                crop = frame[y:y + s, x:x + s]
                crop64 = cv2.resize(crop, (64, 64))
                yield crop64, (x, y, x + s, y + s)


def detect_in_frame(model, frame, device):
    """Return [(class_id, confidence, x1, y1, x2, y2), ...] after NMS, per-class."""
    crops = []
    bboxes = []
    for crop, bbox in sliding_windows(frame, SCALES, STRIDE):
        crops.append(crop)
        bboxes.append(bbox)
    if not crops:
        return []

    # Batch classify all windows.
    arr = np.stack(crops)                              # (N, 64, 64, 3) BGR
    arr = arr[..., ::-1]                               # BGR -> RGB (channel-axis reverse)
    arr = np.ascontiguousarray(arr, dtype=np.float32) / 255.0
    arr = arr.transpose(0, 3, 1, 2)                    # (N, 3, 64, 64)
    tensor = torch.from_numpy(arr)

    detections = []
    with torch.no_grad():
        for i in range(0, len(tensor), BATCH_SIZE):
            batch = tensor[i:i + BATCH_SIZE].to(device)
            probs = torch.softmax(model(batch), dim=1)
            confs, classes = probs.max(dim=1)
            confs = confs.cpu().numpy()
            classes = classes.cpu().numpy()
            for j in range(batch.size(0)):
                if confs[j] >= CONFIDENCE_THRESH:
                    x1, y1, x2, y2 = bboxes[i + j]
                    detections.append((int(classes[j]), float(confs[j]), x1, y1, x2, y2))

    # Per-class NMS.
    kept = []
    by_class = defaultdict(list)
    for d in detections:
        by_class[d[0]].append(d)
    for cls, dets in by_class.items():
        dets.sort(key=lambda d: -d[1])  # descending confidence
        suppressed = [False] * len(dets)
        for i in range(len(dets)):
            if suppressed[i]:
                continue
            kept.append(dets[i])
            for j in range(i + 1, len(dets)):
                if suppressed[j]:
                    continue
                if iou(dets[i][2:], dets[j][2:]) > NMS_IOU:
                    suppressed[j] = True
    return kept


# ---- Per-class AP (VOC2007 11-point interpolation) ----
def compute_ap(preds, gts_by_frame, cls):
    """preds: list of (frame_id, confidence, x1, y1, x2, y2) for this class.
       gts_by_frame: dict frame_id -> list of (x1, y1, x2, y2) for this class.
    Returns (ap, n_gt, n_pred, n_tp, n_fp)."""
    n_gt = sum(len(v) for v in gts_by_frame.values())
    if n_gt == 0:
        return None, 0, len(preds), 0, 0

    preds.sort(key=lambda p: -p[1])

    # Track which ground-truth boxes have been matched (per frame).
    matched = {fid: [False] * len(boxes) for fid, boxes in gts_by_frame.items()}
    tp, fp = [], []
    for p in preds:
        fid, _, x1, y1, x2, y2 = p
        gt_boxes = gts_by_frame.get(fid, [])
        best_iou, best_idx = 0.0, -1
        for k, gbox in enumerate(gt_boxes):
            if matched[fid][k]:
                continue
            i = iou((x1, y1, x2, y2), gbox)
            if i > best_iou:
                best_iou, best_idx = i, k
        if best_iou >= AP_IOU and best_idx >= 0:
            matched[fid][best_idx] = True
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / n_gt
    precision = cum_tp / (cum_tp + cum_fp + 1e-10)

    # 11-point interpolated AP
    ap = 0.0
    for t in np.arange(0, 1.01, 0.1):
        mask = recall >= t
        ap += (precision[mask].max() if mask.any() else 0.0) / 11
    return ap, n_gt, len(preds), int(sum(tp)), int(sum(fp))


# ---- Main ----
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Weights: {WEIGHTS}")
    print(f"Scales: {SCALES}, stride={STRIDE}, "
          f"conf>={CONFIDENCE_THRESH}, nms_iou={NMS_IOU}, ap_iou={AP_IOU}")

    model = SimpleCNN(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()

    # Collect all val (frame.png, frame.txt) pairs, sample SAMPLE_SIZE of them.
    all_pairs = []
    for map_name in VAL_MAPS:
        map_dir = DATA_DIR / map_name
        if not map_dir.is_dir():
            continue
        for png in sorted(map_dir.glob("*.png")):
            txt = png.with_suffix(".txt")
            if txt.exists():
                all_pairs.append((png, txt))
    random.shuffle(all_pairs)
    sample = all_pairs[:SAMPLE_SIZE]
    print(f"Val frames available: {len(all_pairs):,};  evaluating sample: {len(sample):,}")

    # Per-class collected predictions and ground truths.
    preds_by_class = defaultdict(list)              # cls -> [(frame_id, conf, box...)]
    gts_by_class = defaultdict(lambda: defaultdict(list))  # cls -> {frame_id -> [box...]}

    t0 = time.time()
    n_windows_total = 0
    for idx, (png_path, txt_path) in enumerate(sample):
        frame = cv2.imread(str(png_path))
        if frame is None:
            continue
        H, W = frame.shape[:2]
        for cid, x1, y1, x2, y2 in parse_ground_truth(txt_path, W, H):
            gts_by_class[cid][idx].append((x1, y1, x2, y2))

        # Estimate window count (one-time, for headline)
        if idx == 0:
            n_each = sum(((H - s) // STRIDE + 1) * ((W - s) // STRIDE + 1)
                         for s in SCALES if s <= min(W, H))
            print(f"Windows per frame: {n_each}")

        detections = detect_in_frame(model, frame, device)
        n_windows_total += sum(((H - s) // STRIDE + 1) * ((W - s) // STRIDE + 1)
                               for s in SCALES if s <= min(W, H))
        for cls, conf, x1, y1, x2, y2 in detections:
            preds_by_class[cls].append((idx, conf, x1, y1, x2, y2))

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta = (len(sample) - (idx + 1)) / rate / 60
            bar_len = 30
            done = int(bar_len * (idx + 1) / len(sample))
            bar = "█" * done + "░" * (bar_len - done)
            print(f"  [{bar}] {idx+1}/{len(sample)}  "
                  f"{rate:.1f} fps  ETA {eta:.1f}m", flush=True)

    elapsed = time.time() - t0
    print(f"\nProcessed {len(sample)} frames in {elapsed/60:.1f} min "
          f"({len(sample)/elapsed:.2f} fps)")
    print(f"Total windows classified: {n_windows_total:,}")

    # Per-class AP
    print(f"\n=== Per-class AP @IoU={AP_IOU} ===")
    aps = []
    for cls in range(NUM_CLASSES):
        ap, n_gt, n_pred, n_tp, n_fp = compute_ap(preds_by_class[cls],
                                                  gts_by_class[cls], cls)
        if ap is None:
            print(f"  {ENEMY_CLASSES[cls]:18s}    --    (no gt in sample)")
        else:
            aps.append(ap)
            print(f"  {ENEMY_CLASSES[cls]:18s}  {ap*100:5.2f}%   "
                  f"gt={n_gt:5d}  pred={n_pred:5d}  tp={n_tp:5d}  fp={n_fp:5d}")
    print("-" * 55)
    if aps:
        print(f"  {'mAP':18s}  {sum(aps)/len(aps)*100:5.2f}%   "
              f"(macro-average over {len(aps)} classes with gt)")


if __name__ == "__main__":
    main()
