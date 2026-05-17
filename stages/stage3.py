"""Stage 3 — From-scratch YOLO-style detector.

Single-scale, 3-anchor, 13x13 grid. See stages/stage_003_plan.md for design.

Requires GPU. CPU training is impractical (~50–100h extrapolated).
Designed to run on Colab T4 GPU (~30–45 min for 50 epochs).
"""
import time
import random
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- Constants ----
DATA_DIR = Path("data")
WEIGHTS_OUT = Path("stage3_best.pt")

INPUT_SIZE = 416
GRID_SIZE = 13
NUM_ANCHORS = 3

# Class list and split (same as stage1.py / stage2.py)
with open(DATA_DIR / "classes.txt") as f:
    ENEMY_CLASSES = [l.strip() for l in f if l.strip() and not l.startswith("#")]
NUM_CLASSES = len(ENEMY_CLASSES)
TRAIN_MAPS = set(f"MAP{i:02d}" for i in range(1, 16)) | {"MAP31"}
VAL_MAPS   = set(f"MAP{i:02d}" for i in range(16, 26))

# Anchor box sizes (width, height) in pixels at INPUT_SIZE.
# Hand-picked to cover small / medium / large enemies.
ANCHORS_PX = torch.tensor([
    [ 30.0,  40.0],   # small (Zombieman/ChaingunGuy/distant Imps)
    [ 60.0,  80.0],   # medium (Demons, Cacodemons, mid-range Imps)
    [130.0, 160.0],   # large (Cyberdemons, Mancubi, Pain Elementals)
])

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
LAMBDA_BOX   = 5.0
LAMBDA_OBJ   = 1.0
LAMBDA_NOOBJ = 0.5
LAMBDA_CLS   = 1.0
CONF_THRESH = 0.25
NMS_IOU = 0.45
AP_IOU = 0.5
VAL_EVAL_SIZE = 500   # frames per epoch-end mAP eval
SEED = 42


# ---- Dataset ---------------------------------------------------------------

class FrameDetectionDataset(Dataset):
    """Full-frame images with their YOLO-format boxes.
    __getitem__ returns (image_tensor, list_of_(cls,cx,cy,w,h))."""
    def __init__(self, root: Path, allowed_maps: set, input_size: int = INPUT_SIZE):
        self.items = []
        for map_name in sorted(allowed_maps):
            map_dir = root / map_name
            if not map_dir.is_dir():
                continue
            # Accept both .png (original) and .jpg (preresized via preresize_data.py).
            images = sorted(list(map_dir.glob("*.png")) + list(map_dir.glob("*.jpg")))
            for png in images:
                txt = png.with_suffix(".txt")
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
                    self.items.append((png, boxes))
        self.input_size = input_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        png_path, boxes = self.items[idx]
        img = cv2.imread(str(png_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)        # CHW
        return torch.from_numpy(img), boxes


def collate_fn(batch):
    """Batch images, keep boxes as variable-length list-of-lists."""
    images = torch.stack([b[0] for b in batch])
    boxes  = [b[1] for b in batch]
    return images, boxes


# ---- Model -----------------------------------------------------------------

def conv_block(in_ch, out_ch):
    """One stage of the backbone: conv → BN → LeakyReLU → conv(stride=2) → BN → LeakyReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    )


class YOLODetector(nn.Module):
    """5-stage stride-2 backbone (3→32→64→128→256→512) + 1×1 detection head."""
    def __init__(self, num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.out_per_anchor = 5 + num_classes
        self.backbone = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
        )
        self.head = nn.Conv2d(512, num_anchors * self.out_per_anchor, 1)

    def forward(self, x):
        return self.head(self.backbone(x))   # (B, A*(5+C), 13, 13)


# ---- Target assignment -----------------------------------------------------

def iou_wh(w1, h1, w2, h2):
    """IoU between two width-height pairs, assuming boxes centered at origin."""
    inter = min(w1, w2) * min(h1, h2)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def build_targets(gt_boxes_batch, anchors_px, grid_size, input_size, num_classes, device):
    """Build per-cell-per-anchor training targets for a batch.
    Returns box_target, obj_target, cls_target, pos_mask, noobj_mask all on `device`."""
    B = len(gt_boxes_batch)
    A = anchors_px.size(0)
    G = grid_size
    box_target = torch.zeros(B, A, 4, G, G, device=device)
    obj_target = torch.zeros(B, A, G, G, device=device)
    cls_target = torch.zeros(B, A, G, G, dtype=torch.long, device=device)
    pos_mask   = torch.zeros(B, A, G, G, dtype=torch.bool, device=device)
    noobj_mask = torch.ones(B, A, G, G, dtype=torch.bool, device=device)
    anchors_list = anchors_px.cpu().tolist()

    for b, gt_boxes in enumerate(gt_boxes_batch):
        for cls, cx, cy, w, h in gt_boxes:
            gt_w_px = w * input_size
            gt_h_px = h * input_size
            # Find best-IoU anchor by width-height only.
            best_iou, best_a = 0.0, 0
            for a, (aw, ah) in enumerate(anchors_list):
                i = iou_wh(gt_w_px, gt_h_px, aw, ah)
                if i > best_iou:
                    best_iou, best_a = i, a
            # Grid cell containing the GT box center.
            gx = min(G - 1, int(cx * G))
            gy = min(G - 1, int(cy * G))
            # Positive targets at (b, best_a, gy, gx).
            box_target[b, best_a, 0, gy, gx] = cx * G - gx
            box_target[b, best_a, 1, gy, gx] = cy * G - gy
            box_target[b, best_a, 2, gy, gx] = float(np.log(max(gt_w_px / anchors_list[best_a][0], 1e-6)))
            box_target[b, best_a, 3, gy, gx] = float(np.log(max(gt_h_px / anchors_list[best_a][1], 1e-6)))
            obj_target[b, best_a, gy, gx] = 1.0
            cls_target[b, best_a, gy, gx] = cls
            pos_mask[b, best_a, gy, gx] = True
            noobj_mask[b, best_a, gy, gx] = False
            # Mark other same-cell anchors with IoU > 0.5 as ignored.
            for a, (aw, ah) in enumerate(anchors_list):
                if a == best_a:
                    continue
                if iou_wh(gt_w_px, gt_h_px, aw, ah) > 0.5:
                    noobj_mask[b, a, gy, gx] = False
    return box_target, obj_target, cls_target, pos_mask, noobj_mask


# ---- Loss ------------------------------------------------------------------

def yolo_loss(predictions, box_target, obj_target, cls_target, pos_mask, noobj_mask,
              num_anchors, num_classes):
    """Multi-task YOLO loss. Returns (total_loss, per-component dict (cpu floats))."""
    B = predictions.size(0)
    A, OPA = num_anchors, 5 + num_classes
    pred = predictions.view(B, A, OPA, GRID_SIZE, GRID_SIZE)
    pred_tx = pred[:, :, 0]
    pred_ty = pred[:, :, 1]
    pred_tw = pred[:, :, 2]
    pred_th = pred[:, :, 3]
    pred_obj = pred[:, :, 4]
    pred_cls = pred[:, :, 5:5 + num_classes]   # (B, A, C, G, G)

    zero = torch.zeros((), device=predictions.device)

    n_pos = pos_mask.sum().item()
    if n_pos > 0:
        # Box loss: MSE on sigmoid(tx,ty), smooth-L1 on raw tw,th. All sum-reduced.
        box_loss = (
            F.mse_loss(torch.sigmoid(pred_tx[pos_mask]), box_target[:, :, 0][pos_mask], reduction="sum") +
            F.mse_loss(torch.sigmoid(pred_ty[pos_mask]), box_target[:, :, 1][pos_mask], reduction="sum") +
            F.smooth_l1_loss(pred_tw[pos_mask], box_target[:, :, 2][pos_mask], reduction="sum") +
            F.smooth_l1_loss(pred_th[pos_mask], box_target[:, :, 3][pos_mask], reduction="sum")
        )
        obj_loss = F.binary_cross_entropy_with_logits(
            pred_obj[pos_mask], obj_target[pos_mask], reduction="sum")
        # Class loss: CE for assigned anchors.
        pred_cls_perm = pred_cls.permute(0, 1, 3, 4, 2)   # (B, A, G, G, C)
        cls_loss = F.cross_entropy(pred_cls_perm[pos_mask], cls_target[pos_mask], reduction="sum")
    else:
        box_loss = zero; obj_loss = zero; cls_loss = zero

    if noobj_mask.sum() > 0:
        noobj_loss = F.binary_cross_entropy_with_logits(
            pred_obj[noobj_mask], obj_target[noobj_mask], reduction="sum")
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


# ---- Inference decoding ----------------------------------------------------

def decode_predictions(predictions, anchors_px, grid_size, input_size, num_classes, device):
    """Decode raw network output to (cx, cy, w, h, obj_prob, class_probs) — all normalized."""
    B = predictions.size(0)
    A = anchors_px.size(0)
    OPA = 5 + num_classes
    G = grid_size

    pred = predictions.view(B, A, OPA, G, G)
    yy, xx = torch.meshgrid(torch.arange(G, device=device),
                            torch.arange(G, device=device), indexing='ij')

    cx = (torch.sigmoid(pred[:, :, 0]) + xx) / G
    cy = (torch.sigmoid(pred[:, :, 1]) + yy) / G
    aw = anchors_px[:, 0].view(1, A, 1, 1).to(device)
    ah = anchors_px[:, 1].view(1, A, 1, 1).to(device)
    w = (aw * torch.exp(pred[:, :, 2])) / input_size
    h = (ah * torch.exp(pred[:, :, 3])) / input_size
    obj = torch.sigmoid(pred[:, :, 4])
    cls = F.softmax(pred[:, :, 5:5 + num_classes], dim=2)
    return cx, cy, w, h, obj, cls


def predictions_to_detections(cx, cy, w, h, obj, cls, conf_thresh, frame_w, frame_h):
    """Per frame: list of (class, conf, x1, y1, x2, y2) above threshold (no NMS yet)."""
    B, A, G, _ = obj.shape
    cls_probs, cls_ids = cls.max(dim=2)
    conf = obj * cls_probs
    out = [[] for _ in range(B)]
    for b in range(B):
        mask = conf[b] > conf_thresh
        if mask.sum() == 0:
            continue
        confs = conf[b][mask].detach().cpu().numpy()
        classes = cls_ids[b][mask].detach().cpu().numpy()
        cxs = cx[b][mask].detach().cpu().numpy() * frame_w
        cys = cy[b][mask].detach().cpu().numpy() * frame_h
        ws  = w[b][mask].detach().cpu().numpy() * frame_w
        hs  = h[b][mask].detach().cpu().numpy() * frame_h
        for i in range(len(confs)):
            x1 = max(0.0, cxs[i] - ws[i] / 2)
            y1 = max(0.0, cys[i] - hs[i] / 2)
            x2 = min(float(frame_w), cxs[i] + ws[i] / 2)
            y2 = min(float(frame_h), cys[i] + hs[i] / 2)
            if x2 > x1 and y2 > y1:
                out[b].append((int(classes[i]), float(confs[i]), x1, y1, x2, y2))
    return out


# ---- IoU + NMS + AP (mirrors stage2.py) -----------------------------------

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    return inter / (aa + bb - inter)


def nms_per_class(detections, iou_threshold):
    kept = []
    by_class = defaultdict(list)
    for d in detections:
        by_class[d[0]].append(d)
    for cls, dets in by_class.items():
        dets.sort(key=lambda d: -d[1])
        suppressed = [False] * len(dets)
        for i in range(len(dets)):
            if suppressed[i]:
                continue
            kept.append(dets[i])
            for j in range(i + 1, len(dets)):
                if not suppressed[j] and iou_xyxy(dets[i][2:], dets[j][2:]) > iou_threshold:
                    suppressed[j] = True
    return kept


def compute_ap_per_class(preds, gts_by_frame, ap_iou):
    n_gt = sum(len(v) for v in gts_by_frame.values())
    if n_gt == 0:
        return None
    preds.sort(key=lambda p: -p[1])
    matched = {fid: [False] * len(boxes) for fid, boxes in gts_by_frame.items()}
    tp, fp = [], []
    for p in preds:
        fid, _, x1, y1, x2, y2 = p
        gt_boxes = gts_by_frame.get(fid, [])
        best_iou, best_idx = 0.0, -1
        for k, gbox in enumerate(gt_boxes):
            if matched[fid][k]:
                continue
            i = iou_xyxy((x1, y1, x2, y2), gbox)
            if i > best_iou:
                best_iou, best_idx = i, k
        if best_iou >= ap_iou and best_idx >= 0:
            matched[fid][best_idx] = True
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    if not tp:
        return 0.0
    cum_tp = np.cumsum(tp); cum_fp = np.cumsum(fp)
    recall = cum_tp / n_gt
    precision = cum_tp / (cum_tp + cum_fp + 1e-10)
    ap = 0.0
    for t in np.arange(0, 1.01, 0.1):
        mask = recall >= t
        ap += (precision[mask].max() if mask.any() else 0.0) / 11
    return float(ap)


@torch.no_grad()
def evaluate_map(model, val_ds, device, anchors_px, sample_size=VAL_EVAL_SIZE):
    """Per-epoch mAP over a random val subset (kept small for speed)."""
    model.eval()
    indices = list(range(len(val_ds)))
    random.Random(SEED).shuffle(indices)
    indices = indices[:sample_size]

    preds_by_class = defaultdict(list)
    gts_by_class   = defaultdict(lambda: defaultdict(list))

    for fid_local, dataset_idx in enumerate(indices):
        img, gt_boxes = val_ds[dataset_idx]
        png_path = val_ds.items[dataset_idx][0]
        orig = cv2.imread(str(png_path))
        frame_h, frame_w = orig.shape[:2]
        for cls, cx, cy, w, h in gt_boxes:
            x1 = (cx - w / 2) * frame_w
            y1 = (cy - h / 2) * frame_h
            x2 = (cx + w / 2) * frame_w
            y2 = (cy + h / 2) * frame_h
            gts_by_class[cls][fid_local].append((x1, y1, x2, y2))
        img_b = img.unsqueeze(0).to(device)
        cx_d, cy_d, w_d, h_d, obj_d, cls_d = decode_predictions(
            model(img_b), anchors_px, GRID_SIZE, INPUT_SIZE, NUM_CLASSES, device)
        dets = predictions_to_detections(cx_d, cy_d, w_d, h_d, obj_d, cls_d,
                                         CONF_THRESH, frame_w, frame_h)[0]
        dets = nms_per_class(dets, NMS_IOU)
        for cls, conf, x1, y1, x2, y2 in dets:
            preds_by_class[cls].append((fid_local, conf, x1, y1, x2, y2))

    aps = []
    per_class = {}
    for cls in range(NUM_CLASSES):
        ap = compute_ap_per_class(preds_by_class[cls], gts_by_class[cls], AP_IOU)
        per_class[cls] = ap
        if ap is not None:
            aps.append(ap)
    mAP = sum(aps) / len(aps) if aps else 0.0
    return mAP, per_class


# ---- Training --------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, anchors_px, device):
    model.train()
    total = 0.0
    components = {"box": 0.0, "obj": 0.0, "noobj": 0.0, "cls": 0.0}
    n = 0
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
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    if device.type == "cpu":
        print("  WARNING: CPU training is impractical (~50-100h). Run on Colab T4.")

    anchors = ANCHORS_PX.to(device)
    train_ds = FrameDetectionDataset(DATA_DIR, TRAIN_MAPS)
    val_ds   = FrameDetectionDataset(DATA_DIR, VAL_MAPS)
    print(f"Train frames: {len(train_ds):,}")
    print(f"Val frames:   {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda"))

    model = YOLODetector().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_map = 0.0
    print("\nEpoch  total  box   obj   noobj cls    val_mAP   time")
    print("-" * 60)
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        avg_loss, comp = train_one_epoch(model, train_loader, optimizer, anchors, device)
        t_train = time.time() - t0

        t0 = time.time()
        val_map, per_class = evaluate_map(model, val_ds, device, anchors)
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

    # Final per-class breakdown using best weights.
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
