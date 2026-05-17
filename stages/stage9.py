"""Stage 9 — Final test-set evaluation.

Takes Stage 6's saved weights (the cross-stage val winner at 33.89% mAP) and
evaluates on the held-out test set (MAP26-30, MAP32). This is the project's
honest unbiased final number — touched exactly once, never iterated.
"""
import random
import time
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch

from stage3 import (
    ENEMY_CLASSES, NUM_CLASSES, ANCHORS_PX, GRID_SIZE, INPUT_SIZE,
    DATA_DIR, SEED, AP_IOU, CONF_THRESH, NMS_IOU,
    decode_predictions, predictions_to_detections,
    nms_per_class, compute_ap_per_class,
)
from stage4 import FrameDetectionDatasetImageNet
from stage5 import FineTunedYOLO

WEIGHTS = Path("stage6_best.pt")
TEST_MAPS = {f"MAP{i:02d}" for i in range(26, 31)} | {"MAP32"}
MAX_SAMPLE = 2000


@torch.no_grad()
def evaluate_test_with_breakdown(model, test_ds, device, anchors_px, sample_size):
    """Like stage3's evaluate_map but also tracks per-map AP breakdown."""
    model.eval()
    indices = list(range(len(test_ds)))
    random.Random(SEED).shuffle(indices)
    indices = indices[:sample_size]

    preds_by_class = defaultdict(list)
    gts_by_class = defaultdict(lambda: defaultdict(list))
    preds_by_map = defaultdict(lambda: defaultdict(list))
    gts_by_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for fid_local, dataset_idx in enumerate(indices):
        img, gt_boxes = test_ds[dataset_idx]
        img_path = test_ds.items[dataset_idx][0]
        map_name = img_path.parent.name
        orig = cv2.imread(str(img_path))
        frame_h, frame_w = orig.shape[:2]
        for cls, cx, cy, w, h in gt_boxes:
            x1 = (cx - w / 2) * frame_w
            y1 = (cy - h / 2) * frame_h
            x2 = (cx + w / 2) * frame_w
            y2 = (cy + h / 2) * frame_h
            gts_by_class[cls][fid_local].append((x1, y1, x2, y2))
            gts_by_map[map_name][cls][fid_local].append((x1, y1, x2, y2))
        img_b = img.unsqueeze(0).to(device)
        cx_d, cy_d, w_d, h_d, obj_d, cls_d = decode_predictions(
            model(img_b), anchors_px, GRID_SIZE, INPUT_SIZE, NUM_CLASSES, device)
        dets = predictions_to_detections(cx_d, cy_d, w_d, h_d, obj_d, cls_d,
                                         CONF_THRESH, frame_w, frame_h)[0]
        dets = nms_per_class(dets, NMS_IOU)
        for cls, conf, x1, y1, x2, y2 in dets:
            preds_by_class[cls].append((fid_local, conf, x1, y1, x2, y2))
            preds_by_map[map_name][cls].append((fid_local, conf, x1, y1, x2, y2))

    # Overall mAP
    aps = []
    per_class = {}
    for cls in range(NUM_CLASSES):
        ap = compute_ap_per_class(preds_by_class[cls], gts_by_class[cls], AP_IOU)
        per_class[cls] = ap
        if ap is not None:
            aps.append(ap)
    mAP = sum(aps) / len(aps) if aps else 0.0

    # Per-map mAP
    per_map_mAP = {}
    for m in sorted(gts_by_map.keys()):
        map_aps = []
        for cls in range(NUM_CLASSES):
            ap = compute_ap_per_class(preds_by_map[m][cls], gts_by_map[m][cls], AP_IOU)
            if ap is not None:
                map_aps.append(ap)
        per_map_mAP[m] = sum(map_aps) / len(map_aps) if map_aps else 0.0

    return mAP, per_class, per_map_mAP


def main():
    random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading weights: {WEIGHTS}")
    if not WEIGHTS.exists():
        print(f"  ERROR: {WEIGHTS} not found.")
        return

    print(f"Test maps: {sorted(TEST_MAPS)}")

    model = FineTunedYOLO().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()
    anchors = ANCHORS_PX.to(device)

    test_ds = FrameDetectionDatasetImageNet(DATA_DIR, TEST_MAPS)
    available = len(test_ds)
    sample_size = min(available, MAX_SAMPLE)
    print(f"Test frames available: {available:,}")
    print(f"Sampling: {sample_size:,}")

    if available == 0:
        print("  ERROR: No test frames found. Check that data/MAPxx/ exist for test maps.")
        return

    print(f"\nRunning inference + evaluation... (~5-10 min on CPU)")
    t0 = time.time()
    test_map, per_class, per_map = evaluate_test_with_breakdown(
        model, test_ds, device, anchors, sample_size)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s")

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS — the project's headline number")
    print("=" * 60)
    print(f"\nOverall test mAP @ IoU=0.5:  {test_map*100:.2f}%")
    print(f"(Val mAP for comparison:       33.89%)")
    print(f"(Val→Test delta:               {(test_map - 0.3389)*100:+.2f} pp)")

    print("\nPer-class AP @ IoU=0.5:")
    print(f"  {'class':<18}  {'test AP':>8}")
    print(f"  {'-'*18}  {'-'*8}")
    for cls in range(NUM_CLASSES):
        ap = per_class[cls]
        if ap is None:
            print(f"  {ENEMY_CLASSES[cls]:<18}    --     (no gt in sample)")
        else:
            print(f"  {ENEMY_CLASSES[cls]:<18}  {ap*100:6.2f}%")

    print("\nPer-test-map mAP:")
    print(f"  {'map':<8}  {'mAP':>8}")
    print(f"  {'-'*8}  {'-'*8}")
    for m in sorted(per_map.keys()):
        print(f"  {m:<8}  {per_map[m]*100:6.2f}%")


if __name__ == "__main__":
    main()
