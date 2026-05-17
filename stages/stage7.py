"""Stage 7 — Per-map vs random split contrast.

Re-evaluates Stage 6's saved weights on two contrasting val sets:
  A. Honest per-map val (MAP16-25 only) — should reproduce ~33.89%
  B. Leaky random val (random sample across train+val maps) — should be higher
     because ~half the sample is training data the model has memorized.

The gap quantifies the inflation that random-frame splitting would have caused
if used throughout the project. Does NOT train anything; pure inference.
"""
import random
from pathlib import Path
import torch

from stage3 import (
    ENEMY_CLASSES, NUM_CLASSES, ANCHORS_PX,
    TRAIN_MAPS, VAL_MAPS, DATA_DIR, SEED,
    evaluate_map,
)
from stage4 import FrameDetectionDatasetImageNet
from stage5 import FineTunedYOLO

WEIGHTS = Path("stage6_best.pt")
SAMPLE_SIZE = 500


def main():
    random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading weights: {WEIGHTS}")
    if not WEIGHTS.exists():
        print(f"  ERROR: {WEIGHTS} not found.")
        print(f"  Download stage6_best.pt from Colab session storage and place "
              f"in {Path.cwd()}.")
        return

    model = FineTunedYOLO().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()
    anchors = ANCHORS_PX.to(device)

    # Scenario A — honest per-map val (MAP16-25)
    print("\n" + "=" * 60)
    print("Scenario A — Honest per-map val (MAP16-25)")
    print("=" * 60)
    honest_ds = FrameDetectionDatasetImageNet(DATA_DIR, VAL_MAPS)
    print(f"Available frames: {len(honest_ds):,}")
    honest_map, honest_per_class = evaluate_map(
        model, honest_ds, device, anchors, sample_size=SAMPLE_SIZE)
    print(f"mAP @ {SAMPLE_SIZE}-frame sample: {honest_map*100:.2f}%")

    # Scenario B — leaky random val across train+val
    print("\n" + "=" * 60)
    print("Scenario B — Leaky random val (train+val maps combined)")
    print("=" * 60)
    all_maps = TRAIN_MAPS | VAL_MAPS
    leaky_ds = FrameDetectionDatasetImageNet(DATA_DIR, all_maps)
    train_frames_in_leaky = sum(
        1 for item in leaky_ds.items if item[0].parent.name in TRAIN_MAPS)
    val_frames_in_leaky = len(leaky_ds) - train_frames_in_leaky
    train_frac = train_frames_in_leaky / len(leaky_ds)
    print(f"Available frames: {len(leaky_ds):,}  "
          f"({train_frames_in_leaky:,} train + {val_frames_in_leaky:,} val)")
    print(f"Train frame fraction (in leaky pool): {train_frac*100:.1f}%")
    leaky_map, leaky_per_class = evaluate_map(
        model, leaky_ds, device, anchors, sample_size=SAMPLE_SIZE)
    print(f"mAP @ {SAMPLE_SIZE}-frame sample: {leaky_map*100:.2f}%")

    # Summary
    inflation = (leaky_map - honest_map) * 100
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Honest per-map val mAP:        {honest_map*100:6.2f}%")
    print(f"  Leaky random val mAP:          {leaky_map*100:6.2f}%")
    print(f"  Inflation (data-leakage cost): {inflation:+6.2f} pp")
    print(f"  Ratio (leaky / honest):        {leaky_map/honest_map:6.2f}×")

    # Per-class comparison
    print("\nPer-class AP (Honest -> Leaky):")
    print(f"  {'class':<18}  {'honest':>8}  {'leaky':>8}  {'delta':>7}")
    print(f"  {'-'*18}  {'-'*8}  {'-'*8}  {'-'*7}")
    for i, cls in enumerate(ENEMY_CLASSES):
        h = honest_per_class.get(i)
        l = leaky_per_class.get(i)
        h_s = f"{h*100:6.2f}%" if h is not None else "    --"
        l_s = f"{l*100:6.2f}%" if l is not None else "    --"
        if h is not None and l is not None:
            d = (l - h) * 100
            d_s = f"{d:+5.1f}"
        else:
            d_s = "  n/a"
        print(f"  {cls:<18}  {h_s}  {l_s}  {d_s}")


if __name__ == "__main__":
    main()
