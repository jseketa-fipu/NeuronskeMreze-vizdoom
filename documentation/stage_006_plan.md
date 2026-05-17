# Stage 6 — Data Augmentation: Plan

## Goal

Stage 5's val mAP plateaued at 30.62% by epoch 5 — clear overfitting (train
loss kept falling 8× while val flatlined). The cause is capacity-to-data
mismatch: 11.2M trainable params vs ~12.5k train frames.

Stage 6 introduces **on-the-fly data augmentation** to multiply effective
training-set diversity without capturing more frames. Same architecture, same
optimizer, same LRs — only the Dataset class gets randomized transforms.

Expected outcome: **35–45% mAP** (+5–15 pp over Stage 5).

This stage also explicitly *demonstrates* the augmentation lesson the peer's
project missed — including a deliberately-broken variant (heavy color jitter)
as a comparison point.

---

## 1. Augmentations applied

Two safe, well-understood, bbox-aware transforms applied at training time only
(val and test see no augmentation):

### 1.1 Horizontal flip (50% probability)

- Image: flip along the width axis.
- Bboxes: `cx_new = 1 - cx_old`, others unchanged.
- Safe for Doom: enemies are roughly symmetric (no left/right asymmetric
  features that flipping would invalidate, with the minor exception of
  weapon-holding hand orientation on zombies — accepted as noise).

### 1.2 Color jitter (modest)

```
brightness: ±20%
contrast:   ±20%
saturation: ±15%
hue:        unchanged (no perturbation — Doom's palette is intentional)
```

These ranges are deliberately *modest*. The Stage 1 §6 reference noted that
the peer's project used `channel_shift_range=150` and *destroyed* the signal
for some classes (Doom's colored monsters are partly identified by their
palette). Stage 6 uses standard photographic ranges that introduce variation
without breaking class identity.

Bboxes unchanged (color jitter is photometric, not geometric).

### 1.3 What we're NOT doing (and why)

- **Random crop / random scale**: would require bbox cropping/scaling logic;
  edge cases (enemy partially cropped out, scaled below `MIN_BBOX_AREA`) need
  filter handling. Defer to Stage 8 if needed.
- **Mosaic** (4-image stitching): high-impact but complex implementation;
  defer to Stage 8.
- **Random rotation**: would require bbox-rotation logic (axis-aligned bboxes
  don't rotate cleanly); also unrealistic for Doom (no rotated cameras).
- **Random hue shift**: would change the palette signal; *explicitly avoided*
  per the §1.1 reasoning.

The Stage 6 baseline is the smallest set of augmentations that should still
produce a measurable improvement, kept simple so the cause/effect is clear.

---

## 2. What changes from Stage 5

| Item | Stage 5 | Stage 6 |
|---|---|---|
| Backbone | Fine-tuned ResNet18 | Fine-tuned ResNet18 (same) |
| Detection head | 1×1 conv → 66 channels | Same |
| Optimizer | Adam, backbone 1e-4, head 1e-3 | Same |
| Train Dataset | No augmentation | **Flip + color jitter** |
| Val Dataset | No augmentation | No augmentation (same) |
| Epochs | 30 | **40** (augmentation delays convergence) |
| Best-by-val | val mAP | val mAP (same) |
| Loss / decoder / NMS / eval | identical | identical |

---

## 3. Why augmentation should help

Stage 5's overfitting was structural: the model saw the same ~12.5k training
frames repeatedly, with no variation. By epoch 5 it had memorized them. The
remaining 25 epochs were *gradient-on-training-set-noise*, not learning.

Augmentation breaks this:
- **Each epoch shows the model a slightly different version of each frame**:
  different horizontal flip, different brightness/contrast/saturation. The
  effective training-set size becomes much larger (effectively, infinitely
  many small variations of each frame).
- **The model must learn features invariant to these variations**: an Imp is
  an Imp regardless of orientation or lighting. This produces more
  generalizable features.
- **Overfitting is delayed but not eliminated**: with 40 epochs, val should
  continue improving past where Stage 5 plateaued.

Expected loss/mAP dynamics:
- **Train loss decreases more slowly** because each example is harder (random
  perturbation).
- **Val mAP continues climbing longer** before plateauing.
- **Final val mAP is higher** by 5–15 pp.

---

## 4. Implementation

Single new Dataset class (`FrameDetectionDatasetAugmented`) with an `augment`
flag. Train Dataset uses `augment=True`; val uses `augment=False`. Everything
else is reused from Stage 5.

Augmentation order in `__getitem__`:
1. Read + resize image (no change).
2. Convert to tensor in [0, 1].
3. **If augment**:
   - With 50% probability: horizontal flip image + adjust bbox `cx`.
   - Apply `torchvision.transforms.ColorJitter` with modest ranges.
4. ImageNet-normalize.

---

## 5. Risk: the Stage 1 §6 finding

Stage 1 explored augmentation in option B and found that *light* augmentation
helps (~73-74% accuracy), close to dropout. Heavy augmentation (the peer's
mistake) hurts.

Stage 6 is constrained to the light-augmentation regime. Hue shift is
explicitly omitted; color-jitter ranges are conservative. If results show no
improvement or a regression, the most likely cause is over-aggressive jitter
ranges; reduce brightness/contrast/saturation by half and retry.

---

## 6. Expected results

| Class category | Stage 5 AP | Predicted Stage 6 AP |
|---|---|---|
| Strong (≥39%) | 39–49% | +5–10 pp |
| Decent (30–39%) | 30–35% | +5–10 pp |
| Modest (15–30%) | 15–30% | +5–15 pp (rare classes benefit more) |
| Failing (<5%) | Spectre 0%, etc. | unchanged (fundamental, not data-volume) |

Predicted mAP: **35–45%** based on typical augmentation lifts in small-data
detection regimes.

---

## 7. Reproducibility

| Artifact | Location |
|---|---|
| Script | `stage6.py` |
| Notebook | `stage6_colab.ipynb` |
| Best weights | `stage6_best.pt` |
| Random seed | 42 |
