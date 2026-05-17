# Stage 3 — From-Scratch YOLO-Style Detector: Plan

## Goal

Replace the Stage 2 sliding-window kludge with a proper one-shot object detector
trained from scratch on the existing train/val frames. Same per-map split, same
class list, same mAP@0.5 metric.

The "from-scratch" framing means:
- **No pretrained backbone.** We train all weights on the Doom data from random
  initialization. (Stages 4–5 then *add* a pretrained backbone to contrast.)
- **No ultralytics / yolov5 / mmdet.** We write the model, loss, assignment rule,
  decoding, and NMS as plain PyTorch code, so the implementation is legible and
  the design choices are explicit.

Expected outcome: **15–30% mAP** on val — a meaningful recovery from Stage 2's
4.30%, but still well below what a pretrained backbone will give us in Stage 4.

---

## 1. Architecture

YOLOv3-lite, single-scale.

### Input
- Frames resized from native 640×480 → **416×416**. Bbox labels (already in
  normalized [0,1] YOLO format) need no transformation.
- 3 channels (RGB, normalized to [0,1]).

### Backbone
Five stride-2 blocks doubling channels each time:

```
Block 1:   3 →  32, downsample 416 → 208
Block 2:  32 →  64, downsample 208 → 104
Block 3:  64 → 128, downsample 104 →  52
Block 4: 128 → 256, downsample  52 →  26
Block 5: 256 → 512, downsample  26 →  13

Each block: Conv(3×3, pad=1) → BN → LeakyReLU → Conv(3×3, stride=2) → BN → LeakyReLU
```

Output feature map: **`(B, 512, 13, 13)`**. Each cell of the 13×13 grid corresponds
to a 32×32-pixel region of the input.

### Detection head
A single 1×1 convolution producing per-cell predictions:

```
Output: (B, A × (4 + 1 + C), 13, 13)
       = (B, 3 × (4 + 1 + 17),  13, 13)
       = (B, 66, 13, 13)
```

Where:
- **A = 3 anchors per cell** — three predefined box-shape priors
- **4** box parameters per anchor (`tx, ty, tw, th`)
- **1** objectness logit per anchor
- **C = 17** class logits per anchor

Total parameters: ~1.6M (still tiny by modern standards).

### Anchors

Three anchor sizes covering small / medium / large enemies. At the 416×416 input
resolution, in pixels:

```
A0 = (30,  40)   - small enemies (Zombieman, ChaingunGuy, distant Imps)
A1 = (60,  80)   - medium (Demons, close Imps, Cacodemons)
A2 = (130, 160)  - large (Cyberdemons, Mancubi, Pain Elementals)
```

These could be derived via k-means clustering on training ground-truth bboxes for
better fit; for Stage 3 we hardcode them as a reasonable approximation. Stage 8
could revisit anchor optimization.

---

## 2. Output decoding

Per anchor at cell (cx, cy):
```
predicted_box_center_x = (sigmoid(tx) + cx) / 13
predicted_box_center_y = (sigmoid(ty) + cy) / 13
predicted_box_width    = anchor_w * exp(tw) / 416
predicted_box_height   = anchor_h * exp(th) / 416
objectness             = sigmoid(to)
class_probs            = softmax(tc_1 ... tc_17)
```

- `tx, ty` are post-sigmoid → constrained to [0, 1] within the cell
- `tw, th` are post-exp → arbitrary positive scale relative to anchor
- Predicted box coords are normalized [0, 1] of input image (matches GT format)

---

## 3. Training target assignment

For each ground-truth box `(class, cx, cy, w, h)` (all normalized [0,1]):

1. Compute the cell containing the box center: `(int(cx*13), int(cy*13))`.
2. Among the 3 anchors at this cell, pick the one with the highest IoU to the GT
   box (treating both as centered at origin — only width/height matter).
3. Set positive targets at `(cell, best_anchor)`:
   - `tx_target = cx*13 - int(cx*13)`  (sub-cell offset, [0, 1])
   - `ty_target = cy*13 - int(cy*13)`
   - `tw_target = log(w*416 / anchor_w)`
   - `th_target = log(h*416 / anchor_h)`
   - `objectness_target = 1.0`
   - `class_target = class_id` (used by cross-entropy)

For all other (cell, anchor) positions in the grid:
- If predicted box IoU with any GT > 0.5 → **ignore** (don't contribute to
  objectness loss, but no positive signal either)
- Otherwise → **negative**: `objectness_target = 0.0` (only objectness loss
  contributes; no class or box loss)

This is the standard YOLOv3 assignment.

---

## 4. Loss

Three components, weighted:

```
L_total = λ_box * L_box  +  λ_obj * L_obj  +  λ_noobj * L_noobj  +  λ_cls * L_cls
```

Where:
- **L_box**: only for positive matches.
  `MSE(predicted_tx, target_tx) + MSE(predicted_ty, target_ty) +`
  `SmoothL1(predicted_tw, target_tw) + SmoothL1(predicted_th, target_th)`.
- **L_obj**: BCE on objectness logits, for positive matches only.
- **L_noobj**: BCE on objectness logits, for negative matches only.
- **L_cls**: CrossEntropy on class logits, for positive matches only.

Weights:
- `λ_box = 5.0`   (upweight; box regression matters)
- `λ_obj = 1.0`
- `λ_noobj = 0.5` (downweight; background is the majority class, would dominate otherwise)
- `λ_cls = 1.0`

These weights are the YOLOv1/v2/v3 defaults; reasonable to start with.

---

## 5. Training

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 initially, possibly with cosine decay |
| Batch size | 16 (limited by GPU memory at 416×416) |
| Epochs | 50 |
| Augmentation | None for Stage 3 baseline (Stage 6 adds it) |
| Best-by | Val mAP |

Per epoch:
1. Forward pass over all train batches; compute loss; backward.
2. Run inference on a val sample (~500 frames) and compute mAP.
3. Save best-by-val-mAP weights to `stage3_best.pt`.

Estimated runtime (Colab T4): ~30–45 min for 50 epochs.

---

## 6. Inference

Per frame:
1. Resize to 416×416, normalize, forward → `(1, 66, 13, 13)` output.
2. Decode every (cell, anchor) → `(class_probs, box_normalized, objectness)`.
3. Compute per-anchor confidence = `objectness × max(class_probs)`.
4. Discard anchors with confidence < 0.25 (tuneable).
5. Per-class NMS at IoU 0.45.
6. Resulting list of detections feeds the same Stage 2 mAP evaluator.

---

## 7. Reuse from previous stages

| Component | Reused from |
|---|---|
| `parse_ground_truth` | stage2.py |
| `iou(box, box)` | stage2.py |
| `compute_ap` (per-class) | stage2.py |
| `ENEMY_CLASSES` / `TRAIN_MAPS` / `VAL_MAPS` | stage1.py |

New components in `stage3.py`:
- `Backbone`, `DetectionHead`, `YOLODetector` — model classes
- `FrameDetectionDataset` — full-frame + bbox tensor dataset
- `build_targets(gt_boxes, anchors)` — assignment rule
- `YOLOLoss` — multi-task loss
- `decode_predictions(raw, anchors)` — output postprocessing
- Training loop with mAP eval

---

## 8. Why this is GPU-only

A single forward pass costs:
- Backbone: ~1.6 GFLOPs
- 50 epochs × 12,500 train frames / batch 16 = ~39k forward+backward passes

On Stage 1's CPU: 232k params, 64×64 input, 3 GFLOPs total per batch — trained in
42 minutes. Stage 3 has 7× more params, 42× more input pixels per image, and a
much bigger backbone. A linear extrapolation puts CPU training at ~50–100 hours.
Not feasible.

On Colab T4 (12 TFLOPs):  ~30–45 minutes for the full 50-epoch run.

So: **the training run goes on Colab**. Code is written to work on either device
(`device = "cuda" if torch.cuda.is_available() else "cpu"`); the data transfer is
the same train-zip path used for Stage 1 + B.

---

## 9. Open issues / decisions

- **Anchor selection.** Hardcoded vs. k-means on train boxes. Hardcoded is fine
  for Stage 3 baseline; revisit in Stage 8.
- **Class loss: CE vs. BCE per class.** CE assumes mutually exclusive classes
  (correct for our setup — each enemy is exactly one class). BCE allows multi-
  label (not what we want). Use CE.
- **Anchor box matching.** Best-IoU rule (above) vs. all-IoU-above-threshold (more
  permissive, multiple positives per GT). Best-IoU is canonical; use that.
- **Multi-scale prediction.** Real YOLOv3 has 3 output scales (13×13, 26×26,
  52×52) for small/medium/large objects. We use single-scale (13×13) for
  simplicity. Could add multi-scale in Stage 8 as a refinement.
- **Loss scale for `λ_box`.** Standard is 5.0; if box loss dominates we'll dial back.

---

## 10. Expected results

| Metric | Stage 1 (classifier) | Stage 2 (sliding window) | Stage 3 (YOLO from scratch) |
|---|---|---|---|
| Val accuracy / mAP | 71% | 4.3% | **estimated 15–30%** |
| Inference per frame | n/a (cropped input) | ~0.79s on CPU | ~30ms on T4 |

The headline contrast for the writeup:
- **Stage 1→2**: 67 pp drop showing classification ≠ detection.
- **Stage 2→3**: order-of-magnitude AP recovery from real detection architecture.
- **Stage 3→4**: further gain from pretrained features (the next stage).

Stage 3 is the inflection point where the project's modelling actually becomes
detection rather than classification with a sliding window. It's the largest
single coding step. The architecture is intentionally minimal so all the moving
parts (anchors, assignment, multi-task loss, decoding) are visible and editable.
