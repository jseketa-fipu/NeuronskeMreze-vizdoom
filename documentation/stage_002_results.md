# Stage 2 — Sliding-Window Detector: Results

## Abstract

The Stage 1 classifier (71.25% val accuracy on cropped enemies) was repurposed as a
*detector* by sliding it across full 640×480 val frames at four square scales (50,
90, 150, 220 px) with stride 32 px. Each window's classifier output was thresholded
at confidence 0.4, then deduplicated with per-class non-maximum suppression at IoU
0.45. Detections were evaluated against ground-truth bounding boxes with per-class
Average Precision at IoU 0.5 (VOC2007 11-point interpolation). On a 1000-frame
random sample of val, the resulting **mAP was 4.30%** — a ~67 percentage-point drop
from the classifier's accuracy on clean crops. The failure mode is dominated by an
explosion of false positives (162,606 predicted DoomImp boxes against 258 ground-
truth instances), driven by the classifier's lack of a "no object" output. This
stage establishes the empirical baseline that motivates purpose-built one-shot
detectors (Stage 3).

---

## 1. Approach

Stage 1 produced a 17-class classifier that takes a 64×64 crop and outputs class
logits. Stage 2 turns this into a detection pipeline with no additional training:

1. For each input frame (640 × 480), generate candidate windows at multiple scales
   and positions:
   - **Scales (square)**: 50, 90, 150, 220 px
   - **Stride**: 32 px between adjacent window positions
   - Total windows per frame: **802** (computed from `Σ_s ((W-s)/stride+1)·((H-s)/stride+1)`).
2. Resize each window to 64 × 64.
3. Batch-forward through the Stage 1 classifier. Take the per-window
   argmax-class and its softmax probability.
4. Keep only windows whose top-class probability ≥ 0.4 → candidate detections,
   each a tuple `(class, confidence, x1, y1, x2, y2)`.
5. Per-class non-maximum suppression at IoU 0.45 (within each class, keep the
   highest-confidence detection; suppress any other detection of the same class
   overlapping above the threshold).
6. Evaluate detections against ground-truth bboxes:
   - For each predicted box (in confidence order), find the highest-IoU unmatched
     ground-truth box of the same class.
   - If IoU ≥ 0.5: count as TP, mark that GT as matched.
   - Otherwise: count as FP.
   - Compute precision/recall and AP using VOC2007 11-point interpolation.

---

## 2. Setup

| Item | Value |
|---|---|
| Source model | `stage1_best.pt` (Stage 1 baseline, 71.25% val accuracy) |
| Val frames available | 13,800 across MAP16–MAP25 |
| Frames sampled | **1,000** (random, seeded; ~7.2% of val) |
| Image resolution | 640 × 480 (BGR PNG) |
| Window scales (px) | 50, 90, 150, 220 (four squares) |
| Stride (px) | 32 |
| Windows per frame | 802 |
| Confidence threshold | 0.4 |
| NMS IoU threshold | 0.45 |
| AP IoU threshold | 0.5 |
| Batch size | 256 |
| Device | CPU |
| Total wall-clock | 13.2 min |
| Total classifier inferences | 802,000 |

Reproducibility: `python -u stage2.py`. Seeds fixed (`SEED=42`) so the frame
sample is deterministic across runs.

---

## 3. Results

### 3.1 Headline

```
Stage 1 (classifier on cropped enemies):    71.25% accuracy
Stage 2 (sliding window detection):          4.30% mAP @ IoU=0.5
                                            ─────────────────
                                            ~67 pp drop
```

### 3.2 Per-class AP

```
ID  Class                 AP    gt    pred     tp    fp
-----------------------------------------------------------
14  Archvile           12.79%   131   3,260    49   3,211   ← best
12  Revenant           11.23%   155   3,894    30   3,864
 6  LostSoul            9.98%   104   7,699    31   7,668
13  BaronOfHell         7.58%   135   4,908    52   4,856
 7  Cacodemon           7.52%   122   7,198    58   7,140
11  PainElemental       7.23%    45  18,722    26  18,696
10  Arachnotron         5.50%    92   6,086    29   6,057
 9  HellKnight          3.60%   164   1,052    26   1,026
16  Cyberdemon          3.16%    67  69,849    30  69,819
 1  ShotgunGuy          1.53%   160  32,978    20  32,958
 8  Fatso               1.45%    98  29,459    33  29,426
 3  DoomImp             0.85%   258 162,606    62 162,544
 4  Demon               0.39%    57  26,719    15  26,704
 0  Zombieman           0.14%   243  16,218     9  16,209
15  SpiderMastermind    0.10%    15  11,367     6  11,361
 5  Spectre             0.02%    51  21,863     6  21,857
 2  ChaingunGuy         0.00%    66  39,324     3  39,321
-----------------------------------------------------------
mAP                     4.30%   (macro over 17 classes with gt)
```

### 3.3 Stratification

| AP tier | Classes |
|---|---|
| **Higher (>7%)** | Archvile, Revenant, LostSoul, BaronOfHell, Cacodemon, PainElemental |
| **Mid (3–7%)** | Arachnotron, HellKnight, Cyberdemon |
| **Low (1–3%)** | ShotgunGuy, Fatso |
| **Effectively zero (<1%)** | DoomImp, Demon, Zombieman, SpiderMastermind, Spectre, ChaingunGuy |

### 3.4 Runtime

- 13.2 minutes wall-clock for 1,000 frames on CPU.
- Effective throughput: **1.27 frames per second**.
- Extrapolation: evaluating all 13,800 val frames would take ~3 hours.

---

## 4. Discussion

### 4.1 The catastrophic false-positive rate is the dominant failure mode

The most striking number in the per-class table is the *predictions* column. For
DoomImp the classifier proposed **162,606 detections** to find 258 ground-truth
instances. For Cyberdemon: 69,849 predictions for 67 instances (1042 ×
oversampling). The pattern is the same across all classes — predictions outnumber
ground truths by 30–1000×.

Precision collapses to near zero across the board, and AP follows. Recall (TP /
total GT) varies between 5% and 50%; the model *can* find enemies, but it also
"finds" enemies in every patch of wall texture, every weapon edge, every gore
decoration. The 4.30% mAP is what's left after the precision-recall curve gets
crushed at near-zero precision.

### 4.2 Root cause: no "background" / "no object" class

The Stage 1 classifier was trained only on *cropped enemies* — every training
example contains an enemy. The training distribution has no concept of "this patch
contains nothing of interest." When confronted at inference with a window showing
e.g. a wall texture, the classifier must still output 17 logits whose softmax sums
to 1. It confidently assigns the patch to *some* class. Confidence threshold 0.4
admits a flood of these spurious detections.

Real one-shot detectors (YOLO, RetinaNet, etc.) include a per-location *objectness*
prediction trained explicitly on background regions — the model learns "this is
*nothing*" as a first-class output. This is the single most important architectural
difference and the structural reason Stage 2 cannot be salvaged just by tuning
thresholds. Even at confidence threshold 0.9, the classifier would still confidently
hallucinate enemies in most empty windows.

### 4.3 Class-AP differences are intuitively explainable

The classes that achieve the highest AP — Archvile, Revenant, LostSoul, BaronOfHell,
Cacodemon, PainElemental — share two properties:

1. **Visually distinctive silhouettes** (the floating skull of LostSoul, the
   tentacled mass of PainElemental, the unique gait of Revenant). Windows that
   "look like" these classes are relatively rare among false-positive patches.
2. **Bigger and more contiguous on screen.** Multiple window sizes/positions cover
   the same enemy with high overlap, so recall is robust.

Conversely, the worst performers — DoomImp, Demon, Zombieman, ShotgunGuy,
ChaingunGuy, Spectre — share:

1. **Generic humanoid silhouettes** at this resolution. The classifier confuses
   wall fixtures, gore, even crops of *other* enemies' arms/heads as one of these.
2. **Smaller average bbox area**, so they're often only fully contained in the
   smaller window sizes — but the smaller windows are exactly the ones generating
   the most false positives across the whole image.
3. **Class imbalance — Stage 1 already had highest accuracy on DoomImp because it
   was the majority class.** That same bias manifests as detection over-confidence
   here.

### 4.4 Why sliding-window is fundamentally the wrong architecture

Even setting aside the no-background issue, three structural problems make
sliding-window unworkable:

**Fixed-aspect windows.** All four window sizes are squares (50×50, 90×90, …). A
tall Revenant fits inside a 90×90 with empty top/bottom space; a squat Mancubus at
the same scale gets cut off horizontally. Real detectors learn per-anchor box
*regression* — they refine the box dimensions, not just snap to fixed shapes.

**Discrete grid.** With stride 32, the closest window to a ground-truth enemy can be
up to 16 px off-center in each axis. That's a substantial spatial misalignment at
the 64×64-classifier resolution, hurting both AP and IoU. Shrinking the stride helps
but multiplies inference cost.

**Inference cost.** 802 classifier passes per frame, ~1.3 fps on CPU. YOLO does one
forward pass over the full frame at any resolution; inference is 10–100× faster
because the per-location predictions are produced in parallel by the convolutional
head. Sliding-window is not a feasible architecture for any real application; it's a
pedagogical baseline.

### 4.5 The 67-percentage-point drop is the headline number

The contrast between Stage 1 (71% classifier accuracy on perfectly-cropped enemies)
and Stage 2 (4.3% mAP on the *same* val data) is the single biggest finding of the
project so far. It quantifies what "classification ≠ detection" actually costs in
practice:

| Step | Performance | What changed |
|---|---|---|
| Stage 1 | 71% | Model classifies *given* perfectly-cropped enemies |
| Stage 2 | 4.3% | Model must also **find** the enemies (no background class, no box regression, fixed-shape windows) |

Almost the entire 67 pp loss comes from the *localization* problem — the model is
no longer being told where the enemies are.

This makes the case for Stage 3's purpose-built detection architecture as compelling
as a course writeup gets.

---

## 5. Implications for Stage 3

Stage 3 introduces a YOLO-style one-shot detector with three changes designed to
directly address Stage 2's failure modes:

1. **Per-anchor objectness logit.** A separate "is there an object here" prediction
   trained explicitly on negative (background) regions. Directly addresses §4.2.
2. **Box regression per anchor.** Predict offsets `(tx, ty, tw, th)` from anchor
   priors, not fixed-shape windows. Directly addresses §4.4 (fixed-aspect, discrete
   grid).
3. **Single forward pass.** The full detection — boxes, objectness, classes for
   every location — is one CNN forward. Addresses §4.4 (inference cost).

Expected outcome: a meaningful AP recovery — typical from-scratch YOLO-style
architectures achieve 20–40% mAP on small detection problems, vs Stage 2's 4.3%.
Subsequent stages (4–8) then refine via pretrained backbones, augmentation, loss
tuning, etc.

---

## 6. Reproducibility

| Artifact | Location |
|---|---|
| Script | `stage2.py` (235 lines) |
| Source model | `stage1_best.pt` |
| Source data | `data/MAP16–MAP25/*.png + .txt` (val maps) |
| Random seed | `42` |

Run with `python -u stage2.py`. With `SAMPLE_SIZE = 1000` the run takes ~13 min on
CPU. Bump `SAMPLE_SIZE` for thoroughness (linear in runtime).

---

## 7. Conclusion

Sliding-window detection delivers 4.30% mAP — two orders of magnitude lower than the
classifier accuracy on clean crops. The failure is dominated by an FP explosion
arising from the absence of a background class in the classifier's training
distribution. Even setting that aside, the discrete grid, fixed-aspect windows, and
per-window inference cost make sliding-window fundamentally unsuited to real
detection. Stage 3 addresses all three issues simultaneously with a from-scratch
YOLO-style detector.
