# Stage 3 — From-Scratch YOLO-Style Detector: Results

## Abstract

A 4.75-million-parameter YOLO-style detector with a 5-stage stride-2 backbone and
single-scale 13×13 detection head was trained from scratch for 50 epochs on
freedoom2 training maps. Anchor priors (3 sizes covering small/medium/large
enemies) were hand-selected; loss combined sum-reduced MSE/smooth-L1 (box),
binary cross-entropy (objectness), and cross-entropy (class). Training reached
**best val mAP@IoU=0.5 of 20.18% at epoch 15** with subsequent epochs oscillating
in the 16–20% range as overfitting set in. On a 2000-frame independent val sample,
the best-by-checkpoint weights achieved **mAP 21.12%** — a 4.9× improvement over
Stage 2's sliding-window baseline (4.30%) but still well below the architectural
ceiling estimated at ~55% for this from-scratch configuration. The result
validates the YOLO architectural choice while highlighting that *feature quality*
(addressed in Stage 4 by ImageNet-pretrained backbones), not the detection-head
design, is the primary remaining bottleneck.

---

## 1. Setup recap

| Item | Value |
|---|---|
| Architecture | 5-stage CNN backbone → 1×1 detection head |
| Parameters | 4,748,066 |
| Input resolution | 416 × 416 (resized from native 640 × 480) |
| Grid size | 13 × 13 (single scale) |
| Anchors | 3 per cell: (30,40), (60,80), (130,160) — pixels at input scale |
| Train frames | 12,499 (MAP01–15, MAP31) |
| Val frames | 13,800 (MAP16–25) |
| Optimizer | Adam, lr=1e-3, gradient clip max_norm=10 |
| Batch size | 16 |
| Epochs | 50 |
| Loss weights | λ_box=5.0, λ_obj=1.0, λ_noobj=0.5, λ_cls=1.0 |
| Per-epoch val sample | 500 frames |
| Final val sample | 2,000 frames |
| Device | Colab T4 GPU |
| Wall-clock | ~2h 45min (50 × ~200s/epoch) |

See `stages/stage_003_plan.md` for the full architectural reasoning.

---

## 2. Training dynamics

### 2.1 Loss curve

```
Epoch  total   box    obj    noobj  cls     val_mAP
─────────────────────────────────────────────────────
  1    16.08   0.61   5.95   6.63   3.78    0.67%
  4     8.98   0.37   3.65   2.68   2.12   11.27%
  7     6.76   0.30   2.87   2.32   1.22   17.72%
 10     4.99   0.24   2.18   1.96   0.65   19.76%
 15     2.47   0.15   0.97   1.12   0.17   20.18%   ← peak val mAP
 20     1.22   0.10   0.38   0.53   0.07   19.95%
 30     0.57   0.06   0.13   0.21   0.02   17.51%
 40     0.37   0.05   0.07   0.13   0.01   16.19%
 50     0.30   0.04   0.06   0.11   0.01   17.39%
```

The total loss falls **53×** across training (16.08 → 0.30); all four components
decay smoothly. Yet val mAP plateaus around epoch 12–15 and the subsequent 35
epochs produce no consistent improvement, oscillating between 15% and 20%.

This is **textbook overfitting**: the model fits the training set to near-zero
loss while validation accuracy stagnates. The save-best-by-val-mAP mechanism
correctly identifies epoch 15 as the optimal checkpoint, before the plateau
became degradation.

### 2.2 Loss-component dynamics

| Component | Start | End | Change | Interpretation |
|---|---|---|---|---|
| `box` | 0.61 | 0.04 | −93% | Box regression converges quickly and steadily |
| `obj` | 5.95 | 0.06 | −99% | Objectness for positive cells learned cleanly |
| `noobj` | 6.63 | 0.11 | −98% | Background suppression learned (the hardest part for early epochs) |
| `cls` | 3.78 | 0.01 | −99.7% | Classification on positive cells eventually near-perfect |

The early-epoch behavior is dominated by `obj` and `noobj` learning to discriminate
background from foreground — the bulk of the loss-mass reduction in the first 5
epochs comes from learning "most cells contain nothing." This matches the YOLOv3
training pattern documented in the original paper.

By epoch 15, all components are below 1.0; further reduction is overfitting to
training-specific patterns that don't transfer to val.

---

## 3. Per-class results (best-weights, 2000-frame val sample)

```
Class               mAP@0.5    Tier
──────────────────────────────────────
ChaingunGuy         42.05%     ── strong ──
Archvile            33.39%
HellKnight          31.22%
LostSoul            30.51%
ShotgunGuy          28.29%
DoomImp             27.56%
Demon               27.43%
BaronOfHell         26.30%
Revenant            22.40%     ── decent ──
Cacodemon           20.99%
Cyberdemon          16.48%
Arachnotron         16.29%
Fatso               13.42%
Zombieman           11.78%     ── weak ──
PainElemental       10.68%
SpiderMastermind     0.00%     ── failing ──
Spectre              0.25%
──────────────────────────────────────
mAP                 21.12%
```

### 3.1 Tier observations

**Strong tier (>30%): ChaingunGuy, Archvile, HellKnight, LostSoul**
- All have visually distinctive silhouettes at the 416×416 resolution.
- ChaingunGuy's mounted-chaingun pose is hard to mistake for anything else.
- LostSoul's floating-skull silhouette is unique among classes.
- Archvile and HellKnight have characteristic vertical proportions.

**Decent tier (20-30%): ShotgunGuy, DoomImp, Demon, BaronOfHell, Revenant, Cacodemon**
- Common enemies with consistent silhouettes; many training examples.
- BaronOfHell at 26% is meaningful given the visual near-identity with HellKnight
  (see Stage 1 §5.4 — same pair confusion expected here).

**Weak tier (10-20%): Cyberdemon, Arachnotron, Fatso, Zombieman, PainElemental**
- Cyberdemon (16%) is *surprisingly* low for such a distinctive enemy. Likely
  cause: only ~800 train instances mostly from MAP31, where the agent often
  approaches a Cyberdemon from one angle. Limited pose variety.
- Zombieman (12%) is the headline disappointment. Despite ~2,300 train
  instances, it ends up with the lowest per-class AP among non-failing classes.
  Likely cause: zombies appear small on screen, often partially occluded, and
  visually generic — they get confused with bullet-puffs, weapon edges, and
  every humanoid in the dataset.

**Failing tier (~0%): Spectre, SpiderMastermind**
- **Spectre (0.25%)** is the predicted failure. The sprite is byte-identical to
  Demon (see `sprites/animations/Demon_sheet.png` and `sprites/animations/Spectre_sheet.png`).
  The runtime semi-transparency cue is too subtle for the small from-scratch
  backbone to learn at this resolution. Demon scored 27%; Spectre scored 0.25%.
  This is **a data-distribution result**, not a training failure: the model
  literally cannot distinguish them from RGB pixels alone with the features it
  can learn from this dataset.
- **SpiderMastermind (0%)** is the second predicted failure. Only 147 training
  crops. The model never sees enough variety to learn the class reliably.

### 3.2 Headline numbers

```
Stage 2 (sliding window):     4.30% mAP
Stage 3 (YOLO from scratch): 21.12% mAP
                            ─────────────
                            4.9× improvement
```

---

## 4. Discussion

### 4.1 The architecture choice was the right call

Stage 2's sliding-window approach hit 4.30% mAP because the underlying
classifier had no concept of "background." Stage 3's YOLO design directly
addresses this:

| Stage 2 problem | Stage 3 fix |
|---|---|
| No background class — classifier hallucinated enemies everywhere | Per-cell objectness logit trained on negative locations |
| Fixed-aspect square windows | Per-anchor box regression with separate w/h |
| Fixed grid stride misaligned with enemy positions | Sub-cell offsets (`tx, ty` sigmoid-bounded) per anchor |
| 802 classifier passes per frame | Single forward pass over full image |

The 4.9× mAP improvement quantifies the architectural value. Inference time is
also dramatically better: Stage 2 took ~0.79 s/frame on CPU; Stage 3 inference
on T4 GPU takes ~5–10 ms/frame, enough for real-time use.

### 4.2 Severe overfitting after epoch 15

The 25 epochs after the val-mAP peak produced no improvement — only train-loss
reduction. By epoch 50, train loss is 0.30 while val mAP has actually drifted
*downward* from 20.18% (epoch 15) to 17.39%. The model has learned training-
specific pixel patterns that don't transfer.

Mitigations available in subsequent stages:
- **Augmentation (Stage 6)**: random flips, color jitter, mosaic — multiplies
  effective training data, delays overfitting onset.
- **Dropout in the detection head**: cheap regularization.
- **Weight decay**: should have been on from the start; Stage 3 omitted it for
  simplicity but it's typically 5e-4 for YOLO training.
- **Cosine learning rate decay**: gradual lr reduction over epochs helps the
  model fine-tune late in training rather than thrashing.

These together typically push from-scratch YOLO another 5–15 pp mAP on
small-data regimes. Section 5 of this writeup suggests this is the *second*-
biggest gain available; the biggest is pretrained features (Stage 4).

### 4.3 The Spectre catastrophe is a clean diagnostic

Spectre's 0.25% AP vs Demon's 27.43% is one of the most striking results in the
project so far. Both have the *same* ground-truth bbox shapes, the *same*
sprite data, the *same* class-list position. The only difference is the
runtime rendering: Spectre is drawn semi-transparent.

For any vision model operating on RGB pixels, distinguishing "an opaque Demon"
from "a 30%-transparent Demon overlaid on background" requires recognizing
fine-grained alpha-blending artifacts. A small from-scratch CNN with 4.7M
parameters trained on 23k Doom frames cannot learn this — there isn't enough
training signal, and the features available at this scale don't encode this
kind of texture variation.

This is **not** a training failure. It's an inherent limitation of the data
distribution. The bbox ground truth is correct, the labels are correct, but
the input doesn't contain enough information for *this* model to solve the
problem. Pretrained backbones with much richer feature representations might
recover some performance on this class (transparency is a known visual concept
in natural images), but the ceiling for Spectre likely remains 20–40% even in
the best case. **This is a real-world long-tail class that the project's
Stage 9 evaluation should explicitly call out.**

### 4.4 SpiderMastermind is a small-data problem, not a feature problem

SpiderMastermind 0% AP comes from a different mechanism: only 147 training
instances. The model never sees enough examples to learn the class. Unlike
Spectre, the visual cues are *there* (the giant spider boss is unmistakable),
but the model never converges on them because gradient updates from this
class are vanishingly rare during training.

Mitigations:
- **Class-balanced sampling**: oversample SpiderMastermind frames during
  training so each minibatch has at least one.
- **Focal loss (Stage 8)**: down-weights well-classified easy examples,
  amplifying gradient from rare classes.
- **More data**: capture additional frames from the maps containing
  SpiderMastermind (MAP21, MAP23). This is the simplest fix but requires
  re-running data capture.

A modest amount of any of these should push SpiderMastermind from 0% to
20–40% without changing the architecture.

### 4.5 The per-class disparity is huge — and predictable

```
Best:  ChaingunGuy 42%
Worst: SpiderMastermind 0% / Spectre 0.25%

Range: ~42 pp between best and worst
```

The macro-averaged mAP of 21.12% hides this enormous variance. A per-class
report is essential — and for the writeup, the *narrative* should explicitly
note that "21% mAP" isn't "21% on each class" but "great on some, broken
on others." This is realistic and reflects the underlying data distribution
faithfully.

### 4.6 Compared to Stage 1's classifier accuracy

| Metric | Stage 1 (classifier) | Stage 3 (detector) |
|---|---|---|
| Best val accuracy/mAP | 71.25% | 21.12% |
| What was measured | Class given a perfect crop | Class + box localization (IoU ≥ 0.5) |
| Comparable? | No — Stage 3 also has to find the enemy | |

Detection is fundamentally harder than classification — the model must both
localize and classify in one pass. The 71% → 21% drop is not "the detector
is worse at classifying" but "the detector has to do more work and only gets
credit when both pieces are right."

The truer comparison is Stage 2 → Stage 3: same task, same evaluator, different
architecture. 4.30% → 21.12% is the meaningful improvement.

---

## 5. Implications for Stage 4

Stage 4 (pretrained backbone, frozen) is expected to be the largest single jump
in the project arc. The prediction:

```
Stage 3 (from-scratch backbone):     21% mAP
Stage 4 (frozen ImageNet ResNet18):  50-65% mAP  (estimated)
Stage 5 (unfrozen, fine-tuned):      70-80% mAP  (estimated)
```

The reasoning: Stage 3 is bottlenecked on feature quality, not detection-head
design. The detection head architecture itself is correct (validated by 21%
mAP on classes the model *can* see well). What's missing is rich, transferable
visual features in the backbone — exactly what ImageNet pretraining provides.

A standard transfer-learning result for small-data object detection:

| Backbone | Typical mAP on small custom datasets |
|---|---|
| From scratch | 15–30% |
| Frozen ImageNet pretrained | 40–60% |
| Unfrozen fine-tuned | 60–80% |

Stage 3's 21% is at the lower end of expected for from-scratch. Stage 4's
target of ~50%+ would represent a 2.4× jump — and a meaningful narrative beat
for the writeup.

---

## 6. Reproducibility

| Artifact | Location |
|---|---|
| Training script | `stage3.py` (~550 lines) |
| Plan / architecture doc | `stages/stage_003_plan.md` |
| Colab notebook | `stage3_colab.ipynb` |
| Preresize utility | `preresize_data.py` (640×480 PNG → 416×416 JPEG) |
| Best weights | `stage3_best.pt` (4.7M params, epoch 15) |
| Random seed | 42 |

Run: upload `doom_data_416.zip` to Drive, run `stage3_colab.ipynb` on T4. ~2h 45m.

---

## 7. Conclusions

A from-scratch YOLO-style detector achieves 21.12% mAP — a 4.9× improvement
over the Stage 2 sliding-window baseline, validating the architectural choice.
Performance is bottlenecked by two distinct issues: (1) the from-scratch
backbone's feature quality, addressable by pretraining in Stage 4; and (2)
per-class data and ambiguity issues (Spectre, SpiderMastermind) requiring
loss reshaping and additional data in Stages 6–8. The overfitting curve
visible after epoch 15 is the most directly actionable target — augmentation
and regularization in the next iteration should close some of the gap with no
architectural change. Stage 4's pretrained backbone is expected to deliver the
largest single mAP jump in the project arc.
