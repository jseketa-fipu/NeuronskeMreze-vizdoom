# Stage 8 — Focal Loss for Class Imbalance: Results

## Abstract

Stage 6's CrossEntropy class loss and BCE objectness loss were replaced with
focal-loss equivalents (γ=2.0; α=0.25 for objectness) to address class
imbalance. Architecture, optimizer, augmentation, and per-map split were
unchanged. After 40 epochs, **val mAP best-by-epoch reached 34.80% (peak at
epoch 20) but full-eval on the 2000-frame sample settled at 32.90%, a
−0.99 pp regression vs Stage 6 (33.89%).** Per-class deltas are heavily
mixed: substantial gains for LostSoul (+10.0 pp), small gains for ShotgunGuy
and Demon, but significant regressions for Cacodemon (−8.0), Fatso (−6.6),
Cyberdemon (−6.6), Archvile (−6.5), and DoomImp (−5.3). Focal loss
*reshaped* per-class performance without lifting the mAP ceiling, suggesting
the project's remaining bottleneck is not class imbalance per se but the
combination of small dataset + small model capacity + single-scale
prediction.

This is the project's second negative result (after Stage 4's frozen-
pretrained regression). Like Stage 4, the value lies in the *clarity* of
the negative finding and what it teaches about the dataset.

---

## 1. Setup recap

| Item | Value |
|---|---|
| Backbone | ResNet18 pretrained, fine-tuned |
| Augmentation | Horizontal flip + color jitter (same as Stage 6) |
| Optimizer | Adam, backbone 1e-4, head 1e-3 |
| Epochs | 40 |
| Batch size | 16 |
| **Class loss** | **Focal CE (γ=2.0)** instead of CrossEntropy |
| **Objectness loss** | **Focal BCE (γ=2.0, α=0.25)** instead of BCE |
| Box loss | smooth-L1 + MSE (unchanged) |
| Loss weights | λ_box=5, λ_obj=1, λ_noobj=0.5, λ_cls=1 (unchanged) |
| Device | Colab T4 GPU |
| Wall-clock | ~2h 20min |

The only change vs Stage 6 is the loss function. All other variables are held
constant for clean attribution.

---

## 2. Training dynamics

### 2.1 Loss curve

```
Epoch  total   box    obj    noobj  cls     val_mAP
─────────────────────────────────────────────────────
  1     4.89   0.51   0.45   0.68   1.54    15.21%
  5     1.28   0.16   0.22   0.20   0.13    29.85%
 10     0.65   0.08   0.13   0.15   0.03    30.52%
 20     0.26   0.04   0.03   0.05   0.01    34.80%   ← peak val
 30     0.17   0.03   0.01   0.02   0.00    31.98%
 40     0.13   0.02   0.01   0.01   0.00    28.51%
```

Loss components are quantitatively smaller than Stage 6 (5.88 vs 0.51 box at
ep 1; etc) because focal loss is normalized by `(1 - p_t)^γ` which is
typically < 1. This is expected — the absolute numbers aren't comparable
across loss functions; what matters is gradient direction.

Two observations:

1. **Plateau at the same level as Stage 6**, just reached differently. Stage 6
   peaked at 33.47% at epoch 8 then oscillated; Stage 8 peaked at 34.80% at
   epoch 20 then oscillated. The *peak* is slightly higher, but the
   *plateau band* is essentially identical (~28-33%).
2. **Best-by-val captured the lucky epoch 20 spike**. On full eval (2k
   sample), best weights produce 32.90% — within 1 pp of Stage 6's 33.89%.
   The peak was sampling noise on the per-epoch 500-frame eval.

### 2.2 Headline numbers

```
Stage 2 (sliding window):                4.30% mAP
Stage 3 (from-scratch YOLO):            21.12% mAP
Stage 4 (frozen pretrained):             5.94% mAP
Stage 5 (fine-tuned pretrained):        30.62% mAP
Stage 6 (+ light augmentation):         33.89% mAP
Stage 8 (+ focal loss):                 32.90% mAP   ← −0.99 pp vs Stage 6
```

---

## 3. Per-class results (best-weights, 2000-frame val sample)

```
Class               Stage 6 AP   Stage 8 AP    Δ        Direction
────────────────────────────────────────────────────────────────
LostSoul            34.29%       44.29%       +10.0    ← biggest gain
ShotgunGuy          37.49%       40.37%       +2.9
Demon               37.98%       41.17%       +3.2
HellKnight          44.12%       45.55%       +1.4
SpiderMastermind    26.14%       27.27%       +1.1
BaronOfHell         43.35%       44.26%       +0.9
Zombieman           14.33%       14.73%       +0.4
Arachnotron         31.68%       32.06%       +0.4
Revenant            35.18%       35.29%       +0.1
ChaingunGuy         47.74%       47.70%        0.0
Spectre              0.45%        0.00%       −0.45
PainElemental       27.79%       24.19%       −3.6
DoomImp             38.32%       32.99%       −5.3    ← significant regression
Archvile            51.94%       45.45%       −6.5
Cyberdemon          30.75%       24.14%       −6.6
Fatso               32.64%       26.04%       −6.6
Cacodemon           41.88%       33.87%       −8.0    ← biggest regression
────────────────────────────────────────────────────────────────
mAP                 33.89%       32.90%       −0.99
```

### 3.1 The pattern is "reshaping, not lifting"

Stage 8 made *some* classes substantially better and *some* substantially
worse, netting to a small negative overall. The classes that gained
(LostSoul, ShotgunGuy, Demon) are *medium-frequency, visually distinctive*
ones. The classes that regressed (Cacodemon, Archvile, Cyberdemon, Fatso,
DoomImp) are a heterogeneous mix that doesn't fit a clean pattern.

Notably absent from the "biggest gainers" list: the rare classes focal loss
was *supposed* to help most. SpiderMastermind +1.1, PainElemental −3.6,
Cyberdemon −6.6. Focal loss did not deliver its predicted impact on the
imbalance problem.

---

## 4. Discussion

### 4.1 Why focal loss didn't help (a hypothesis)

Focal loss is designed for *extreme* class imbalance, typically background-
vs-foreground in dense detectors where the imbalance is 1000:1 or worse.
Our positive-class imbalance is much milder: DoomImp:SpiderMastermind ≈
43:1, and most non-rare classes are within ~5x of each other.

At γ=2.0, focal loss aggressively down-weights easy examples — but our
dataset doesn't have huge numbers of easy examples for the model to ignore.
Most of our training examples are *somewhat hard* (the model can't perfectly
fit even DoomImp at this dataset size). Down-weighting them simply reduces
gradient signal overall without redirecting it to the hard tail.

The Stage 8 result is consistent with the empirical observation that focal
loss helps *most* on large datasets with extreme imbalance (COCO, with
80 classes and many easy negatives) and helps *least* on small datasets
with mild imbalance (this project).

### 4.2 Why some classes specifically regressed

Cacodemon (−8.0), Archvile (−6.5), and Cyberdemon (−6.6) regressed
substantially. These are large, visually distinctive enemies where the
model in Stage 6 had presumably learned to confidently detect them at high
class probability. With focal CE, high-confidence-correct predictions get
~0 gradient (the `(1-p_t)^γ` factor is near 0 when `p_t` is near 1),
meaning the model stops reinforcing what it already does well. Over many
epochs of training, the precision on these classes drifts as the model
focuses gradient elsewhere.

This is a known failure mode of focal loss on small datasets: it
*intentionally* under-trains the easy cases, which is great when you have
millions of easy cases drowning out gradient. With only thousands of
medium-difficulty cases, the under-training hurts.

### 4.3 Why LostSoul +10 pp

The biggest single gain. LostSoul has distinctive features (flaming floating
skull silhouette) but is small on screen — often partially correct
detections with moderate confidence. Focal loss's *up-weighting* of
moderate-confidence examples gave LostSoul detection more gradient, lifting
its AP by 10 pp.

This is the focal loss success case in our dataset: a class that's neither
trivial nor impossible, where the model was producing many ~0.4-0.6
confidence detections that focal loss helped sharpen.

### 4.4 Spectre stays at 0

Spectre's score went from 0.45% to 0.00%. The Stage 7 finding (Spectre is
detectable only on memorized training frames; per-map generalization is
~zero) still holds. No loss reshaping changes the structural-vision
limitation. Confirmed across five stages now.

### 4.5 What this tells us about the project's bottlenecks

Stages 5-8 have established a plateau around 30-34% per-map val mAP across
multiple interventions:
- Fine-tuned backbone alone: 30.62%
- + Augmentation: 33.89%
- + Focal loss: 32.90%

The interventions that *didn't* break the plateau:
- Augmentation (light) added 3 pp
- Focal loss (no help)

The remaining bottlenecks therefore are *not* primarily addressable by loss
tweaks. The realistic options for substantially higher mAP would be:

1. **More data** — capture another 10-30k frames. Most direct fix.
2. **Multi-scale prediction** — change to 3-grid output (13/26/52) to
   improve small-object detection (Zombieman, distant ShotgunGuy). Requires
   architecture redesign.
3. **Larger backbone** — ResNet50 or EfficientNet-B0 instead of ResNet18.
   More feature capacity. Requires ~4× compute.
4. **Mosaic + mixup augmentation** — far more aggressive than the Stage 6
   light augmentation. Predicted +3-8 pp.

For this project, the right move is to **accept the plateau as the
attainable ceiling for this architecture/data combination** and move to
Stage 9 (test set evaluation) using Stage 6's weights — which produced
the higher 33.89%.

### 4.6 What goes in the writeup

Stage 8 is a clean Stage 4-style negative result: a *predicted* improvement
that didn't materialize, with a *clean explanation* for why. The writeup
should:

1. Report Stage 8 honestly as "focal loss did not improve mAP on this
   dataset."
2. Explain the dataset-property reason (mild class imbalance, small total
   data → focal's mechanism doesn't apply cleanly).
3. Highlight the per-class reshaping pattern: LostSoul +10, others mixed.
4. Use this as motivation for Stage 9's choice of model: **Stage 6 weights
   for test evaluation, not Stage 8's**.

A negative result clearly explained is more valuable than a small positive
result that's hard to attribute.

---

## 5. Implications for Stage 9

**Stage 9 will use Stage 6's weights, not Stage 8's.** Stage 6 produced
33.89% val mAP; Stage 8 produced 32.90%. The test-set evaluation should use
the model that performed best on val.

This is consistent with standard practice: when comparing models, the one
with the best validation metric is the one you ship/report.

Stage 9 itself is the final test-set evaluation. Expected outcome based on
all stages so far: **test mAP ≈ 28–35%**, similar to or slightly below
val. The honest project headline number.

---

## 6. Reproducibility

| Artifact | Location |
|---|---|
| Script | `stage8.py` (~210 lines; imports from stage3.py + stage5.py + stage6.py) |
| Plan | `stages/stage_008_plan.md` |
| Notebook | `stage8_colab.ipynb` |
| Best weights | `stage8_best.pt` (saved but **not** used for Stage 9) |
| Random seed | 42 |

Runtime: ~2h 20m on Colab T4.

---

## 7. Conclusions

Replacing standard cross-entropy + BCE losses with focal-loss equivalents
produced a 32.90% val mAP — essentially flat against Stage 6's 33.89%, with
a slight (−1.0 pp) regression. Per-class results show *reshaping* rather
than lifting: LostSoul gained +10 pp; Cacodemon, Archvile, Cyberdemon, and
Fatso lost 6-8 pp each. Net effect was neutral-to-slightly-negative.

The likely root cause is dataset-property mismatch: focal loss is designed
for extreme imbalance regimes (COCO-style, 1000:1 background ratios) and
provides less benefit on this project's modest 43:1 class imbalance and
small absolute dataset size. The mechanism of "down-weight easy examples"
backfires when the model doesn't have a lot of easy examples to begin with.

Combined with the plateau observed across Stages 5-8 (30-34% mAP regardless
of intervention), Stage 8 confirms that the project's remaining bottleneck
is not loss design but the combination of small dataset, single-scale
prediction, and limited backbone capacity. Significant further gains would
require qualitative changes (more data, multi-scale architecture, larger
backbone, or stronger augmentation like mosaic) — out of scope for the
remaining stages.

Stage 9 will use Stage 6's best weights (33.89% val) for the final test-set
evaluation.
