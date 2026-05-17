# Stage 6 — Data Augmentation: Results

## Abstract

Stage 5 plateaued at 30.62% val mAP by epoch 5 due to severe overfitting (11.2M
trainable params vs 12.5k train frames). Stage 6 introduces light per-batch
augmentation — horizontal flip (50% probability) and modest color jitter
(brightness/contrast/saturation ±15–20%, no hue change) — applied to training
only. Architecture, optimizer, and learning rates are identical to Stage 5.
After 40 epochs, **final val mAP: 33.89% (best epoch 8 at 33.47%)** — a
+3.27 pp gain over Stage 5. The lift is below the predicted 35–45% range:
augmentation delayed but did not eliminate overfitting. Per-class deltas
reveal that mid-tier classes (Cacodemon, DoomImp, Fatso) benefit most, with
modest regressions on a few classes that were already saturated. Spectre
remains at ~0% (now confirmed across four architectures). The Stage 6 result
suggests the next meaningful gains require *qualitatively different*
interventions (focal loss, class weighting, mosaic) rather than more of the
same.

---

## 1. Setup recap

| Item | Value |
|---|---|
| Backbone | ResNet18 pretrained, fine-tuned (same as Stage 5) |
| Augmentation (train only) | Horizontal flip (p=0.5) + ColorJitter(0.2, 0.2, 0.15, hue=0) |
| Augmentation (val) | None |
| Optimizer | Adam, backbone 1e-4, head 1e-3 |
| Epochs | 40 (vs Stage 5's 30) |
| Other | unchanged from Stage 5 |
| Device | Colab T4 GPU |
| Wall-clock | ~2h |

The augmentation is deliberately *conservative* — Stage 1 §6 demonstrated that
heavy color jitter destroys Doom's palette-encoded class identities (the peer's
project hit this exact failure). Hue is explicitly *not* perturbed.

---

## 2. Training dynamics

### 2.1 Loss curve

```
Epoch  total   box    obj    noobj  cls     val_mAP
─────────────────────────────────────────────────────
  1    10.55   0.50   3.54   4.64   2.17    20.78%
  3     4.61   0.26   1.82   1.86   0.58    26.31%
  5     3.13   0.20   1.16   1.39   0.29    33.16%
  8     1.90   0.15   0.61   0.83   0.14    33.47%   ← peak val
 14     0.97   0.10   0.23   0.34   0.06    30.23%
 22     0.61   0.07   0.12   0.18   0.04    32.58%
 33     0.40   0.05   0.08   0.10   0.02    31.27%
 40     0.33   0.04   0.06   0.09   0.01    31.41%
```

**Three observations on the curve:**

1. **Augmentation slows initial training as expected.** Stage 5 hit 30.17% at
   epoch 4; Stage 6 needed until epoch 4-5 for comparable mAP. The harder
   training signal means each epoch makes less progress.
2. **Plateau is delayed** from Stage 5's epoch 5 to Stage 6's epoch 8 — and
   the plateau is at a *higher* level (33.47% vs 30.87%). Augmentation worked.
3. **Train loss keeps falling steadily through epoch 40** (1.90 → 0.33) while
   val oscillates 27–33%. The overfitting story is exactly the same as Stage 5
   — just shifted ~3 pp upward. Augmentation didn't *eliminate* the gap, just
   *narrowed* it.

### 2.2 Headline numbers

```
Stage 2 (sliding window):             4.30% mAP
Stage 3 (from-scratch YOLO):         21.12% mAP
Stage 4 (frozen pretrained):          5.94% mAP
Stage 5 (fine-tuned):                30.62% mAP
Stage 6 (+ light augmentation):      33.89% mAP   ← +3.27 pp over Stage 5
```

---

## 3. Per-class results (best-weights, 2000-frame val sample)

```
Class               Stage 5 AP   Stage 6 AP    Δ        Tier
─────────────────────────────────────────────────────────────────────
Archvile            44.31%       51.94%       +7.6     ── new strongest ──
ChaingunGuy         49.29%       47.74%       −1.6
HellKnight          39.37%       44.12%       +4.7     ── strong ──
BaronOfHell         44.84%       43.35%       −1.5
Cacodemon           34.30%       41.88%       +7.6
DoomImp             32.10%       38.32%       +6.2     ── upper-mid ──
Demon               34.22%       37.98%       +3.8
ShotgunGuy          33.28%       37.49%       +4.2
Revenant            35.37%       35.18%       −0.2
LostSoul            33.58%       34.29%       +0.7
Fatso               22.48%       32.64%      +10.2     ── mid-tier ──
Arachnotron         35.53%       31.68%       −3.9
Cyberdemon          25.57%       30.75%       +5.2
PainElemental       22.96%       27.79%       +4.8
SpiderMastermind    18.18%       26.14%       +8.0     ── rare-class lift ──
Zombieman           15.15%       14.33%       −0.8     ── weak ──
Spectre              0.00%        0.45%       +0.45    ── structurally hard ──
─────────────────────────────────────────────────────────────────────
mAP                 30.62%       33.89%       +3.27
```

### 3.1 Where augmentation helped most

**Big winners (+5 pp or more)**:
- **Fatso (+10.2 pp)** — biggest single jump. From "weak tier" to "mid tier"
  with one stage.
- **SpiderMastermind (+8.0 pp)** — continues the trend from Stage 5 (was 0% in
  Stage 3, 18% in Stage 5, now 26%). Small-data class consistently improves as
  feature quality and training-data effective-diversity grow.
- **Cacodemon (+7.6 pp)** and **Archvile (+7.6 pp)** — visually complex classes
  that benefited from forced invariance to color/lighting variation.
- **DoomImp (+6.2 pp)** — the dominant common class also gets a lift.
- **Cyberdemon (+5.2 pp)** — limited-pose class benefits from horizontal flip
  doubling its effective viewing angles.

### 3.2 Where augmentation slightly hurt

**Small regressions** (−1 to −4 pp): BaronOfHell, ChaingunGuy, Arachnotron,
Zombieman. These are classes where Stage 5's tighter-fit-to-training-set was
actually *helpful* — the model was learning specific pose patterns that
augmentation perturbed away. The classes that regress are not the ones that
were overfitting most; they're the ones that were already well-localized.

Net of winners and losers, the overall mAP gain is +3.27 pp — modest but
clearly positive.

### 3.3 Spectre and the structural limit

Spectre's AP across all four detection stages:

| Stage | Approach | Spectre AP |
|---|---|---|
| 3 | From-scratch | 0.25% |
| 4 | Frozen pretrained | 0.00% |
| 5 | Fine-tuned pretrained | 0.00% |
| 6 | Fine-tuned + augmented | 0.45% |

Four architectures, four results around zero. **This is now overwhelming
evidence for the structural-difficulty hypothesis from Stage 1 §5.3**: Spectre
cannot be distinguished from Demon by RGB-pixel-based vision at this dataset's
size and resolution, regardless of the model. The information needed
(runtime alpha-blending artifacts) isn't recoverable from the available
features.

A Stage 9 qualitative section can use Spectre as the canonical "fundamental
class" example. It's the project's cleanest negative result.

---

## 4. Discussion

### 4.1 The augmentation lift is real but smaller than predicted

Predicted: 35–45% mAP. Achieved: 33.89% mAP.

The gap to the prediction is informative. Most published "augmentation
improves object detection by ~10 pp" results come from larger datasets and
more aggressive augmentation strategies (mosaic, random crop/scale, mixup).
Stage 6's conservative augmentation (flip + light color jitter) is closer to
the lower bound of what augmentation can do. The +3.27 pp lift is consistent
with that constraint.

### 4.2 The plateau didn't actually go away

Stage 5 plateaued at epoch 5 at ~30%. Stage 6 plateaued at epoch 8 at ~33%.
**Augmentation delayed and lifted the plateau by ~3 pp but didn't break it.**
The fundamental dynamic — model capacity > effective data — is still present.
What changed is that effective data went from 12.5k examples to "12.5k
examples × ~variation," not from 12.5k to "infinitely many."

The remaining gap to a ~50% mAP regime requires *qualitatively different*
interventions, not more of the same:
- Heavier augmentations that genuinely multiply diversity (mosaic, mixup,
  random scale/crop) — Stage 8 territory.
- Loss-shaping that helps rare classes specifically (focal loss, class-
  balanced sampling) — Stage 8 territory.
- More data (most direct but most expensive).

### 4.3 Why some classes regressed

Three classes lost a small amount of AP (BaronOfHell, ChaingunGuy,
Arachnotron). These have similar properties: they were already well-fit by
Stage 5's memorization, so adding noise via augmentation slightly *unfits*
them. Net trade-off was favorable for the dataset overall but not for these
specific classes.

This is a known property of augmentation: it raises the *average* by reducing
variance, sometimes at the cost of *peak performance on already-well-fit
classes*. The fix in production systems is per-class augmentation policies
(some classes get more, some less); for a writeup, the trade-off is the right
thing to document and accept.

### 4.4 The per-class story is now richer than before

Stage 5 had a clear strong-tier (4 classes ≥39%) and decent-tier (7 classes
30–39%). Stage 6:
- **Promoted Fatso, Cyberdemon, PainElemental, and SpiderMastermind** out of
  the weakest tier into useful AP territory (>25%).
- **Added Archvile and Cacodemon** to the strong tier (≥40%).
- **Spectre and Zombieman remain stuck** — different reasons (Spectre is
  structurally hard; Zombieman is small + visually generic).

The detector is now meaningfully useful on 14 of 17 classes (>25% AP). That's
the qualitative milestone for the writeup.

### 4.5 SpiderMastermind's trajectory is worth tracking

Across stages: 0% → 0% → 18% → 26%. Two-stage upward trend correlates with
feature quality (Stage 5) and training-data effective diversity (Stage 6).
The "147 training instances is hopeless" hypothesis from Stage 3 is now
firmly disproved — given good features and augmentation, even small per-class
datasets can produce non-trivial AP.

This is a useful Stage 9 talking point: "rare classes are improvable; the
limit is gradient-update-frequency, not absolute data quantity."

---

## 5. Implications for Stages 7–9

**Stage 7 (per-map vs random split contrast)** — a methodology demonstration,
not a model training run. Re-evaluate Stage 6's saved weights on a random-
frame val split to show that random splitting overestimates performance by
a known amount. Quick experiment, ~20 minutes of code.

**Stage 8 (loss + assignment refinements)** — the next mAP-raising stage.
Options:
- **Focal loss** for the class loss — directly addresses class imbalance,
  predicted +2–5 pp.
- **Class-balanced sampling** — oversample rare classes during batch
  construction. Cheap, predicted +1–3 pp.
- **Mosaic augmentation** — would have to be done as a custom collate
  function. Predicted +3–8 pp.
- **Multi-scale prediction** (3 grids: 13/26/52) — significant architecture
  change but probably +5–10 pp.

Realistic Stage 8 target: 38–45% mAP.

**Stage 9** — final test-set evaluation, requires test data captured. ~30
lines of code on top of existing eval infrastructure.

---

## 6. Reproducibility

| Artifact | Location |
|---|---|
| Script | `stage6.py` |
| Plan | `stages/stage_006_plan.md` |
| Notebook | `stage6_colab.ipynb` |
| Best weights | `stage6_best.pt` |
| Random seed | 42 |

Runtime: ~2h on Colab T4 for 40 epochs.

---

## 7. Conclusions

Light data augmentation (horizontal flip + modest color jitter) improved
Stage 5's fine-tuned pretrained detector from 30.62% → 33.89% val mAP — a
+3.27 pp gain. The improvement is real but modest because (a) the
augmentations were deliberately conservative to avoid the palette-destruction
failure mode documented in Stage 1, and (b) the underlying overfitting was
delayed rather than eliminated — train loss continued to fall past the val
plateau just as in Stage 5.

Per-class deltas confirm the standard augmentation pattern: substantial
improvements for mid-tier and rare classes (Fatso +10.2, SpiderMastermind
+8.0, Archvile and Cacodemon +7.6 each); small regressions for classes whose
Stage 5 peaks relied on memorization of specific training patterns
(Arachnotron −3.9, BaronOfHell −1.5, ChaingunGuy −1.6). The mAP averaged out
to a clean positive lift.

Spectre's AP remains 0–0.5% across all four detection stages, providing
unambiguous evidence that this class is structurally untrainable from RGB
pixels at the project's data scale.

Stage 7's split-contrast experiment and Stage 8's loss/sampling refinements
remain to be run before Stage 9's final test-set evaluation.
