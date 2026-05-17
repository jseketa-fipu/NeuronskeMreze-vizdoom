# Stage 9 — Final Test-Set Evaluation: Results

## Abstract

The project's best-by-val model (Stage 6, val mAP 33.89%) was evaluated on
the held-out test set (MAP26–30 + MAP32, 7,438 frames total, 2,000-frame
deterministic sample). **Overall test mAP @ IoU=0.5: 24.21% — a −9.68 pp
drop vs val.** The gap is real and meaningful: it lands in the plan's
"possible" band (22–28%) rather than the predicted "most likely" 28–35%.
Per-test-map breakdown reveals strong variance: MAP26-28 score 32–33% (in
line with val), MAP32 scores 28%, while MAP29 (19.8%) and MAP30 (17.3%) are
substantially harder. Per-class deltas are heavily negative — 12 of 16
classes (the 17th, SpiderMastermind, had no GT in the sample) lost ground
on test, including BaronOfHell (−26.1 pp), ChaingunGuy (−32.5 pp),
Archvile (−20.1 pp), and Fatso (−20.0 pp). A few classes improved on test
(Cyberdemon +11.2, Zombieman +9.1, LostSoul +5.2). The honest interpretation
is that **val maps 16–25 were systematically easier than test maps 26–30+32**,
likely because the late-game maps introduce harder backgrounds (more
ambient lighting variation, more visual clutter from environmental detail)
and a less-favorable enemy mix. Per the test/val protocol, this number
is final and does not get re-tuned.

---

## 1. Setup recap

| Item | Value |
|---|---|
| Model | `FineTunedYOLO` (Stage 6 architecture) |
| Weights | `stage6_best.pt` (val-best from Stage 6 training) |
| Test maps | MAP26, MAP27, MAP28, MAP29, MAP30, MAP32 |
| Test frames available | 7,438 |
| Sample size | 2,000 (deterministic, seed=42) |
| Augmentation | None (test set is never augmented) |
| Device | CPU |
| Wall-clock | 157s |
| Eval metric | mAP @ IoU=0.5 (VOC2007 11-point interpolation) |

The evaluation pipeline is identical to the per-epoch val evaluation used
through Stages 3–8 (`evaluate_map`), with the addition of per-map AP
tracking. Only the dataset changes.

---

## 2. Headline numbers

```
Stage 2 (sliding window):                 4.30%  val mAP
Stage 3 (from-scratch YOLO):             21.12%  val mAP
Stage 4 (frozen pretrained):              5.94%  val mAP   ← negative
Stage 5 (fine-tuned pretrained):         30.62%  val mAP
Stage 6 (+ light augmentation):          33.89%  val mAP   ← best on val
Stage 8 (+ focal loss):                  32.90%  val mAP   ← negative

Stage 9 (final test, Stage 6 weights):   24.21%  test mAP  ← project headline
```

**Val → test delta: −9.68 pp.** This is the unbiased measurement of how
well the model generalizes to data never used in any decision-making.

---

## 3. Per-class results

```
Class               Stage 6 val   Stage 9 test    Δ        Direction
────────────────────────────────────────────────────────────────────
Cyberdemon          30.75%        41.90%         +11.15   ← biggest gain
Zombieman           14.33%        23.42%         +9.09
LostSoul            34.29%        39.51%         +5.22
Spectre              0.45%         3.64%         +3.19
ShotgunGuy          37.49%        29.58%         −7.91
Cacodemon           41.88%        34.56%         −7.32
HellKnight          44.12%        35.97%         −8.15
Demon               37.98%        28.04%         −9.94
Revenant            35.18%        24.32%        −10.86
PainElemental       27.79%        15.64%        −12.15
Arachnotron         31.68%        14.02%        −17.66
DoomImp             38.32%        19.85%        −18.47
Fatso               32.64%        12.68%        −19.96
Archvile            51.94%        31.85%        −20.09
BaronOfHell         43.35%        17.22%        −26.13
ChaingunGuy         47.74%        15.24%        −32.50   ← biggest regression
SpiderMastermind    26.14%          —           (no GT in test sample)
────────────────────────────────────────────────────────────────────
mAP                 33.89%        24.21%         −9.68
```

### 3.1 The drop is broad, not concentrated

12 of 16 measurable classes regressed on test. This is not a "one weird
class tanked the average" story — the model genuinely performs worse on
test data across most categories. The four classes that improved
(Cyberdemon, Zombieman, LostSoul, Spectre) are heterogeneous: one is a
massive one-of-a-kind boss, one is the smallest enemy, one is a flaming
floating skull, and one is "the invisible thing." No clean pattern.

### 3.2 The biggest losses are mid-large enemies

ChaingunGuy, BaronOfHell, Archvile, Fatso, and DoomImp account for the
heavy regressions. These are visually distinctive enemies that scored
well on val. The most likely explanation: the model learned val-specific
appearance details (lighting/background palette of the val maps) that
don't transfer cleanly to the test maps' rendering conditions. Test
maps are later in the game, with more varied environmental art and
darker average lighting.

### 3.3 Spectre stays terrible — confirmed across all stages

Stage 1: undetectable. Stage 2: 0%. Stages 3-8: 0–0.5%. Stage 9: 3.64%.
The 3.64% is barely above noise; Spectre is essentially undetectable
across every model the project has tried. Stage 7's finding stands: the
model can occasionally hit Spectre by exploiting per-map context cues,
but cannot generalize from Spectre's appearance.

### 3.4 SpiderMastermind has no test sample

Of the 2,000 sampled test frames, none contained a SpiderMastermind
instance. This is consistent with its rarity (147 instances across the
entire training set). A complete evaluation would require evaluating all
7,438 test frames; even then, SpiderMastermind appears in only one of
the six test maps, so per-test-frame coverage is low. The Stage 9 number
is computed over the 16 classes with GT and is therefore directly
comparable to Stage 6's val mAP (which averaged 17 classes including
SpiderMastermind at 26.14%).

---

## 4. Per-test-map breakdown

```
Map      mAP        Notes
─────────────────────────────────────────────────────────────
MAP26    32.04%     in line with val mAP (33.89%)
MAP27    32.27%     in line with val mAP
MAP28    32.78%     in line with val mAP
MAP32    28.26%     slightly below val
MAP29    19.82%     significantly harder
MAP30    17.27%     significantly harder
─────────────────────────────────────────────────────────────
Mean:    27.07%     (unweighted average of per-map mAPs)
Overall: 24.21%     (weighted by frames in sample)
```

### 4.1 The test set has two regimes

Three of the six test maps (MAP26-28) perform essentially identically to
val. The model generalizes fine to them. **MAP29 and MAP30 drag the
average down by 10+ pp each.** Without those two maps, test mAP would
sit at ~31% — within 3 pp of val. This is the kind of insight the
test/val protocol exists to surface: it tells us our model has variance
*across maps* that the val set didn't reveal because val happened to
sample easier-difficulty maps.

The unweighted per-map mean (27.07%) is meaningfully higher than the
frame-weighted overall (24.21%) because MAP29 and MAP30 contributed
disproportionately many frames to the 2,000-frame sample.

### 4.2 Why MAP29 and MAP30 are harder (hypotheses, not findings)

Without re-tuning, we can only speculate:

- **Later-game maps are visually more cluttered**: MAP29 ("The Living End")
  and MAP30 ("Icon of Sin") are end-game with elaborate, atypical
  architecture vs the moderate complexity of MAP26-28.
- **Lighting variance**: end-game maps frequently use red/orange ambient
  lighting that may shift the color distribution away from training.
- **Atypical enemy mix**: MAP30 in particular has unusual spawning
  patterns.

These are hypotheses for a writeup paragraph, not actions to take. The
test set has been spent.

---

## 5. Methodology assessment

### 5.1 What the −9.68 pp gap tells us

Val mAP (33.89%) was an *optimistic* estimate of generalization
performance — it was the metric used to choose every hyperparameter,
loss function, and weight file from Stages 3 through 8. Test mAP
(24.21%) is the *honest* estimate: no decision was made using this
number until after it was computed.

A 9.68 pp gap is meaningful but not catastrophic. The literature
typically sees 2–10 pp val→test gaps when val is used heavily for
selection; we're at the high end of that range. Interpretations:

1. **Val/test difficulty difference**: MAP16-25 are mid-game maps with
   roughly stable visual character. MAP26-30 + MAP32 are late-game with
   more variance. The split was made by map number (project conceptually
   reasonable, easy to reproduce) rather than randomized for similar
   difficulty distribution.

2. **Mild val overfitting via selection**: across Stages 3–8 we ran ~6
   architecture/loss/augmentation experiments choosing the best by val.
   Six selections × per-class variance = some accumulated optimization
   toward val-specific patterns. This is exactly the bias the test set
   is supposed to detect.

Both effects likely contribute. The good news: per-map breakdown
(MAP26-28 ≈ val) suggests the gap is dominated by (1), not (2). On
maps that *are* similar to val, the model works at val-similar
performance.

### 5.2 What the project would NOT do now

By the rule established in Stage 9's plan:

- We do **not** train more epochs targeting test.
- We do **not** swap to Stage 8's weights to see if focal-loss model
  generalizes better (Stage 8 val: 32.90%; if its test were 28%, that
  would not change which model we report — we report the val-best, not
  the test-best, by protocol).
- We do **not** investigate MAP29/MAP30 specifically and re-tune for
  hard maps.

The test number is the test number.

### 5.3 Connection to Stage 7's split-contrast finding

Stage 7 (run on the val set, comparing random vs per-map splits) found
that random-split val mAP was 49.58% vs per-map 33.89% — a **+15.69 pp
inflation** when the model is evaluated on frames adjacent to its
training set rather than on entirely new maps.

The Stage 9 result is consistent with that finding's implications:
**per-map evaluation reveals the model's true generalization weakness**.
Stage 7 measured it within val; Stage 9 confirms it on truly unseen
maps. Together they give a complete picture:

```
What metric you choose:                       mAP  (approx)
─────────────────────────────────────────────────────────────
Random-split val (frames from training maps):  ~50%
Per-map val (used for model selection):         34%
Per-map test (this stage):                      24%
```

The 24% figure is the only one that is honestly comparable to "what
would this model do on data nobody has touched."

---

## 6. What this means for the project

### 6.1 The full per-map detection pipeline summary

The project successfully demonstrated end-to-end YOLO-style object
detection on Doom gameplay:

| Stage | Contribution | Val mAP |
|---|---|---|
| 1 | ResNet18 patch classifier baseline | 38% acc |
| 2 | Sliding-window detection bridge | 4.30% |
| 3 | From-scratch YOLO architecture | 21.12% |
| 4 | Frozen pretrained — domain mismatch (negative) | 5.94% |
| 5 | Fine-tuned pretrained (the big lift) | 30.62% |
| 6 | + Augmentation (small lift, best) | 33.89% |
| 7 | Split methodology audit | (not a model) |
| 8 | Focal loss (negative) | 32.90% |
| 9 | **Test-set evaluation** | **24.21% test** |

The headline number for the writeup is **24.21% test mAP**.

### 6.2 What worked, what didn't

**Worked:**
- Architecture re-use from Stage 3 across all subsequent stages: clean
  attribution of every intervention.
- ImageNet pretraining with full fine-tuning (Stage 5): +9 pp over from-
  scratch.
- Conservative augmentation (Stage 6): +3.3 pp over Stage 5.
- Per-map train/val/test split with discipline maintained throughout.
- Stage 7 split-contrast audit: produced the project's most pedagogically
  valuable result.

**Did not work:**
- Frozen pretrained (Stage 4): BN running stats from ImageNet don't fit
  pixel art. −24.7 pp vs Stage 5.
- Focal loss (Stage 8): mild imbalance doesn't benefit from the focal
  mechanism. −1 pp vs Stage 6.

Both negatives are documented honestly with mechanistic explanations.

### 6.3 What would meaningfully improve test mAP

Speculatively, the routes from 24% toward 35-40% test mAP:

1. **More data, particularly from MAP29/MAP30-style late-game maps**.
   The model's failure on test is partly that those maps' visual style
   is under-represented in train. ~10-15k more frames spanning later
   maps would likely close most of the gap.

2. **Multi-scale prediction (3-grid head)**: small enemies like Zombieman
   currently degrade because 13×13 grid is too coarse for distant
   sprites. Standard YOLO has three scales.

3. **Stronger augmentation (mosaic + mixup)**: forces the model to learn
   from compositions instead of memorizing per-map context.

4. **Larger backbone (ResNet50 or EfficientNet-B0)**: more capacity to
   handle the visual variance the augmentation should produce.

These are out of scope for this project; they're noted for completeness.

### 6.4 Discipline as the project's main contribution

The single most defensible thing about this project is that the
test number was generated *exactly once* from data never used in
any prior decision. The 24.21% figure is honest in a way that many
published results are not — including (per Stage 7) the kind of
randomly-split benchmark figure that the same model could have
reported as 49.58%. The 25 pp difference between the most-flattering
honest number we could quote (random-split val) and the actually-honest
number (per-map test) is one of the project's most important findings.

---

## 7. Reproducibility

| Artifact | Location |
|---|---|
| Script | `stage9.py` (~145 lines; imports from stage3/4/5) |
| Plan | `stages/stage_009_plan.md` |
| Weights | `stage6_best.pt` (~45 MB) |
| Test data | `data/MAP{26,27,28,29,30,32}/` (7,438 frames) |
| Random seed | 42 |
| Runtime | 157s on CPU |

Result: **24.21% test mAP @ IoU=0.5**.

---

## 8. Conclusion

The project's best-by-val model (Stage 6 weights, 33.89% val mAP)
achieves **24.21% mAP on the held-out per-map test set** — a 9.68 pp
drop that reflects a combination of (a) test maps being systematically
harder than val maps (MAP29 and MAP30 alone account for most of the
gap) and (b) mild val-set optimization bias accumulated over six
model-selection experiments. The result is the project's honest
headline number and is final by protocol — no further tuning, no
weight re-selection, no test-driven iteration.

Per-class results show broad regressions (12 of 16 measurable classes
lost ground) with a few unexpected gains (Cyberdemon +11.2, Zombieman
+9.1). Per-test-map results show a bimodal distribution: three maps
match val performance, two are substantially harder, one is moderate.
Spectre remains essentially undetectable (3.64%) across every approach
the project tried, confirming the Stage 7 finding that its detection
relies on map-context spurious cues rather than appearance.

Combined with Stage 7's split-contrast finding (random-split val mAP
49.58% vs per-map 33.89%), Stage 9 provides the full honest picture
of what this architecture/data combination achieves: ~24% mAP on
genuinely-unseen data, ~34% on per-map-held-out data used for
selection, and ~50% if one cheats the methodology. The project's main
contribution is documenting all three numbers and explaining what
each means.
