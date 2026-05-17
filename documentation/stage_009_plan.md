# Stage 9 — Final Test-Set Evaluation: Plan

## Goal

Run the project's *honest* final evaluation: take the best model (Stage 6
weights, val mAP 33.89%) and evaluate it on the held-out test set (MAP26–30 +
MAP32), which has never been touched during training, hyperparameter tuning,
model selection, or any other decision-making.

The number produced by Stage 9 is the project's headline result. It goes in
the writeup. It does not get revised.

---

## 1. Why we're doing this

The discipline of train/val/test separation has been maintained throughout
Stages 1–8:
- **Train** (MAP01–15, MAP31): used to fit model weights.
- **Val** (MAP16–25): used to choose hyperparameters, compare model variants,
  pick the best checkpoint, decide when to stop training.
- **Test** (MAP26–30, MAP32): captured ahead of Stage 9. Never loaded or
  evaluated until now.

Every val mAP reported in Stages 3–8 was used to make *some* decision (which
augmentation, which loss, which weights to save). That makes those numbers
*biased* — they're optimistic estimates of generalization performance
because we selected for what looked good on val.

Test is the *unbiased* number. It tells us what the model actually does on
truly unseen data, with no selection bias.

If test mAP is close to val mAP (~33%), we've reported honestly throughout.
If test mAP is dramatically lower (~10%), we've been overfitting to val
without realizing.

---

## 2. Model selection

The winning model from cross-stage val comparison:

| Stage | Val mAP (2k sample) |
|---|---|
| Stage 5 (fine-tuned baseline) | 30.62% |
| **Stage 6 (+ light augmentation)** | **33.89%** ← best on val |
| Stage 8 (+ focal loss) | 32.90% |

**Stage 6's `stage6_best.pt` is the model used for Stage 9.** Standard
practice: ship the model with the highest validation metric.

Stage 8's weights are *not* used. Stage 8's writeup explicitly concluded
that focal loss did not help on this dataset.

---

## 3. Experiment design

| Item | Value |
|---|---|
| Model | Stage 6's `FineTunedYOLO` |
| Weights | `stage6_best.pt` (~45 MB) |
| Test maps | MAP26, MAP27, MAP28, MAP29, MAP30, MAP32 |
| Sample size | All available test frames (or 2000 if more) |
| Device | CPU (inference only) |
| Augmentation | None (test set is never augmented) |
| Random seed | 42 (for deterministic sampling if subsampling) |
| Runtime | ~5-10 min on CPU |

The evaluation pipeline is *identical* to the per-epoch val evaluation used
in Stages 3-8 (`evaluate_map`). Only the data changes.

---

## 4. What we report

For the project writeup:

1. **Overall test mAP @ IoU=0.5** — the single headline number.
2. **Per-class AP @ IoU=0.5** — 17-row table showing per-class performance.
3. **Val-vs-test comparison** — sanity check; large divergence would suggest
   val was over-fitted.
4. **Optional**: per-test-map breakdown (mAP per map separately), useful to
   note variance across maps.

---

## 5. Predictions

Based on the consistent ~33% val mAP across Stages 6-8, expected test mAP:

| Outcome | Likelihood | Interpretation |
|---|---|---|
| **28-35% test mAP** | most likely | Val and test agree; methodology was sound; no surprises |
| 22-28% | possible | Slight drop suggests val maps were slightly easier than test maps |
| <22% | unlikely | Val overfitting OR test maps systematically harder |
| >35% | unlikely | Lucky test sample OR test maps systematically easier |

Per-class predictions follow the same pattern as Stage 6's val numbers, with
each per-class AP likely within ±5 pp of its Stage 6 val value.

**Spectre is predicted to score 0–5% on test**, confirming the structural-
difficulty finding from Stage 7 (which already showed Spectre is essentially
undetectable on per-map evaluation).

---

## 6. Why we DON'T re-tune anything after seeing this

The single most important rule: **test is touched exactly once.** No going
back to:
- Tweak loss weights "because test showed something."
- Try a different augmentation policy "because test exposed an issue."
- Pick a different epoch's checkpoint "because that scored higher on test."

Any of these would re-introduce the very bias the test/val split was
designed to prevent. The test number is final. We document it, discuss what
the number means, and move on to the project conclusions.

---

## 7. Reproducibility

| Artifact | Location |
|---|---|
| Script | `stage9.py` |
| Required input | `stage6_best.pt`, captured test data in `data/MAP{26,27,28,29,30,32}/` |
| Random seed | 42 |
| Runtime | ~5-10 min on CPU |
