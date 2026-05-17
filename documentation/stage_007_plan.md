# Stage 7 — Per-Map vs Random Split Contrast: Plan

## Goal

Quantify how much *random-frame splitting* would have inflated this project's
reported val mAP, by re-evaluating Stage 6's saved weights on two
contrasting val sets — the honest per-map split and a "leaky" random sample
that mixes train and val frames.

This stage is *not* a model training run. It's a methodology demonstration
that takes ~5 minutes on CPU.

---

## 1. The argument being demonstrated

Every Stage 3–6 result has been reported on a per-map val split: train maps
(MAP01–15, MAP31) are entirely disjoint from val maps (MAP16–25). No frame
from a train map appears in val.

The alternative is **random splitting**: throw all captured frames into one
pool, shuffle, and take 20% as val. This is the *default* approach in many
ML tutorials and is what most ML beginners do.

It's almost always wrong for video-derived datasets. The reason:

> Consecutive frames from the same map are nearly identical — same textures,
> same lighting, same enemy positions appearing repeatedly across many frames.
> If a frame from MAP05 is in train and another from MAP05 (perhaps just 5
> frames later in capture) is in val, the model has *effectively seen* the
> val frame during training. Val accuracy reflects memorization, not
> generalization.

Per-map splitting forces the model to generalize to *entirely new
environments* (different textures, different sectors, different enemy
compositions). Reported per-map val mAP is an honest estimate of what the
model would do on unseen maps. Random-split val mAP is a substantially
inflated estimate.

Stage 7 measures the gap by experiment.

---

## 2. The experiment

Take Stage 6's saved weights (`stage6_best.pt`). The model was trained on
per-map train (MAP01–15, MAP31). Evaluate it on two different val sets:

**Scenario A — Honest per-map val**
- Sample 500 random frames from MAP16–25 (the per-map val maps).
- Model has *never seen* these frames during training.
- Result: should match Stage 6's reported 33.89% mAP (within noise).

**Scenario B — Leaky random val**
- Sample 500 random frames from the union of train+val maps (MAP01–25 +
  MAP31).
- ~48% of these frames are from train maps the model *has* seen during
  training.
- Model gets near-perfect mAP on those (it memorized them).
- Result: should be substantially higher than 33.89% due to the train
  fraction.

**Inflation** = Scenario B mAP − Scenario A mAP. The size of this gap is the
quantified data-leakage cost.

### 2.1 What we expect

Stage 6's training accuracy was very high (loss 0.33 by epoch 40 → near-
perfect fit on train). On train frames the model has seen, mAP is probably
80–95%. Expected:

```
Scenario A (honest):   ~33%        (matches Stage 6's reported number)
Scenario B (leaky):    ~60–70%     (~half train at ~90%, half val at ~33%)
Inflation:             ~30 pp
```

Whatever the actual numbers, the inflation makes the per-map vs random-split
distinction concrete: "if we had used random splitting, we'd have reported
~65% instead of 34%. Almost double."

---

## 3. Limitations

This is a *cheap* version of the comparison. The more rigorous experiment
would be to train a *separate model* on a randomly-split train set, then
evaluate on its own random val. That fully simulates "what if we had random-
split from the start" and would take another ~2-hour Colab run.

The Stage 7 cheap version captures the essential lesson — "random splitting
inflates reported numbers when consecutive frames are similar" — without
the cost of another training run.

If the cheap demo shows a >20 pp inflation, that's a sufficiently strong
result to motivate the per-map discipline. The expensive version could be
revisited as a Stage 8+ refinement if needed.

---

## 4. Implementation

`stage7.py` takes Stage 6's weights and runs two `evaluate_map` calls with
different `allowed_maps` arguments. Uses CPU (no GPU needed — inference only,
500 frames takes ~2-5 minutes).

Outputs:
- mAP for each scenario
- Per-class AP for each scenario, side-by-side
- The numerical inflation
- Estimated fraction of leaky val that came from train maps

---

## 5. Reproducibility

| Artifact | Location |
|---|---|
| Script | `stage7.py` |
| Required input | `stage6_best.pt` (download from Colab) |
| Random seed | 42 |
| Runtime | ~5 min on CPU |
