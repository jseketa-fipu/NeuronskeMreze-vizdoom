# Stage 10 — Future Work: Out of Scope

## Status

**This stage is documentation only. No code, no experiments, no
training.** It records concrete directions that would likely move the
Stage 9 test mAP (24.21%) toward 35–45%, but which fall outside the
scope of the current project deliverable. The purpose is to make the
"obvious next steps" explicit so they don't need to be reverse-engineered
from a writeup conclusion, and to be honest about *why* they were not
attempted.

---

## 1. Why these are out of scope

The project's scope was *deliberately bounded* by the course's
expectations and the precedent set by the peer project:

- **Pedagogical arc, not maximum mAP.** The nine-stage structure was
  chosen to demonstrate a clean iteration process with explicit
  positive and negative results, not to push the leaderboard.
- **Single training notebook per stage.** Each intervention was meant
  to be isolated and attributable. Multi-week architecture rewrites
  break that contract.
- **Time budget.** ~2–3 hours per Colab T4 run × 8 training stages
  already consumed the bulk of available compute. The directions below
  each add another full training run.
- **The test set has been spent.** By the test/val protocol, no further
  iteration can use the Stage 9 test set as feedback. Pursuing any of
  these would require capturing a *new* held-out test set (MAP33+) to
  retain methodological honesty.

Each direction below is therefore documented as "what would likely work
if this were a longer project", not "what we should do next sprint."

---

## 2. Direction A — Capture more training data, targeting the failure modes

**Motivation.** Stage 9 per-map breakdown was unusually informative:

```
MAP26     32.04%   in line with val
MAP27     32.27%   in line with val
MAP28     32.78%   in line with val
MAP32     28.26%   slightly below val
MAP29     19.82%   significantly harder
MAP30     17.27%   significantly harder
```

The model generalizes fine to test maps that visually resemble training
(MAP26-28, MAP32). It fails on the two late-game maps with atypical
lighting and architecture (MAP29 "The Living End", MAP30 "Icon of Sin").
The cause is almost certainly that training maps (MAP01-15 + MAP31) do
not cover that visual style.

**Proposed approach.** Capture an additional 5–10k frames specifically
on:
- Late-game maps (MAP29, MAP30 *or visually similar maps from the WAD
  not used for test*) — to broaden the lighting/architecture distribution.
- Levels with under-represented enemies (SpiderMastermind had zero GT
  in the 2k test sample; deliberate playthroughs of MAP20 and MAP28
  would correct this).

**Expected impact.** Largest single lever. Plausibly +5 to +10 pp on
test mAP, primarily by lifting MAP29/MAP30 performance from ~18% toward
30%.

**Effort.** ~10–15 hours of capture (the most tedious part of the
project), one re-train of Stage 6, evaluation on a *new* test set.

**Why out of scope.** Re-running Stage 6 with augmented data would
require also re-running the entire test protocol with a new test
partition — effectively a second nine-stage arc compressed into one
re-train. The project's scope is the original arc.

---

## 3. Direction B — Multi-scale prediction head (3-grid YOLO)

**Motivation.** The current head outputs at a single 13×13 grid. At
input 416×416, each grid cell corresponds to a 32×32-pixel patch. Small
sprites (Zombieman, distant ShotgunGuy, LostSoul) span less than one
grid cell. This forces the single anchor for that cell to compete
between "small distant enemy" and "no enemy", which biases toward "no
enemy" because the latter dominates training.

Standard YOLOv3 onward uses three prediction scales (13/26/52) with
three anchors per scale (9 anchors total). The 52×52 grid is
specifically responsible for small-object detection.

**Proposed approach.**

- Modify `FineTunedYOLO` to expose intermediate feature maps from the
  ResNet backbone (after block2: 52×52, block3: 26×26, block4: 13×13).
- Add a small FPN-style upsample path so each scale sees both deep
  features and high-resolution spatial information.
- Cluster training boxes into 9 anchors (k-means with k=9, three per
  scale) instead of the current three.
- Modify `build_targets` to assign each GT box to the scale whose
  anchor IoU is best, not just to the 13×13 grid.

**Expected impact.** +3 to +6 pp on test mAP. Largest gains on small,
distant, or partially-occluded enemies (Zombieman currently at 23.42%
test; would likely be the biggest winner).

**Effort.** ~1 full day of architecture work, plus a re-train. The
target-builder changes are the hardest part because they cascade into
the loss function's masking.

**Why out of scope.** This is a *qualitative* architecture change. The
project's narrative is "what happens when you iterate on a fixed
architecture", and substituting a different head would invalidate the
clean attribution from Stages 3 through 8.

---

## 4. Direction C — Stronger augmentation (Mosaic + MixUp)

**Motivation.** Stage 9 implicitly diagnosed the failure mode: the
model learned per-val-map context cues (Stage 7's split-contrast finding
made this explicit; Stage 9's val→test gap confirmed it). Light
augmentation (Stage 6's flip + color jitter) did not break that
memorization.

Mosaic augmentation stitches 4 random training images into one combined
training sample at random crop positions. The resulting image contains
enemies from different maps in different lighting on a single canvas.
MixUp linearly blends two images. Together they force the model to
learn enemy *appearance* in isolation from map context — directly
attacking the failure mode Stage 9 surfaced.

**Proposed approach.**

- Implement `mosaic_4(samples)` that picks 4 random training samples,
  resizes each to a random portion of the canvas, composites them, and
  re-targets all GT boxes accordingly.
- Apply with probability 0.5 during training; pure flip+jitter (Stage 6)
  with probability 0.5.
- MixUp with α=0.2 layered on top with low probability (0.15) — too
  aggressive blending hurts detection.

**Expected impact.** +3 to +8 pp on test mAP, with disproportionate
gains on the regressed-on-test classes (BaronOfHell, ChaingunGuy,
Archvile — the ones that overfit to val-map appearance).

**Effort.** ~1 day. Mosaic's hardest part is correctly re-mapping
boxes through the per-quadrant scaling.

**Why out of scope.** Same argument as Direction B: replacing the
augmentation pipeline meaningfully would invalidate the per-stage
attribution. Specifically, Stage 6's "light augmentation adds +3 pp"
result would no longer be a clean comparison to anything subsequent.

---

## 5. Direction D — Larger backbone

**Motivation.** Stage 8 demonstrated that *loss* tweaks don't lift the
plateau, suggesting the ceiling is somewhere between data and capacity.
ResNet18 has 11.4M backbone parameters; ResNet50 has 23.5M. For
detection-on-pixel-art, the modest jump in capacity is plausibly enough
to capture the additional discriminative features needed to push past
the 34% val plateau.

**Proposed approach.** Swap `torchvision.models.resnet18` for `resnet50`
in `FineTunedYOLO`. Adjust the head's input channel count (512 → 2048).
Reduce batch size to 8 to fit on T4. Halve the learning rate
(2048-channel head needs gentler updates).

**Expected impact.** +2 to +4 pp on test mAP. Lower expected impact than
A/B/C because Stage 8 evidence suggests capacity is not the binding
constraint.

**Effort.** ~2 hours of code, ~3–4 hours of training (the 4× FLOP cost
roughly doubles per-epoch time on T4 because GPU utilization improves).

**Why out of scope.** Lower expected payoff, and conflicts with the
"single backbone family for clean comparison" structure of Stages 4-8.

---

## 6. Direction E — Cheap polish (EMA, TTA, cosine LR)

**Motivation.** Three standard tricks each add ~1–2 pp with near-zero
risk. Combined, ~+3 pp at no architectural cost.

**Proposed approach.**

- **EMA weights**: maintain an exponential moving average of model
  weights (decay 0.9999), evaluate using the EMA copy. Standard YOLO
  trick. ~30 lines.
- **Test-time augmentation**: at inference, predict on both the original
  image and its horizontal flip; merge detections via NMS. ~50 lines.
  Doubles inference time.
- **Cosine LR schedule**: replace the constant Adam learning rate with
  a cosine decay from initial LR to LR/100 over 40 epochs. ~10 lines.

**Expected impact.** +2 to +4 pp on test mAP combined.

**Effort.** ~3 hours total.

**Why out of scope.** Pure engineering polish with no pedagogical
content. Doesn't generate a teachable result. Would be appropriate as
an unmarked "Appendix B: production-ready improvements" rather than a
stage.

---

## 7. Recommended ranking if the project were resumed

If a future version of this project had additional scope and the
capacity to capture a new held-out test set, the recommended order
would be:

| Priority | Direction | Effort | Expected test mAP |
|---|---|---|---|
| 1 | A (more data on hard maps) | ~15h capture + retrain | +5 to +10 pp |
| 2 | C (mosaic + mixup) | ~1 day | +3 to +8 pp |
| 3 | B (multi-scale head) | ~1 day | +3 to +6 pp |
| 4 | E (cheap polish) | ~3 hours | +2 to +4 pp |
| 5 | D (larger backbone) | ~half day | +2 to +4 pp |

Stacking the top three plausibly reaches the 40–50% test mAP range —
which would still be below the random-split val number for the *current*
model (49.58%), but would now be a defensible per-map test figure
representing real generalization.

---

## 8. What this stage explicitly does NOT contain

- No code.
- No experiments.
- No new training runs.
- No new evaluations on the existing test set (which would violate the
  Stage 9 "touch test exactly once" protocol).
- No claim that any of these directions *would* work — only that they
  are the most credible next steps if the project's scope expanded.

---

## 9. Closing note

The Stage 9 test mAP of 24.21% is the project's final reported number.
Stage 10 exists to acknowledge that the result is not a ceiling on the
problem — it is a ceiling on *this specific arc of work*. The model can
be improved; the project, by design, does not improve it further.

The honest version of "future work" in a research writeup is a section
that states what you would do next and explicitly does not do it. This
is that section.
