# Stage 8 — Focal Loss for Class Imbalance: Plan

## Goal

Replace the cross-entropy class loss and BCE objectness loss with **focal
loss** to directly address the class imbalance that Stages 5–6 left
unresolved. Architecture, optimizer, augmentation, and per-map split are all
unchanged from Stage 6 — only the loss function changes.

Expected outcome: **38–45% mAP**, with the biggest per-class gains on rare
and hard classes (SpiderMastermind, Cyberdemon, PainElemental, Zombieman).

---

## 1. What changes from Stage 6

| Item | Stage 6 | Stage 8 |
|---|---|---|
| Backbone | ResNet18 pretrained, fine-tuned | Same |
| Detection head | 1×1 conv → 66 channels | Same |
| Augmentation | Flip + color jitter | Same |
| Optimizer / LRs | Adam, backbone 1e-4, head 1e-3 | Same |
| Epochs | 40 | 40 |
| **Class loss** | CrossEntropy | **Focal loss (γ=2.0)** |
| **Objectness loss** | BCE | **Focal loss (γ=2.0, α=0.25)** |
| Box loss | smooth-L1 + MSE | unchanged |
| Loss weights | λ_box=5, λ_obj=1, λ_noobj=0.5, λ_cls=1 | unchanged |

---

## 2. Why focal loss

### 2.1 The class-imbalance problem

Stage 6 produced per-class APs ranging from 0.45% (Spectre) to 51.94%
(Archvile). The spread comes partly from class abundance:

| Class | Train count | Stage 6 AP |
|---|---|---|
| DoomImp | 6,391 | 38.32% |
| Zombieman | 2,320 | 14.33% |
| SpiderMastermind | 147 | 26.14% |

DoomImp's 6,391 training instances dominate gradient updates. A rare class
like SpiderMastermind contributes only ~2% of the positive-class gradient
signal per epoch. Standard cross-entropy treats all positive samples
equally, so the model preferentially fits the common classes.

### 2.2 What focal loss does

Focal loss modifies cross-entropy with a factor `(1 − p_t)^γ` that
down-weights *easy* examples (high confidence, correct class) and
up-weights *hard* ones:

```
FL(p_t) = -(1 - p_t)^γ · log(p_t)
```

- When `p_t = 0.9` (model is confident and right): `(1-0.9)^2 = 0.01`. Loss is
  100× smaller than CE.
- When `p_t = 0.5` (model is uncertain): `(1-0.5)^2 = 0.25`. Loss is 4×
  smaller than CE.
- When `p_t = 0.1` (model is confidently wrong): `(1-0.1)^2 = 0.81`. Loss is
  ~unchanged from CE.

Net effect: gradient updates focus on the examples the model is *currently
struggling with*. For class imbalance, this means:
- DoomImp examples that the model already gets right → near-zero gradient.
- SpiderMastermind examples that it gets wrong → full gradient.

The model spends its capacity on what's hard instead of polishing what's
easy.

### 2.3 Why focal also helps objectness

Objectness is even more imbalanced than classes: every val frame has
~3 × 13 × 13 = 507 anchor positions, of which typically 1-5 are positive
(have a real enemy). The other ~500 are background.

Standard BCE on this imbalance produces an objectness signal dominated by
"predict 0 everywhere" — easy and uninformative. Focal-objectness with
α=0.25 down-weights the easy-negative anchors that the model already gets
right ("yes, this random patch of wall is not an enemy") so gradient focuses
on the genuinely-ambiguous positions near actual enemies.

This is the original use case from the focal loss paper (RetinaNet, 2017).

---

## 3. Implementation

Two new loss functions replace the BCE/CE calls in `yolo_loss`:

```python
def focal_bce(logits, targets, alpha=0.25, gamma=2.0):
    """Focal binary cross-entropy for objectness."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * ce).sum()


def focal_ce(logits, targets, gamma=2.0):
    """Focal cross-entropy for multi-class classification."""
    log_p = F.log_softmax(logits, dim=-1)
    log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
    pt = log_pt.exp()
    return (-((1 - pt) ** gamma) * log_pt).sum()
```

Everything else in the training pipeline is identical to Stage 6.

---

## 4. Hyperparameter choice

| Param | Value | Reasoning |
|---|---|---|
| `gamma` (both losses) | 2.0 | Standard from the focal loss paper |
| `alpha` (objectness) | 0.25 | Standard from RetinaNet — down-weights easy negatives |
| `alpha` (class) | none | No per-class re-weighting; focal handles within-class imbalance |
| Loss weights | unchanged | Same λ_box=5, λ_obj=1, λ_noobj=0.5, λ_cls=1 |
| Epochs | 40 | Same as Stage 6 (no reason to change) |

These are the literature defaults; not tuned for our specific dataset. Could
be revisited if Stage 8 underperforms.

---

## 5. Expected outcomes

| Class category | Stage 6 AP | Predicted Stage 8 AP |
|---|---|---|
| Common (DoomImp, Zombieman, ShotgunGuy) | 14–38% | small change (already saturated on common-class gradient) |
| Mid-tier (Demon, HellKnight, Cacodemon) | 38–44% | +2–5 pp |
| Rare (Fatso, PainElemental, Cyberdemon, SpiderMastermind) | 26–33% | **+5–15 pp** (the focal payoff) |
| Spectre | 0.45% | small/none (structural problem; focal can't fix data ambiguity) |

Aggregate mAP prediction: **38–45%**.

---

## 6. What this stage does NOT include

- **CIoU box loss**: would replace smooth-L1; usually +2–4 pp. Skipped to
  keep the experiment scoped to "class imbalance specifically."
- **Mosaic / mixup augmentation**: orthogonal axis. If Stage 8 doesn't reach
  target, a future Stage 8b could add these.
- **Multi-scale prediction (3 grids)**: architectural change; would
  significantly improve small-object detection (Zombieman) but requires
  redesigning the head and target builder.
- **Class-balanced sampling**: oversample rare classes in batch construction.
  Alternative approach to the same imbalance problem. Sticking with focal
  loss as the canonical solution.

---

## 7. Reproducibility

| Artifact | Location |
|---|---|
| Script | `stage8.py` |
| Notebook | `stage8_colab.ipynb` |
| Best weights | `stage8_best.pt` |
| Random seed | 42 |
| Runtime | ~2h on Colab T4 |
