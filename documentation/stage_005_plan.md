# Stage 5 — Pretrained Backbone, Unfrozen + Fine-tuned: Plan

## Goal

Fix Stage 4's frozen-BatchNorm failure by **unfreezing the entire model and
fine-tuning end-to-end with a discriminative learning rate** — head at standard
LR (1e-3), backbone at 10× lower (1e-4). This lets BatchNorm running statistics
adapt to Doom's pixel distribution while preserving the useful low-level
features that ImageNet pretraining bought us.

Expected outcome: **50–70% val mAP** — the lift that Stage 4 was supposed to
deliver but couldn't.

---

## 1. What changes from Stage 4

| Item | Stage 4 | Stage 5 |
|---|---|---|
| Backbone weights | Frozen (`requires_grad=False`) | **Trainable** |
| Backbone mode | Forced `eval()` (BN stats frozen) | **Normal `train()`** (BN stats update) |
| Forward pass | `torch.no_grad()` around backbone | Normal autograd through backbone |
| Optimizer | Adam on head params only | **Adam with two LR groups**: backbone 1e-4, head 1e-3 |
| Trainable params | 34k | **11.2M** (full ResNet18 + head) |
| Epochs | 30 | 30 (same; should converge fast given pretrained start) |
| Everything else | (loss, anchors, data, eval, ImageNet norm) | identical |

---

## 2. Why this fixes Stage 4

Stage 4 failed because of **two compounding issues**:
1. ResNet18's BatchNorm running statistics, baked in during ImageNet pretraining,
   produced mis-scaled features when applied to Doom pixel art.
2. The 34k-parameter head lacked the capacity to compensate for cascaded
   mis-normalization across 17 BN layers.

Stage 5 addresses both:
1. **BN in train mode → running stats update.** Each forward pass during
   training updates the running mean/variance using actual Doom-frame data.
   After a few epochs, the stats reflect Doom's distribution, BN normalizes
   correctly, intermediate features become well-scaled.
2. **All conv weights are tunable.** The backbone gets to specialize its
   features for Doom while starting from useful ImageNet priors. It's not
   limited to whatever pretrained features happened to be useful.

The 10×-lower backbone LR is the standard fine-tuning recipe:
- **High LR on the head** (random initialization → big gradients OK)
- **Low LR on the backbone** (already-trained weights → small adjustments to
  avoid destroying the pretrained priors)

Without this LR split, a uniform LR of 1e-3 would *over-update* the backbone
and destroy the pretrained features in the first few epochs — undoing the
whole reason we used a pretrained model. With a uniform 1e-4, the head would
train too slowly. The two-group setup gets both right.

---

## 3. Implementation

### 3.1 Model — no freezing

```python
class FineTunedYOLO(nn.Module):
    def __init__(self, num_classes, num_anchors):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.head = nn.Conv2d(512, num_anchors * (5 + num_classes), 1)
        # No freezing; no eval() forcing; no torch.no_grad in forward.

    def forward(self, x):
        return self.head(self.backbone(x))
```

When `model.train()` is called (in the training loop), both backbone and head
go into train mode. BN layers in the backbone update their running statistics
during forward passes.

### 3.2 Optimizer — discriminative LR

```python
optimizer = torch.optim.Adam([
    {"params": model.backbone.parameters(), "lr": 1e-4},
    {"params": model.head.parameters(),     "lr": 1e-3},
])
```

This is the single most important hyperparameter in Stage 5. A uniform LR (one
group, single learning rate) is the most common Stage 5 mistake; it usually
loses 5–15 pp of final mAP.

### 3.3 No other architectural changes

- Same anchors as Stage 3/4
- Same target builder, loss, decoder, NMS
- Same ImageNet input normalization (the backbone *expects* this; it's the
  whole point of using a pretrained model)
- Same per-epoch val eval

---

## 4. Hyperparameters

| Hyperparameter | Value | Reasoning |
|---|---|---|
| Backbone LR | 1e-4 | Standard fine-tuning practice; preserves pretrained features |
| Head LR | 1e-3 | Head is random-init; needs larger steps |
| Batch size | 16 | Memory-bound on T4 with full backprop through ResNet18 |
| Epochs | 30 | Pretrained features mean convergence is fast |
| Optimizer | Adam | Same as Stage 3/4 |
| Loss weights | unchanged | λ_box=5, λ_obj=1, λ_noobj=0.5, λ_cls=1 |
| Gradient clip | max_norm=10 | Same as Stages 3/4 for stability |

No LR schedule (cosine, step, etc.). Adding scheduling typically adds 1–3 pp
mAP but doubles experimental complexity; could be a Stage 6+ refinement.

---

## 5. Expected dynamics

**Compared to Stage 4**:

- **Loss should drop steadily**, not plateau. Stage 4 stuck at ~11.0; Stage 5
  should reach ~1.0 within 5 epochs, ~0.5 by epoch 20.
- **First 2–3 epochs**: BN running stats are *still* mostly the ImageNet ones,
  so initial val mAP will be similar to Stage 4 (~5%). Once BN adapts, val
  jumps.
- **Epochs 3–10**: rapid improvement as features specialize.
- **Epochs 10–25**: gradual climb toward the plateau.
- **Epochs 25–30**: convergence; saving best by val.

**Predicted per-class outcomes:**

- All multi-pp jumps compared to Stage 3.
- Classes bottlenecked on features (Cyberdemon, Cacodemon, BaronOfHell/HellKnight
  pair) will improve the most.
- Classes bottlenecked on data or fundamental ambiguity (Spectre,
  SpiderMastermind) may still struggle but should at least appear at some level
  on the radar.

---

## 6. What could still go wrong

- **BN adapts but overfits**: with 11.2M trainable parameters and only 12.5k
  training frames, overfitting risk is real. Mitigation: early stopping by val
  (the save-best mechanism handles this).
- **Catastrophic forgetting of ImageNet features**: too high a backbone LR
  could destroy the pretrained priors. We're using 1e-4 which is the
  literature-standard safe choice.
- **BatchNorm running stats too volatile early**: very rare, but possible if
  Doom batch statistics differ a lot from ImageNet's. If we see oscillation in
  early epochs, would try momentum=0.1 (more averaging) but unlikely to matter
  here.
- **Memory OOM**: full backprop through ResNet18 at 416×416 with batch 16
  is ~6–8 GB. Should fit comfortably in T4's 16 GB but if OOM, drop batch
  to 8.

---

## 7. Reproducibility

| Artifact | Location |
|---|---|
| Script | `stage5.py` |
| Notebook | `stage5_colab.ipynb` |
| Best weights | `stage5_best.pt` (saved each improving epoch) |
| Random seed | 42 |
