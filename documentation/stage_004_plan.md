# Stage 4 — Pretrained Backbone (Frozen): Plan

## Goal

Reuse Stage 3's detection-head architecture but replace the from-scratch CNN
backbone with **ImageNet-pretrained ResNet18**, with backbone weights frozen.
Only the detection head trains. This is the cleanest "transfer learning"
experiment: measure how much pretrained features help, *holding everything
else constant*.

Expected outcome: **50–65% val mAP**, a 2.4–3× jump over Stage 3's 21.12%.
That's the project arc's biggest single beat.

---

## 1. What changes from Stage 3

| Component | Stage 3 | Stage 4 |
|---|---|---|
| Backbone | 5-stage from-scratch CNN (3.7M params) | ResNet18 pretrained on ImageNet (11.2M params, *frozen*) |
| Detection head | 1×1 conv → 66 channels | Same: 1×1 conv → 66 channels |
| Trainable parameters | ~4.75M | **~34,000 (head only)** |
| Input normalization | scale to [0, 1] | scale to [0, 1] **then ImageNet mean/std** |
| Training epochs | 50 | 30 (less needed; head is tiny) |
| LR | 1e-3 (Adam) | 1e-3 (Adam, head only) |
| Loss / assignment / decoder | identical | identical |
| Evaluation | identical | identical |

The head is the same single `nn.Conv2d(512, A*(5+C), 1)` from Stage 3 — produces
exactly the same output shape `(B, 66, 13, 13)`. ResNet18's `layer4` output is
already `(B, 512, 13, 13)` for a 416×416 input, so the dimensions line up
perfectly with no modification.

---

## 2. Why this should work

### 2.1 Pretrained features are a free starting point

ImageNet pretraining gives the backbone:
- **Low-level features**: edges, textures, gradients trained on 1.2M diverse
  photographs.
- **Mid-level features**: object parts, common shapes, color combinations.
- **High-level features**: object-category-like representations.

The from-scratch backbone in Stage 3 had to learn all of these from 12.5k Doom
frames — and ran out of training data / variety before learning them well.
Stage 4 gets them for free.

### 2.2 What the head learns

With backbone frozen, the head is purely a **linear classifier** on the
backbone's last-layer features (well, technically a 1×1 conv, which is a linear
classifier applied independently at each grid cell). It learns:
- For each (cell, anchor): "given these 512-dimensional feature vectors at this
  location, predict (box offsets, objectness, class)."
- The features it operates on are *pretrained, general-purpose visual
  representations* — much richer than what Stage 3 could learn from its data.

### 2.3 The freezing tradeoff

**Pros**: very few trainable parameters (~34k), much faster training, less
overfitting risk, no destruction of useful pretrained features.

**Cons**: the backbone features are tuned for ImageNet (natural photos), not
Doom pixel art. Some performance is left on the table because the backbone
can't adapt to the domain.

Stage 5 will unfreeze the backbone (jointly fine-tune backbone + head at a
much lower LR), allowing domain adaptation. Stage 4 is the cleaner *upper
bound on what transfer learning alone gives you* — without any domain
adaptation contaminating the signal.

---

## 3. Implementation details

### 3.1 Loading the backbone

```python
from torchvision import models
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# Strip avgpool + fc (we don't want classification, we want spatial features)
backbone = nn.Sequential(*list(resnet.children())[:-2])
# Output: (B, 512, 13, 13) for 416×416 input
```

### 3.2 Freezing

```python
for p in backbone.parameters():
    p.requires_grad = False
backbone.eval()  # IMPORTANT: keeps BatchNorm in inference mode (uses running stats)
```

The `eval()` call is critical. ResNet's BatchNorm layers have learnable parameters
*and* running statistics; calling `train()` puts them in update-running-stats
mode, which corrupts pretrained features. We need them in `eval()` mode always.

The detection head trains normally.

### 3.3 Input normalization

ImageNet pretrained models expect inputs normalized with ImageNet statistics:

```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
tensor = (tensor - mean) / std
```

This is critical — feeding raw [0,1] pixels into a model trained on
mean-subtracted input would produce nonsense features. Cuts performance in half
if you forget.

### 3.4 Optimizer setup

```python
# Only optimize the head's parameters
optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
```

Passing only `model.head.parameters()` means no gradient updates touch the
backbone even if `requires_grad` were accidentally True somewhere. Belt and
suspenders.

---

## 4. Hyperparameter intuition

| Hyperparameter | Value | Reasoning |
|---|---|---|
| `LR` | 1e-3 | Head is small, can take aggressive lr; same as Stage 3 |
| `EPOCHS` | 30 | Head converges quickly; Stage 3 needed 50 to overfit but Stage 4 will converge faster |
| `BATCH_SIZE` | 16 | Same as Stage 3 (memory-bound on T4) |
| Loss weights | unchanged | Same architecture, same loss; weights remain λ_box=5, λ_obj=1, λ_noobj=0.5, λ_cls=1 |
| Anchor sizes | unchanged | Box regression target distributions unchanged |

---

## 5. Expected dynamics

**Compared to Stage 3:**

- **Loss components**: should start at *lower* values (richer initial features
  produce more accurate predictions immediately) and converge faster.
- **Val mAP**: should reach Stage 3's peak (~20%) within 3–5 epochs, then keep
  climbing to ~50–60%.
- **Overfitting**: less severe than Stage 3 because the head has 100× fewer
  trainable parameters; head overfits slowly.
- **Training time**: maybe slightly faster per epoch (forward pass through
  frozen backbone uses `torch.no_grad()` — no backward through it).

**Predicted per-class outcomes:**

- All classes should improve relative to Stage 3.
- The ones bottlenecked on *features* (HellKnight/BaronOfHell pair, Cyberdemon
  with limited pose variety, generic-looking Zombieman) will improve the most.
- The ones bottlenecked on *data* or *fundamental ambiguity* (Spectre,
  SpiderMastermind) will improve less or not at all — pretrained features don't
  help with information you don't have.

---

## 6. Implementation notes

`stage4.py` is essentially `stage3.py` with three diffs:
1. New `PretrainedYOLO` class replacing `YOLODetector`.
2. `FrameDetectionDataset` adds ImageNet normalization.
3. Optimizer only updates head parameters; EPOCHS=30 instead of 50.

Everything else (loss, target builder, NMS, mAP eval, training loop, decoding)
is reused verbatim from `stage3.py` via import. The point is *minimal change*
so the mAP delta is attributable specifically to the backbone swap.
