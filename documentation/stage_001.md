# Stage 1 — Cropped Enemy Classifier

> **In one sentence:** Train a CNN to look at a small image containing exactly one enemy and predict which of the 17 classes it is.

> **Why this stage exists:** Establishes a working baseline. Confirms data is sane, labels are consistent, classes are visually distinguishable, and your training infrastructure runs end-to-end. The lesson it teaches — by contrast with Stage 2 — is that **classification is not the same problem as detection**.

> **Target outcome:** ~90–95% test accuracy. If you can't reach that on cleanly cropped enemies, something is wrong with the data, not the model.

---

## 1. The task, precisely

**Input:** a small image (e.g. 64×64×3) containing one Doom enemy, edge to edge.

**Output:** a single integer between 0 and 16 — the class ID.

This is plain image classification — the same problem solved by LeNet on MNIST in 1998. There is no:

- "Where is the enemy in this image?" → the crop *is* the enemy.
- "Are there multiple enemies?" → there's always exactly one.
- "Is there an enemy at all?" → yes, every crop has one by construction.

This simplicity is the point. Stage 1 is the easiest version of the problem; if your code, data, and class definitions aren't sound here, *nothing later will work*.

---

## 2. Where the crops come from

You already have everything needed. Your dataset is currently organized as `data/MAPxx/NNNNNN.png` + `data/MAPxx/NNNNNN.txt` pairs. A typical `.txt` file looks like:

```
3 0.411719 0.625000 0.520312 0.745833
0 0.121094 0.580208 0.045312 0.108333
```

Each line is one bounding box. Fields (after the class ID):
- `cx, cy` — bounding box center, normalized to [0, 1] of frame width/height
- `w, h` — bounding box width/height, normalized

To produce a Stage 1 dataset, you walk every `.png`/`.txt` pair and, for each line, compute the pixel-space bounding box:

```
img_w, img_h = frame.shape[1], frame.shape[0]   # e.g. 640, 480
x_center_px = cx * img_w
y_center_px = cy * img_h
w_px = w * img_w
h_px = h * img_h
x1 = int(x_center_px - w_px / 2)
y1 = int(y_center_px - h_px / 2)
x2 = int(x_center_px + w_px / 2)
y2 = int(y_center_px + h_px / 2)
crop = frame[y1:y2, x1:x2]
```

Then save `crop` somewhere with its class ID known. Two common layouts:

**Option A — ImageFolder layout** (recommended for PyTorch convenience):
```
crops/
  Zombieman/         00000.png, 00001.png, ...
  ShotgunGuy/        00000.png, ...
  DoomImp/           (thousands)
  ...
  Cyberdemon/        (few, possibly zero)
```
PyTorch's built-in `ImageFolder` reads this layout for free.

**Option B — flat directory + manifest**:
```
crops/
  000001.png
  000002.png
  ...
  manifest.csv     # rows: filename, class_id
```
More flexibility (you can include per-crop metadata like source map, source frame), more boilerplate.

Either works. Start with Option A.

### The crop-size problem

Your bbox crops will be wildly different sizes — a far-away Zombieman might be 14×30 px, a close-up Cyberdemon 250×400 px. CNNs need fixed-size input. You'll resize every crop to a standard size before feeding the model.

- **64×64** is a common starting choice for small classifiers.
- **96×96** if you want a bit more detail.
- Larger means slower training and slight diminishing returns on a task this simple.

You lose some information (a tall Revenant squished into a square loses its tall-ness as a class cue), but for Stage 1 this is acceptable. The point is the *foundation*, not state-of-the-art.

```python
# Pseudocode
import cv2
crop = cv2.imread("crop_path.png")
crop = cv2.resize(crop, (64, 64))   # now (64, 64, 3)
```

### Aspect-ratio choice (small but worth a comment)

You have two options when resizing to a square:
- **Squash to square** — `cv2.resize(crop, (64, 64))`. Simple. Distorts aspect ratio.
- **Pad to square first, then resize** — preserves aspect ratio, adds black bars.

Stage 1 commonly uses *squash*. Stage 6 (augmentation) is where you might revisit this if you suspect distortion is hurting rare classes.

---

## 3. Train / val / test split

Stage 7 of the arc is *all about* getting the split right. Stage 1 should still use the split it's eventually going to use — there's no point training on data that'll later be in your test set.

**The split, by map:**

```
train: MAP01–MAP15        (≈55% of total frames)
val:   MAP16–MAP25        (≈25%)
test:  MAP26–MAP32        (≈20%)
```

(Exact map allocation depends on how many maps you ended up capturing — adjust to ~60/20/20 proportions.)

**Why by map and not random?** Frames from the same map are visually very similar (same textures, same lighting, same enemy positions appearing repeatedly). If you split *randomly*, two near-identical frames could land one in train and one in test — and your test "accuracy" would partially reflect memorization. Per-map split forces the model to generalize to *new environments*, which is the honest test.

**Practical caveat:** you only captured a few classes in specific maps. If you split MAP01–15 into train and MAP26+ into test, but Cyberdemons only exist in MAP30/MAP31, then:
- Train sees 0 Cyberdemons → model never learns to recognize them
- Test sees Cyberdemons → model fails 100% on them

This is real, and it's exactly the Stage-7 lesson. For Stage 1 you can either:
- Live with it (rare classes have low AP, model is "honest" but obviously broken on those classes).
- Or do a quick re-balance: ensure each split contains at least some frames of every class. This is impure but pragmatic; just document it.

You'll come back to this in Stage 7.

---

## 4. The model

A small CNN. Conceptually:

```
Input: 64 × 64 × 3 image (RGB pixel values)

  ↓ Conv 3×3, 32 filters → ReLU → MaxPool 2×2
                                            (becomes 32 × 32 × 32)
  ↓ Conv 3×3, 64 filters → ReLU → MaxPool 2×2
                                            (becomes 16 × 16 × 64)
  ↓ Conv 3×3, 128 filters → ReLU → MaxPool 2×2
                                            (becomes 8 × 8 × 128)
  ↓ Flatten to a 8192-dim vector
  ↓ Linear (fully-connected) → 17 outputs (logits, one per class)
```

In PyTorch this is roughly:

```python
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=17):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

### What the layers actually do

- **`Conv2d`** — slides a small filter (3×3 here) across the input image and computes a weighted sum at each position. With 32 filters, the layer learns 32 different "feature detectors" — early ones tend to detect edges, later ones detect more abstract patterns. Each filter produces one "feature map" the size of the input.
- **`ReLU`** — element-wise nonlinearity: `f(x) = max(0, x)`. Without nonlinearities, stacked layers collapse mathematically into a single linear layer.
- **`MaxPool2d(2)`** — downsamples by 2× in each spatial dimension, keeping only the max value in each 2×2 block. Halves spatial size, makes the model translation-tolerant, reduces parameter count.
- **`Flatten`** — reshapes `(batch, 128, 8, 8)` into `(batch, 8192)` so the linear layer can consume it.
- **`Linear(8192, 17)`** — fully-connected: each of 8192 input features can affect each of 17 outputs. Pure dense matrix multiply (+ bias).

Output is a vector of **17 raw scores called *logits***. They're not probabilities yet. We never explicitly call softmax during training because `CrossEntropyLoss` does it internally (more efficient and numerically stable that way).

### Total parameters

Rough math:
- Conv1: 3 × 32 × 3 × 3 = 864 weights + 32 biases
- Conv2: 32 × 64 × 9 + 64 = 18,496
- Conv3: 64 × 128 × 9 + 128 = 73,856
- Linear: 8192 × 17 + 17 = 139,281

**~232k total parameters.** Tiny by modern standards. Trains in minutes even on CPU; seconds on GPU.

---

## 5. The training loop

A single training step is conceptually four operations:

1. **Get a batch:** the DataLoader yields a tuple `(images, labels)`.
   - `images` is a `(batch_size, 3, 64, 64)` float tensor.
   - `labels` is a `(batch_size,)` integer tensor of class IDs.
2. **Forward pass:** `logits = model(images)` — shape `(batch_size, 17)`.
3. **Compute loss:** `loss = criterion(logits, labels)` — a single scalar.
4. **Backward pass:** `optimizer.zero_grad(); loss.backward(); optimizer.step()`.

In PyTorch this looks like:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleClassifier(num_classes=17).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Evaluate on validation set here (see Section 6)
```

### Key concepts

- **Epoch:** one full pass over the training set. 10–20 epochs is plenty for Stage 1.
- **Batch:** a group of training examples processed together. Larger batches → smoother gradients but more memory. 64–256 is typical for tasks this small.
- **`model.train()` / `model.eval()`** — toggles dropout and batchnorm behavior. You don't have either in the simple model above, so it doesn't matter yet, but get in the habit.
- **`zero_grad`** — gradients accumulate by default in PyTorch. You clear them before each backward pass.
- **`loss.backward()`** — computes gradients of the loss with respect to every parameter via automatic differentiation.
- **`optimizer.step()`** — updates parameters using the gradients and the optimizer's update rule (Adam, SGD, etc.).

### Choice of optimizer and learning rate

- **Adam, learning rate 1e-3:** the modern "just works" default. Start here.
- **SGD with momentum 0.9, learning rate 1e-2:** classical, sometimes generalizes slightly better but more sensitive to LR tuning.

You almost certainly don't need to tune for Stage 1.

---

## 6. The loss function

`nn.CrossEntropyLoss()` is the right loss for multi-class single-label classification. What it does internally:

```
P(class i | input) = exp(logit_i) / sum_j exp(logit_j)        # softmax
loss = -log(P(true class | input))                            # negative log likelihood
```

Equivalently: for each example, the loss is `-log` of the predicted probability of the true class. Loss is 0 if the model perfectly predicts the true class with probability 1; loss grows as the predicted probability decreases.

The "cross-entropy" name comes from information theory but you don't need to understand the derivation to use it correctly. Just know:
- Input: raw logits (not softmaxed).
- Input shape: `(batch_size, num_classes)`.
- Target: integer class IDs.
- Target shape: `(batch_size,)`.
- Output: a single scalar loss for the batch.

---

## 7. Evaluation

After each epoch — and definitely at the end of training — run on the validation/test set with gradients off:

```python
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)            # for each example, the class with highest score
        correct += (preds == labels).sum().item()
        total += labels.size(0)
accuracy = correct / total
print(f"Validation accuracy: {accuracy:.3%}")
```

### Three things to report at the end

**1. Overall top-1 accuracy.** What percent of crops were classified correctly. Expect 90–95%+ for clean crops.

**2. Per-class accuracy.** Compute accuracy separately for each of the 17 classes. The class imbalance you saw — DoomImp 10× more frequent than HellKnight — will manifest here:
```python
from collections import defaultdict
correct_per_class = defaultdict(int)
total_per_class   = defaultdict(int)
# ... in the eval loop:
for p, l in zip(preds, labels):
    if p == l:
        correct_per_class[l.item()] += 1
    total_per_class[l.item()] += 1
# then per-class accuracy = correct[c] / total[c] for each c
```

Per-class accuracy shows whether the model genuinely learned each class or just memorized "always say DoomImp when in doubt."

**3. Confusion matrix.** A 17×17 grid where row = true class, column = predicted class, value = count. Diagonal entries are correct predictions; off-diagonal are confusions. Reveals systematic patterns like "the model confuses HellKnight and BaronOfHell" — common since they're visually similar (tall, hooved humanoids; freedoom2's redesigns of them, "Pain Bringer" and "Pain Lord," are nearly indistinguishable).

```python
from sklearn.metrics import confusion_matrix
import numpy as np
# accumulate `all_preds` and `all_labels` lists across the eval loop, then:
cm = confusion_matrix(all_labels, all_preds, labels=list(range(17)))
```

Visualize with matplotlib's `imshow` for a heatmap.

---

## 8. What to expect and what to write up

**Expected outcome:**
- Train accuracy: ~99%
- Val accuracy: 88–95%
- Test accuracy: 85–93%

Train > val > test is normal (it's why you have the splits). A big gap (train 99%, val 60%) indicates *overfitting* — the model memorized the training set. For Stage 1 this is unlikely with so few parameters, but it's worth knowing.

**What goes in the writeup for Stage 1:**

- Brief paragraph: "Stage 1 — cropped classifier on bbox crops. Predicted outcome: ~95% accuracy. Actual outcome: ____%."
- The model architecture (the schematic from Section 4).
- Training curve (loss over epochs, train + val accuracy over epochs) — one matplotlib plot.
- Confusion matrix — one image.
- Per-class accuracy table — 17 rows.
- 1–2 sentences on the most-confused class pairs and why (HellKnight ↔ BaronOfHell is the classic).
- **The lesson:** "This is high accuracy but it's not detection. The model needs to know *where* an enemy is in a full frame, not just classify a pre-cropped enemy. Stage 2 attempts that naively with a sliding window over the trained Stage-1 classifier."

The writeup at this stage should be 1–2 pages tops. The interesting stuff is in later stages — Stage 1's job is to set the stage.

---

## 9. Why Stage 1 isn't a throwaway

Beyond establishing the baseline, Stage 1 produces a **reusable component**: the trained classifier itself.

In Stage 2 (sliding window), you literally take this trained classifier and apply it to a grid of windows across each full frame, treating the classifier's output as "is there a [class] here at this scale?". So Stage 2 isn't a new model — it's *Stage 1 applied repeatedly to crops the model didn't have*. The slowness and duplicate-prediction problems that result are what motivate Stage 3's one-shot detector.

The Stage 1 → 2 → 3 sequence is a guided walk through *why one-shot detectors like YOLO were invented*. You're recreating, in miniature, the history of object detection from ~2012 to ~2016.

---

## 10. What to read to absorb the concepts

You don't need to internalize a huge amount before writing Stage 1 code. Focus on:

1. **PyTorch fundamentals — Dataset and DataLoader.** 30 minutes of reading.
   - Official tutorial: "Datasets & DataLoaders" in PyTorch docs.
   - The mental model: `Dataset` is "how to get item N"; `DataLoader` is "give me batches in shuffled order."

2. **What a training loop looks like.** Any "Hello PyTorch" / "MNIST in PyTorch" tutorial.
   - You're looking for the forward-loss-backward-step pattern.

3. **Softmax + cross-entropy intuitively.** 15 minutes.
   - 3Blue1Brown's video on neural networks ep. 3 has a clear intuitive treatment.
   - Mental model: softmax converts logits to probabilities; cross-entropy measures how confidently-wrong the model was.

4. **Train/val/test split — why it matters.** Any intro ML course covers this.
   - Mental model: the train set is for *learning*; the val set is for *tuning your decisions* (when to stop training, which model variant to use); the test set is for *one final honest measurement* of generalization.

5. **CNNs intuitively — what conv layers actually do.** 30 minutes.
   - Stanford's CS231n notes on convolutional networks.
   - The "what each filter learns" intuition is more useful than the math.

That's the stack you need *for Stage 1*. Don't dive into anchor boxes, IoU, mAP, NMS, focal loss, mosaic augmentation, or detection-head design yet. All of that is Stages 3+ material. Stage 1 is just classification.

---

## 11. Checklist before writing any code

- [ ] Decide on framework (likely **PyTorch**).
- [ ] Decide on compute target (local GPU? Colab T4?).
- [ ] Decide on code structure (likely **Jupyter notebook**, one section per stage).
- [ ] Generate the crops directory from `data/MAPxx/*.png` + `.txt`. Script-once, save to disk, never regenerate.
- [ ] Decide on input size (suggest **64×64**).
- [ ] Decide on the train/val/test map allocation (write it down once, reuse across stages).
- [ ] Confirm `crops/` directory has plausible counts: hundreds of crops for common classes, tens for rare ones.
- [ ] Visualize a few crops manually (just open the PNGs and look at them) — confirms cropping math is correct.
- [ ] Read up on PyTorch Dataset/DataLoader before starting the code.

Only then start writing the model.
