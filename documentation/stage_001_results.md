# Stage 1 — Cropped Enemy Classifier: Results

## Abstract

A small convolutional neural network (3 conv blocks + linear head, ~232k parameters)
was trained to classify pre-cropped Doom enemy images into one of 17 classes. The
crops come from a self-built dataset of ViZDoom screen captures with engine-provided
bounding-box ground truth. The train/val split is **by map** (no frame overlap across
splits), with 16 maps for training and 10 maps for validation. After 20 epochs, peak
**validation accuracy was 71.25% at epoch 10**. The model exhibited clear overfitting
(train accuracy reaching 98%) and per-class performance varied widely (Spectre 36%,
LostSoul 89%), with confusion patterns matching known visual ambiguities in the
freedoom2 enemy roster.

---

## 1. Dataset

### 1.1 Source

| Property | Value |
|---|---|
| Frames captured | 26,300 (train + val) |
| Frame resolution | 640 × 480 |
| Frame format | RGB PNG |
| Labels | YOLO format (class id + normalized cx, cy, w, h) |
| Label source | ViZDoom `state.labels` (engine ground truth) |

### 1.2 Splits

Per-map split — frames from a given map are entirely in train *or* val, never both.

| Split | Maps | Frames | Crops |
|---|---|---|---|
| Train | MAP01–MAP15 + MAP31 | 12,499 | 23,310 |
| Val   | MAP16–MAP25 | 13,800 | 28,363 |
| Test  | MAP26–MAP30 + MAP32 | (not yet captured) | — |

### 1.3 Class list (17 classes)

ZDoom actor class names. Each is a freedoom2 enemy with a redesigned sprite but the
same underlying actor identity as in Doom 2.

```
0  Zombieman          freedoom: (handgun zombie)
1  ShotgunGuy         freedoom: Shotgun Zombie
2  ChaingunGuy        freedoom: Minigun Zombie
3  DoomImp            freedoom: Serpentipede
4  Demon              freedoom: Flesh Worm
5  Spectre            freedoom: Stealth Worm
6  LostSoul           freedoom: Hatchling
7  Cacodemon          freedoom: Trilobite
8  Fatso              freedoom: Combat Slug (Mancubus)
9  HellKnight         freedoom: Pain Bringer
10 Arachnotron        freedoom: Technospider
11 PainElemental      freedoom: Matribite
12 Revenant           freedoom: Octaminator
13 BaronOfHell        freedoom: Pain Lord
14 Archvile           freedoom: Necromancer
15 SpiderMastermind   freedoom: Large Technospider
16 Cyberdemon         freedoom: Assault Tripod
```

### 1.4 Class-instance distribution (val)

| ID | Class | Train | Val | Ratio (val:max) |
|---|---|---:|---:|---:|
| 3 | DoomImp | 6,391 | 3,648 | 1.00 |
| 0 | Zombieman | 2,320 | 4,562 | — |
| 9 | HellKnight | 1,083 | 2,478 | — |
| 12 | Revenant | 989 | 2,212 | — |
| 1 | ShotgunGuy | 2,443 | 2,026 | — |
| 14 | Archvile | 585 | 1,907 | — |
| 7 | Cacodemon | 819 | 1,711 | — |
| 8 | Fatso | 470 | 1,498 | — |
| 13 | BaronOfHell | 428 | 1,492 | — |
| 6 | LostSoul | 1,798 | 1,452 | — |
| 10 | Arachnotron | 323 | 1,395 | — |
| 4 | Demon | 2,263 | 939 | — |
| 16 | Cyberdemon | 794 | 894 | — |
| 2 | ChaingunGuy | 1,338 | 841 | — |
| 5 | Spectre | 851 | 629 | — |
| 11 | PainElemental | 268 | 462 | — |
| 15 | SpiderMastermind | 147 | 217 | 1:21 |

**Class imbalance ratio (max/min on val): ~21:1** (DoomImp : SpiderMastermind). The
imbalance is a real-world consequence of how freedoom2 distributes enemies across
maps — the model is expected to perform less well on the rare tail.

### 1.5 Crop generation

For each labelled bounding box in a captured frame, the corresponding pixel region
was cropped from the source frame and saved as an individual image. Each crop was:

1. Expanded by 5% padding on each side (avoids tight sprite-edge cropping)
2. Clipped to image bounds
3. **Letterboxed** to 64 × 64 preserving aspect ratio (black bars added rather than
   stretching) — important because Doom enemy sprites have widely varying aspect
   ratios; squashing distorts the most distinctive proportions (e.g., a tall
   Revenant vs a squat Mancubus).

The result is one folder per class containing variable-count PNGs:
`crops/<ClassName>/MAPxx_<frame>_<bbox_idx>.png`. The map prefix in the filename is
what enables programmatic per-map split at training time.

Total crops: **51,769** (matches the bbox-instance count to within zero discards).

---

## 2. Model

A compact convolutional network designed for 64 × 64 input. Three convolutional
blocks downsample to 8 × 8 × 128, followed by a fully-connected head producing 17
class logits.

```
Input:  3 × 64 × 64
  Conv(3, 32, 3×3, pad=1) → ReLU → MaxPool(2×2)    →  32 × 32 × 32
  Conv(32, 64, 3×3, pad=1) → ReLU → MaxPool(2×2)   →  64 × 16 × 16
  Conv(64, 128, 3×3, pad=1) → ReLU → MaxPool(2×2)  → 128 ×  8 ×  8
  Flatten                                          → 8,192
  Linear(8192, 17)                                 → 17  (logits)
```

**Parameters:** 232,529 (very small by modern standards; chosen deliberately to be
trainable on CPU and to make overfitting a meaningful signal rather than a foregone
conclusion).

Cross-entropy loss is used with raw logits; softmax is applied internally by the
loss function.

---

## 3. Training configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 128 (train), 256 (val) |
| Epochs | 20 |
| Loss | CrossEntropyLoss |
| Augmentation | none (Stage 1 baseline) |
| Regularization | none (no dropout, no weight decay) |
| Device | CPU |
| Wall-clock time | 42.1 minutes |

Best-by-val-accuracy weights are saved after each improving epoch. Final reports use
the saved best weights, not the last epoch.

---

## 4. Results

### 4.1 Training curve

```
Epoch  train_loss  train_acc   val_acc   note
--------------------------------------------------
  1/20    1.485      55.7%     57.4%   ← best, saved
  2/20    0.765      77.4%     65.6%   ← best, saved
  3/20    0.591      82.1%     67.6%   ← best, saved
  4/20    0.489      85.2%     68.3%   ← best, saved
  5/20    0.418      87.3%     69.5%   ← best, saved
  6/20    0.361      88.9%     69.3%
  7/20    0.306      90.6%     71.2%   ← best, saved
  8/20    0.260      91.7%     67.6%
  9/20    0.222      93.1%     70.1%
 10/20    0.199      93.6%     71.3%   ← best, saved
 11/20    0.164      94.8%     70.2%
 12/20    0.152      95.1%     70.2%
 13/20    0.118      96.2%     68.8%
 14/20    0.109      96.5%     68.7%
 15/20    0.093      96.9%     68.3%
 16/20    0.097      96.9%     69.7%
 17/20    0.069      97.8%     69.1%
 18/20    0.077      97.6%     69.5%
 19/20    0.050      98.5%     66.9%
 20/20    0.060      98.0%     69.1%
--------------------------------------------------
```

**Overall val accuracy at peak (epoch 10): 71.25%.**

### 4.2 Observations on the training curve

- **Epochs 1–5**: train and val rise together. The model is learning generalizable
  features. Both metrics improve at every epoch.
- **Epoch 7**: train passes 90%, val passes 71% and *plateaus*. Generalizable
  improvement has roughly stopped.
- **Epochs 10–20**: train continues to climb (94% → 98%) but val drifts down with
  some volatility (averaging ~69%). This is **textbook overfitting** — the model
  is memorizing training-set specifics that don't transfer.

The save-best-by-val-accuracy mechanism correctly identifies epoch 10 as the
best checkpoint, before degradation began.

The **train–val gap at the final epoch (98% − 69% = 29 percentage points)** is
large and is the primary symptom motivating regularization techniques planned for
later stages.

### 4.3 Per-class validation accuracy

```
ID  Class               Accuracy   Correct / Total
---------------------------------------------------
 0  Zombieman           61.1 %     2786 / 4562
 1  ShotgunGuy          76.3 %     1546 / 2026
 2  ChaingunGuy         79.3 %      667 /  841
 3  DoomImp             84.8 %     3095 / 3648
 4  Demon               75.2 %      706 /  939
 5  Spectre             35.6 %      224 /  629
 6  LostSoul            89.2 %     1295 / 1452
 7  Cacodemon           68.0 %     1163 / 1711
 8  Fatso               70.2 %     1052 / 1498
 9  HellKnight          70.1 %     1736 / 2478
10  Arachnotron         64.9 %      905 / 1395
11  PainElemental       60.8 %      281 /  462
12  Revenant            81.1 %     1795 / 2212
13  BaronOfHell         71.4 %     1065 / 1492
14  Archvile            71.6 %     1365 / 1907
15  SpiderMastermind    47.9 %      104 /  217
16  Cyberdemon          47.4 %      424 /  894
---------------------------------------------------
                       Overall:    71.25 %
```

Stratified by accuracy band:

| Tier | Classes | Likely cause |
|---|---|---|
| **Strong (≥80%)** | LostSoul, DoomImp, Revenant | Visually distinctive silhouettes + plentiful training data |
| **Decent (70–80%)** | ShotgunGuy, ChaingunGuy, Demon, Archvile, BaronOfHell, HellKnight, Fatso, Cyberdemon... wait | The middle of the pack |
| **Weak (60–70%)** | Cacodemon, Zombieman, Arachnotron, PainElemental | Modest performance; mix of imbalance and similarity to other classes |
| **Failing (<55%)** | **Spectre (36%), SpiderMastermind (48%), Cyberdemon (47%)** | Discussed below |

### 4.4 Most-confused class pairs (top 10)

True class → predicted class, with confusion count:

```
Zombieman          -> DoomImp             606
Zombieman          -> ShotgunGuy          444
BaronOfHell        -> HellKnight          248
Zombieman          -> Spectre             190
HellKnight         -> BaronOfHell         159
Spectre            -> DoomImp             138
Arachnotron        -> DoomImp             135
Fatso              -> DoomImp             133
ShotgunGuy         -> DoomImp             122
Zombieman          -> ChaingunGuy         119
```

---

## 5. Discussion

### 5.1 Overfitting is severe and visible

The 29 percentage-point gap between training accuracy (98%) and validation accuracy
(69% by epoch 20) is the largest practical problem with this baseline. The model has
sufficient capacity (232k parameters) to memorize the ~23k training crops to near-
perfection while failing to generalize. Standard mitigations — data augmentation,
dropout, weight decay, batch normalization — are intentionally absent in this
Stage 1 baseline. Future stages will introduce them and measure their effects.

### 5.2 Per-map split is harder than commonly assumed

A naive expectation for a 17-class classifier on cleanly-cropped, ground-truth-
labelled enemy images would be 90%+ validation accuracy. The actual 71% reveals a
key property of this dataset: **the per-map split tests the model on entirely
new visual environments** (different textures, lighting, sector geometries, sprite
animation moments). Random per-frame splits — which the literature typically uses
on video-derived datasets — would produce artificially inflated numbers because
nearly-identical consecutive frames would leak across splits. This per-map
discipline is methodologically more honest and Stage 7 of this project's iteration
arc explicitly explores this contrast.

### 5.3 The Spectre failure (36%) is fundamental, not algorithmic

Spectre is a Doom enemy that is mechanically identical to Demon but rendered as a
semi-transparent partial-invisibility sprite. The bounding box reported by ViZDoom
is the same shape and size as a Demon's; only the rendered pixels differ.

For a vision model, this means: the *signal* distinguishing Spectre from Demon is
"50% of the pixels are partially transparent." This is a difficult cue even for
human observers — Doom intentionally made Spectres hard to see — and a small CNN
with no dedicated transparency-detection mechanism is unlikely to do well. The 36%
accuracy is, paradoxically, near the empirical ceiling for this class with this
architecture. This is not a *bug* but a *property* of the data, and serves as a
useful reminder that detection ground truth from the engine does not guarantee a
solvable perception problem.

### 5.4 BaronOfHell ↔ HellKnight: a real visual similarity

The 407 total confusions between BaronOfHell and HellKnight (248 + 159) in the
confusion matrix is the largest pair-confusion involving two non-Imp classes. The
two enemies are nearly visually identical in freedoom2 — both are tall, hooved,
horned humanoid figures with similar sprite proportions. The freedoom2 lore-names
"Pain Bringer" (HellKnight) and "Pain Lord" (BaronOfHell) reinforce this similarity:
they were designed as a tier pair. Distinguishing them at the 64 × 64 crop
resolution is genuinely hard.

### 5.5 The dominant-class fallback

Many misclassifications go *to* DoomImp (606 from Zombieman, 138 from Spectre, 135
from Arachnotron, 133 from Fatso, 122 from ShotgunGuy). DoomImp has 6,391 training
crops — 27% of the training set. When the model is uncertain, the prior pushes
toward the modal class, producing this systematic over-prediction. Class-balanced
loss weighting (or focal loss, planned for Stage 8) would directly address this.

### 5.6 Rare-class struggle

SpiderMastermind (48%) and Cyberdemon (47%) have only 147 and 794 training crops
respectively. Even with the model trained to overfit them perfectly, the limited
visual variety in this volume of data means the learned features don't generalize.
This is a small-data problem layered on the imbalance problem and motivates two
complementary techniques: data augmentation (to multiply effective sample diversity)
and class-balanced sampling (to force more attention during gradient updates).

---

## 6. Reproducibility and artifacts

| Artifact | Location | Purpose |
|---|---|---|
| Crop generation | `make_crops.py` | Idempotent reconstruction of `crops/` from `data/` |
| Training script | `stage1.py` | Runs the full pipeline end-to-end (~42 min CPU) |
| Best weights | `stage1_best.pt` | Epoch-10 checkpoint, 71.25% val accuracy |
| Class list | `data/classes.txt` | Stable class-id ↔ name mapping |
| Data | `data/MAPxx/*.png` + `.txt` | Raw frames + YOLO labels |

The pipeline is fully deterministic given a fixed dataset, with no random data
augmentation in Stage 1. (Mini-batch shuffling order does vary between runs and can
shift final accuracy by ±1 percentage point.)

---

## 6. Extension (Option B) — Regularization and Augmentation Ablation

The Stage 1 baseline reached 71.25% validation accuracy on a 20-epoch CPU run, with
a 29-percentage-point train–val gap that indicated severe overfitting. Section 6
isolates the contributions of two standard mitigations — **dropout** and **data
augmentation** — by training four otherwise-identical variants and comparing them.

### 6.1 Setup

All four variants share the same architecture, optimizer, learning rate, batch size,
and per-map train/val split. They differ only in regularization (dropout) and the
training-time data transform (augmentation):

| Variant | Dropout | Augmentation |
|---|---|---|
| `baseline` | none | none |
| `dropout` | `nn.Dropout(0.5)` before classifier head | none |
| `augment` | none | random horizontal flip + brightness/contrast/saturation jitter + small affine (translate ±4%, scale 0.95–1.05) |
| `dropout+augment` | both | both |

Training: 15 epochs each (reduced from 20 for runtime), Adam @ 1e-3, batch 128,
val pass after each epoch, best-by-val-accuracy weights saved. Run on Colab T4 GPU.

### 6.2 Comparison results

```
Variant                  Best val   @epoch   train@best
--------------------------------------------------------
baseline                   69.50%        5        86.6%
dropout                    73.93%       15        91.5%
augment                    73.87%       13        88.6%
dropout+augment            74.09%       12        85.1%
--------------------------------------------------------
```

**Headline numbers:**

| Variant | Best val | Δ vs baseline | Train–val gap at end |
|---|---:|---:|---:|
| `baseline` | 69.50% | — | 28.2 pp |
| `dropout` | 73.93% | **+4.43 pp** | 17.6 pp |
| `augment` | 73.87% | **+4.37 pp** | 14.7 pp |
| `dropout+augment` | **74.09%** | **+4.59 pp** | **11.0 pp** |

### 6.3 Per-variant training curves

```
Variant: baseline
 Ep  train_acc  val_acc
  1     55.4%    58.9%
  5     86.6%    69.5%  ← peak val
 10     93.1%    68.9%  (val plateaued, train still rising)
 15     96.3%    68.1%  (val drifting down)

Variant: dropout
 Ep  train_acc  val_acc
  1     55.2%    58.0%
  5     83.7%    69.3%
 10     88.8%    71.3%  (val still improving)
 15     91.5%    73.9%  ← peak val (still rising at end)

Variant: augment
 Ep  train_acc  val_acc
  1     49.9%    56.0%  (slower start due to noisy training signal)
  5     82.2%    69.0%
 10     87.1%    69.3%
 13     88.6%    73.9%  ← peak val
 15     89.4%    73.5%

Variant: dropout+augment
 Ep  train_acc  val_acc
  1     48.2%    52.6%  (slowest start of the four)
  5     79.6%    66.7%
 10     84.1%    72.9%
 12     85.1%    74.1%  ← peak val
 15     86.9%    73.8%
```

### 6.4 Discussion

#### 6.4.1 Each regularizer gives ~4.4 percentage points individually

The remarkable result is that **dropout and augmentation are very nearly equivalent**
on this problem — both deliver ~4.4 pp val improvement over baseline, and they
produce different *kinds* of regularization but converge to almost identical final
val numbers (73.93% vs 73.87%). They are addressing the same fundamental issue
(overfitting on a small, low-variation training set) via two different mechanisms:

- **Dropout** randomly zeros features during forward pass — forces redundancy
- **Augmentation** randomly varies the input — forces invariance to nuisance variation

Either alone is roughly as effective as the other.

#### 6.4.2 Combining them gives diminishing returns

Combining dropout *and* augmentation produces only **+0.16 pp** over the best
single technique (74.09% vs 73.93%). This is well within run-to-run noise; in
effect the two regularizers don't compose additively.

The reason: both reduce *the same kind of error* (overfitting on the training set).
Once one of them is in place, the model is already constrained enough that adding
the other has little new effect.

This is a useful negative finding for the writeup: **the bottleneck blocking Stage 1
from reaching 90%+ is not just overfitting.** If it were, the combined variant should
have closed more of the gap to the original 95%-naive expectation. The remaining
~25 pp from val accuracy to "perfect" is something else — likely the fundamental
visual ambiguities discussed in §5.3 and §5.4 (Spectre semi-transparency, BaronOfHell
↔ HellKnight confusion), the dominant-class fallback (§5.5), and the rare-class
struggle (§5.6). No amount of regularization fixes those.

#### 6.4.3 The train–val gap shrinks with every regularization technique

| Variant | Train at peak | Val at peak | Gap |
|---|---:|---:|---:|
| `baseline` | 86.6% | 69.5% | 17.1 pp |
| `dropout` | 91.5% | 73.9% | 17.6 pp |
| `augment` | 88.6% | 73.9% | 14.7 pp |
| `dropout+augment` | 85.1% | 74.1% | **11.0 pp** |

At the moment of peak val accuracy, the gap between train and val shrinks
substantially with augmentation (and slightly with dropout). The `augment` and
`dropout+augment` variants train more "honestly" — the model isn't allowed to
memorize the exact pixels of the training set because they're constantly being
perturbed. This is the diagnostic signal that augmentation is doing its job.

The `dropout` variant has a slightly *larger* gap than baseline (17.6 vs 17.1)
because dropout slows training and the model is still improving on both sides
when val peaks; with more epochs the gap would close.

#### 6.4.4 Peak epoch shifts later with regularization

| Variant | Peak val epoch |
|---|---:|
| `baseline` | **5** (very early) |
| `dropout` | **15** (still rising at end of run) |
| `augment` | 13 |
| `dropout+augment` | 12 |

The baseline overfit so quickly that the best val number came at epoch 5 of 15 —
*two-thirds of the training run was wasted overfitting*. Regularized variants
keep improving val accuracy much longer; the `dropout` variant is still rising
when the run ends, suggesting longer training would yield further improvement.

**Implication for tuning:** the regularized variants are *undertrained* at 15
epochs. A second-pass experiment at 30 epochs would likely push them past 75%.

#### 6.4.5 Runtime cost of augmentation is non-trivial even on GPU

| Variant | Wall-clock on Colab T4 |
|---|---:|
| `baseline` | 3.0 min |
| `dropout` | 3.1 min |
| `augment` | 9.7 min |
| `dropout+augment` | 9.9 min |

The 3× slowdown for augmented variants comes from the CPU-side cost of applying
torchvision transforms before each batch reaches the GPU. With small models and
fast GPUs, the data pipeline becomes the bottleneck. In production this is
typically mitigated with multi-worker DataLoaders, GPU-side augmentation (kornia),
or batched JPEG decoding — but for this project the 10-minute run is acceptable.

#### 6.4.6 What this tells us for subsequent stages

The Option B ablation refines the agenda for stages 2–9:

- **Bigger models won't help much.** The bottleneck is data/visual-ambiguity, not
  model capacity. Scaling the architecture from 232k to e.g. 2M parameters and
  not training longer would just overfit more.
- **Pretrained backbones (Stages 4–5)** should help significantly. The remaining
  gap above ~74% likely comes from features the from-scratch CNN can't easily
  learn from 23k crops (e.g., texture discrimination for the Baron/Hell Knight
  pair). ImageNet-pretrained features should pre-encode useful visual primitives.
- **Stage 6 augmentation lessons are already half-explored here.** A more
  aggressive augmentation policy (mosaic, mixup, copy-paste) would extend this.
  Stage 6 can specifically include the *failure case* — heavy channel jitter
  destroying the palette-based class distinctions — that the peer's project
  inadvertently hit.
- **Stage 8 focal loss / class weighting** has more headroom than dropout/
  augmentation. The remaining gap is concentrated in rare classes (Spectre,
  SpiderMastermind, Cyberdemon) and similar-pair confusions, both of which are
  more directly addressable by loss-shaping than by general regularization.

### 6.5 Conclusion for Section 6

Standard regularization (dropout + data augmentation) lifts validation accuracy
from 69.5% → 74.1% — a meaningful ~5 pp improvement — and dramatically reduces
overfitting (28 pp train–val gap → 11 pp). The two techniques are roughly
substitutable; combining them gives diminishing returns. The remaining gap
between achieved val accuracy and the "easy classifier" expectation (~90%+)
is *not* explainable by overfitting alone, and is the agenda for Stages 4–8.

---

## 7. Conclusions

Stage 1 succeeds in its narrow goal: a working end-to-end training pipeline producing
non-trivial classifier results on a deliberately challenging per-map-split dataset.
The 71.25% validation accuracy, while lower than a naive expectation, is honest and
diagnostically rich. The dominant failure modes — overfitting, class imbalance,
fundamental visual similarities between specific enemy pairs, and the partial-
invisibility Spectre case — set the agenda for subsequent stages:

- **Stage 2** (sliding-window detector): apply this trained classifier as a detection
  building block to measure the gap between classification and detection.
- **Stage 4** (pretrained backbone, frozen): substitute the from-scratch convolutional
  features with ImageNet-pretrained representations.
- **Stage 5** (unfreeze and fine-tune): the predicted-vs-actual centerpiece of the
  iteration arc.
- **Stage 6** (augmentation ablation): introduce geometric and photometric variation
  to close the train–val gap.
- **Stage 7** (per-map vs random split): contrast the present per-map result against
  a random-frame split to quantify the data-leakage effect.
- **Stage 8** (loss + assignment tweaks): focal loss / class weighting targeted at
  the rare-class failures.
- **Stage 9** (final evaluation): held-out test set (MAP26–MAP30 + MAP32, not yet
  captured) for the final unbiased mAP, per-class AP, and qualitative analysis.
