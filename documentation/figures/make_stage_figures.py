"""Generate one standalone training-curve PNG per trained stage.

Detector stages (3-8): train loss (grey, left axis) + val mAP (colour, right
axis) per epoch, with a star at the best epoch. Stage 1: train vs val accuracy.
Data is the real per-epoch logs from the stage results docs.

Output: documentation/figures/stage{N}_curve.png
(Stages 2, 7, 9 are single-shot inference/eval — no training curve.)
"""
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140, "savefig.bbox": "tight",
    "font.size": 10, "axes.spines.top": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linestyle": "--", "axes.axisbelow": True,
})
C_NEUTRAL, C_BAD, C_BEST = "#4d6b9a", "#b03a3a", "#d99000"
LOSS_C = "#999999"

# detector stages: (file, title, color, epochs, loss, val_mAP, best_epoch)
DET = [
    ("stage3_curve", "Stage 3 — from-scratch YOLO", C_NEUTRAL,
     [1, 4, 7, 10, 15, 20, 30, 40, 50],
     [16.08, 8.98, 6.76, 4.99, 2.47, 1.22, 0.57, 0.37, 0.30],
     [0.67, 11.27, 17.72, 19.76, 20.18, 19.95, 17.51, 16.19, 17.39], 15),
    ("stage4_curve", "Stage 4 — frozen backbone (negative)", C_BAD,
     [1, 5, 10, 15, 20, 25, 30],
     [17.96, 11.99, 11.52, 11.29, 11.16, 11.08, 11.01],
     [2.15, 4.62, 3.70, 6.13, 4.07, 3.53, 4.69], 15),
    ("stage5_curve", "Stage 5 — fine-tuned backbone", C_NEUTRAL,
     [1, 3, 5, 10, 17, 23, 30],
     [10.34, 3.51, 1.63, 0.63, 0.37, 0.26, 0.20],
     [22.72, 28.39, 28.88, 29.60, 30.71, 30.87, 26.99], 23),
    ("stage6_curve", "Stage 6 — + augmentation (best)", C_BEST,
     [1, 3, 5, 8, 14, 22, 33, 40],
     [10.55, 4.61, 3.13, 1.90, 0.97, 0.61, 0.40, 0.33],
     [20.78, 26.31, 33.16, 33.47, 30.23, 32.58, 31.27, 31.41], 8),
    ("stage8_curve", "Stage 8 — + focal loss (negative)", C_BAD,
     [1, 5, 10, 20, 30, 40],
     [4.89, 1.28, 0.65, 0.26, 0.17, 0.13],
     [15.21, 29.85, 30.52, 34.80, 31.98, 28.51], 20),
]

for fname, title, color, ep, loss, mAP, best in DET:
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.plot(ep, loss, marker="o", markersize=4, color=LOSS_C, label="train loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss", color="#777777")
    ax.tick_params(axis="y", labelcolor="#777777")
    ax2 = ax.twinx()
    ax2.plot(ep, mAP, marker="s", markersize=4, color=color, label="val mAP")
    bi = ep.index(best)
    ax2.plot(best, mAP[bi], marker="*", markersize=18, color=color, zorder=5)
    ax2.annotate(f"best {mAP[bi]:.2f}% @ ep{best}", xy=(best, mAP[bi]),
                 xytext=(6, -2), textcoords="offset points", fontsize=8.5,
                 color=color, fontweight="bold")
    ax2.set_ylabel("val mAP @ IoU=0.5 (%)", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0, 40)
    ax.set_title(title, fontweight="bold", pad=10)
    plt.savefig(OUT / f"{fname}.png")
    plt.close()
    print(f"wrote {fname}.png")

# Stage 1: train vs val accuracy
fig, ax = plt.subplots(figsize=(6.4, 3.7))
ep = [1, 5, 10, 15]
ax.plot(ep, [55.4, 86.6, 93.1, 96.3], marker="o", color=LOSS_C, label="train accuracy")
ax.plot(ep, [58.9, 69.5, 68.9, 68.1], marker="o", color=C_NEUTRAL, label="val accuracy")
ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)"); ax.set_ylim(50, 100)
ax.set_title("Stage 1 — classifier accuracy (train vs val)", fontweight="bold", pad=10)
ax.legend(frameon=False, fontsize=9, loc="center right")
ax.annotate("train → 96%, val stalls ~69%  =  overfitting", xy=(10, 68.9),
            xytext=(2.2, 80), fontsize=9, color=C_BAD,
            arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1))
plt.savefig(OUT / "stage1_curve.png")
plt.close()
print("wrote stage1_curve.png")
