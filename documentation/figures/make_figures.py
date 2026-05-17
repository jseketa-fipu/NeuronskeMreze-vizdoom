"""Generate all figures for the final project writeup.

Produces six PNGs in this directory:
- fig1_map_progression.png       Bar chart, mAP across all stages
- fig2_training_curves.png       Val mAP per epoch, stages 3/5/6/8 overlaid
- fig3_val_vs_test_per_class.png Stage 6 val vs Stage 9 test, per class
- fig4_per_test_map.png          Stage 9 per-test-map mAP
- fig5_split_methodology.png     Random-split / per-map val / per-map test
- fig6_time_breakdown.png        Wall-clock per stage
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
})

C_GOOD     = "#2a7e3b"
C_BAD      = "#b03a3a"
C_NEUTRAL  = "#4d6b9a"
C_BEST     = "#d99000"
C_TEST     = "#5b4080"


# ---------- Fig 1: mAP progression across stages ----------

stages   = ["S2\nsliding\nwindow", "S3\nfrom-scratch\nYOLO", "S4\nfrozen\npretrained",
            "S5\nfine-tuned\npretrained", "S6\n+ light\naugmentation",
            "S8\n+ focal\nloss", "S9\nFINAL\ntest"]
maps     = [4.30, 21.12, 5.94, 30.62, 33.89, 32.90, 24.21]
colors   = [C_NEUTRAL, C_NEUTRAL, C_BAD, C_NEUTRAL, C_BEST, C_BAD, C_TEST]
labels   = ["baseline", "from-scratch", "negative", "big lift", "best val", "negative", "test"]

fig, ax = plt.subplots(figsize=(10.5, 5.2))
bars = ax.bar(range(len(stages)), maps, color=colors, edgecolor="black", linewidth=0.6, width=0.7)
for i, (b, v, lab) in enumerate(zip(bars, maps, labels)):
    ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(b.get_x() + b.get_width()/2, -2.4, lab, ha="center", va="top", fontsize=8.5, style="italic", color="#444")

ax.axhline(33.89, color=C_BEST, linestyle=":", linewidth=1, alpha=0.7)
ax.text(len(stages) - 0.5, 33.89 + 0.4, "val ceiling 33.89%", color=C_BEST, fontsize=8.5, ha="right")

ax.set_xticks(range(len(stages)))
ax.set_xticklabels(stages, fontsize=9)
ax.set_ylabel("mAP @ IoU=0.5 (%)")
ax.set_title("mAP progression across the project arc", fontweight="bold", pad=12)
ax.set_ylim(-6, 42)

legend = [
    mpatches.Patch(color=C_NEUTRAL, label="val (intermediate)"),
    mpatches.Patch(color=C_BEST, label="val (best)"),
    mpatches.Patch(color=C_BAD, label="negative result"),
    mpatches.Patch(color=C_TEST, label="test (final, held-out)"),
]
ax.legend(handles=legend, loc="upper left", frameon=False, fontsize=9)
plt.savefig("/home/yoshi/neuronskemreze/stages/figures/fig1_map_progression.png")
plt.close()


# ---------- Fig 2: Training curves ----------

curves = {
    "Stage 3 — from-scratch YOLO":  ([1, 4, 7, 10, 15, 20, 30, 40, 50], [0.67, 11.27, 17.72, 19.76, 20.18, 19.95, 17.51, 16.19, 17.39]),
    "Stage 4 — frozen pretrained":  ([1, 5, 10, 15, 20, 25, 30],         [2.15, 4.62, 3.70, 6.13, 4.07, 3.53, 4.69]),
    "Stage 5 — fine-tuned":         ([1, 3, 5, 10, 17, 23, 30],          [22.72, 28.39, 28.88, 29.60, 30.71, 30.87, 26.99]),
    "Stage 6 — + augmentation":     ([1, 3, 5, 8, 14, 22, 33, 40],       [20.78, 26.31, 33.16, 33.47, 30.23, 32.58, 31.27, 31.41]),
    "Stage 8 — + focal loss":       ([1, 5, 10, 20, 30, 40],              [15.21, 29.85, 30.52, 34.80, 31.98, 28.51]),
}
curve_colors = ["#888888", C_BAD, "#4d6b9a", C_BEST, "#9a4d8a"]

fig, ax = plt.subplots(figsize=(10.5, 5.5))
for (name, (xs, ys)), c in zip(curves.items(), curve_colors):
    ax.plot(xs, ys, marker="o", markersize=5, linewidth=1.8, color=c, label=name)
ax.axhline(33.89, color=C_BEST, linestyle=":", linewidth=1, alpha=0.6)
ax.text(50, 33.89 + 0.5, "Stage 6 best 33.89%", color=C_BEST, fontsize=8.5, ha="right", style="italic")
ax.axhline(24.21, color=C_TEST, linestyle=":", linewidth=1, alpha=0.6)
ax.text(50, 24.21 + 0.5, "Stage 9 final test 24.21%", color=C_TEST, fontsize=8.5, ha="right", style="italic")

ax.set_xlabel("Epoch")
ax.set_ylabel("Val mAP @ IoU=0.5 (%)")
ax.set_title("Training dynamics across all detector stages", fontweight="bold", pad=12)
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.set_xlim(0, 52)
ax.set_ylim(0, 40)
plt.savefig("/home/yoshi/neuronskemreze/stages/figures/fig2_training_curves.png")
plt.close()


# ---------- Fig 3: Per-class val vs test ----------

class_names = ["Zombieman", "ShotgunGuy", "ChaingunGuy", "DoomImp", "Demon", "Spectre",
               "LostSoul", "Cacodemon", "Fatso", "HellKnight", "Arachnotron",
               "PainElemental", "Revenant", "BaronOfHell", "Archvile",
               "SpiderMastermind", "Cyberdemon"]
val_s6   = [14.33, 37.49, 47.74, 38.32, 37.98, 0.45, 34.29, 41.88, 32.64, 44.12,
            31.68, 27.79, 35.18, 43.35, 51.94, 26.14, 30.75]
test_s9  = [23.42, 29.58, 15.24, 19.85, 28.04, 3.64, 39.51, 34.56, 12.68, 35.97,
            14.02, 15.64, 24.32, 17.22, 31.85, None, 41.90]

# sort by val desc for readability
order = sorted(range(len(class_names)), key=lambda i: -val_s6[i])
names_s = [class_names[i] for i in order]
val_s   = [val_s6[i] for i in order]
test_s  = [test_s9[i] if test_s9[i] is not None else 0 for i in order]
test_missing = [test_s9[i] is None for i in order]

x = np.arange(len(names_s))
w = 0.38
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.bar(x - w/2, val_s,  w, color=C_BEST, label="Stage 6 — val",  edgecolor="black", linewidth=0.4)
ax.bar(x + w/2, test_s, w, color=C_TEST, label="Stage 9 — test", edgecolor="black", linewidth=0.4)
for i, missing in enumerate(test_missing):
    if missing:
        ax.text(i + w/2, 1.0, "n/a", ha="center", va="bottom", fontsize=7, color="#555", style="italic")
ax.set_xticks(x)
ax.set_xticklabels(names_s, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("AP @ IoU=0.5 (%)")
ax.set_title("Per-class AP: Stage 6 (val) vs Stage 9 (test)", fontweight="bold", pad=12)
ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.set_ylim(0, 60)
plt.savefig("/home/yoshi/neuronskemreze/stages/figures/fig3_val_vs_test_per_class.png")
plt.close()


# ---------- Fig 4: Per-test-map mAP ----------

map_names = ["MAP26", "MAP27", "MAP28", "MAP32", "MAP29", "MAP30"]
map_maps  = [32.04, 32.27, 32.78, 28.26, 19.82, 17.27]
map_colors = ["#5b4080"] * 4 + [C_BAD] * 2

fig, ax = plt.subplots(figsize=(9, 4.6))
bars = ax.bar(map_names, map_maps, color=map_colors, edgecolor="black", linewidth=0.5, width=0.6)
for b, v in zip(bars, map_maps):
    ax.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax.axhline(33.89, color=C_BEST, linestyle=":", linewidth=1)
ax.text(5.4, 33.89 + 0.4, "val mAP 33.89%", color=C_BEST, fontsize=8.5, ha="right", style="italic")
ax.axhline(24.21, color=C_TEST, linestyle="--", linewidth=1)
ax.text(5.4, 24.21 + 0.4, "overall test 24.21%", color=C_TEST, fontsize=8.5, ha="right", style="italic")
ax.set_ylabel("mAP @ IoU=0.5 (%)")
ax.set_title("Per-test-map breakdown — the test set is bimodal", fontweight="bold", pad=12)
ax.set_ylim(0, 40)
plt.savefig("/home/yoshi/neuronskemreze/stages/figures/fig4_per_test_map.png")
plt.close()


# ---------- Fig 5: Split methodology ----------

method_names = ["Random-split val\n(LEAKY)", "Per-map val\n(model-selection metric)", "Per-map test\n(honest, final)"]
method_maps  = [49.58, 33.89, 24.21]
method_colors = ["#b03a3a", C_BEST, C_TEST]

fig, ax = plt.subplots(figsize=(8.5, 5.0))
bars = ax.bar(method_names, method_maps, color=method_colors, edgecolor="black", linewidth=0.6, width=0.6)
for b, v in zip(bars, method_maps):
    ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.2f}%", ha="center", fontsize=11, fontweight="bold")
ax.annotate("", xy=(2, 24.21), xytext=(0, 49.58),
            arrowprops=dict(arrowstyle="->", color="#444", lw=1.2, connectionstyle="arc3,rad=-0.25"))
ax.text(1, 42, "−25.4 pp from\nleaky → honest", ha="center", fontsize=9.5, style="italic", color="#444")
ax.set_ylabel("mAP @ IoU=0.5 (%)")
ax.set_title("Same model, three different ways to evaluate it", fontweight="bold", pad=12)
ax.set_ylim(0, 60)
plt.savefig("/home/yoshi/neuronskemreze/stages/figures/fig5_split_methodology.png")
plt.close()


# ---------- Fig 6: Time per stage ----------

t_stages = ["S1\nclassifier", "S2\nsliding\nwindow", "S3\nfrom-scratch\nYOLO",
            "S4\nfrozen", "S5\nfine-tuned", "S6\n+ aug", "S7\nsplit\ncontrast",
            "S8\nfocal", "S9\nfinal\ntest"]
t_minutes = [42.1, 13.2, 165.0, 50.0, 75.0, 120.0, 5.0, 140.0, 2.6]
t_colors = ["#4d6b9a"] * 9
t_colors[3] = C_BAD; t_colors[5] = C_BEST; t_colors[7] = C_BAD; t_colors[8] = C_TEST

fig, ax = plt.subplots(figsize=(10.5, 4.6))
bars = ax.bar(t_stages, t_minutes, color=t_colors, edgecolor="black", linewidth=0.5, width=0.7)
for b, v in zip(bars, t_minutes):
    h, m = divmod(int(v), 60)
    lab = f"{h}h{m:02d}m" if h else f"{int(v)}m"
    ax.text(b.get_x() + b.get_width()/2, v + 3, lab, ha="center", fontsize=9, fontweight="bold")
total = sum(t_minutes)
ax.set_ylabel("Wall-clock training/eval time (minutes)")
ax.set_title(f"Compute time per stage (total ≈ {int(total)//60}h {int(total)%60}m of training + eval)",
             fontweight="bold", pad=12)
ax.set_ylim(0, 200)
plt.savefig("/home/yoshi/neuronskemreze/stages/figures/fig6_time_breakdown.png")
plt.close()

print("All figures written to stages/figures/")
