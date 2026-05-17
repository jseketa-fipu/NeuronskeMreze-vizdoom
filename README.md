# NeuronskeMreze-vizdoom

YOLO-style enemy detector trained on Freedoom 2 gameplay frames. The headline tool is `compare.py`, a spectator-mode viewer that overlays engine ground-truth boxes (green) against model predictions (cyan) so you can see what the network actually learned.

![compare.py — engine truth vs. model predictions, side-by-side with the live Doom window](documentation/figures/compare_screenshot.png)

*Left: `compare.py` showing MAP01 with two Zombiemen detected at 100% and 95% confidence, plus engine ground-truth boxes, enemy roster, and rotating minimap. Right: the unmodified ViZDoom window.*

## Setup

**Requires Python 3.11.** Allow ~1 GB for the venv.

```powershell
# 1. Create and activate the venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # PowerShell (Windows)
# .venv\Scripts\activate.bat        # cmd.exe (Windows)
# source .venv/bin/activate         # bash / zsh (macOS / Linux)

# 2. Install dependencies (CPU PyTorch)
pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple `
    torch torchvision opencv-python numpy omgifol vizdoom
```

For a CUDA build of PyTorch, swap `whl/cpu` for `whl/cu121` (or `whl/cu118` for older drivers).

## Running `compare.py`

Run **from the repo root** (so `stage3.py` finds `data/classes.txt`) and point `WEIGHTS` at the checkpoint you want to inspect:

```powershell
$env:WEIGHTS = "stages\stage6_best.pt"
python stages\compare.py
```

On macOS / Linux:

```bash
WEIGHTS=stages/stage6_best.pt python stages/compare.py
```

### In-game controls

| Key | Action |
|-----|--------|
| `q` | quit |
| `n` | next map |
| `m` | toggle minimap (local rotating ↔ full-map) |
| `p` | pause / resume model inference |
| `[` / `]` | lower / raise confidence threshold |

### Useful environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `WEIGHTS` | `stage6_best.pt` | Which checkpoint to load. Try `stages\stage8_best.pt` for a different one. |
| `ONLY_MAPS` | all 32 | Comma-separated list, e.g. `ONLY_MAPS=MAP17,MAP31`. |
| `INFER_EVERY` | `2` | Run model every N ticks. Raise on slow CPUs. |
