"""Count enemy instances per class across all saved YOLO label files."""
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data")

with open(DATA_DIR / "classes.txt") as f:
    CLASS_NAMES = [line.strip() for line in f
                   if line.strip() and not line.startswith("#")]

counts = Counter()
files = 0
for txt_path in DATA_DIR.glob("MAP*/*.txt"):
    files += 1
    with open(txt_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            counts[cid] += 1

print(f"{'ID':>3}  {'Class':<20} {'Count':>7}")
print("-" * 33)
total = 0
for i, name in enumerate(CLASS_NAMES):
    print(f"{i:>3}  {name:<20} {counts[i]:>7}")
    total += counts[i]
print("-" * 33)
print(f"{'':>3}  {'TOTAL':<20} {total:>7}")
print(f"{'':>3}  {'label files':<20} {files:>7}")
