"""Pre-resize data/MAPxx/*.png → data_416/MAPxx/*.jpg for compact Colab upload.

Labels (.txt) are YOLO-format normalized [0,1] so they survive any resize — copied
unchanged. Reduces dataset from ~5 GB to ~1 GB. stage3.py reads either format.
"""
from pathlib import Path
import cv2

DATA_DIR = Path("data")
OUT_DIR = Path("data_416")
TARGET = 416
JPEG_QUALITY = 85

out_classes = OUT_DIR / "classes.txt"
in_classes = DATA_DIR / "classes.txt"
if in_classes.exists():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_classes.write_text(in_classes.read_text())

total = 0
for map_dir in sorted(DATA_DIR.glob("MAP*")):
    if not map_dir.is_dir():
        continue
    out_map = OUT_DIR / map_dir.name
    out_map.mkdir(parents=True, exist_ok=True)
    n = 0
    for png in sorted(map_dir.glob("*.png")):
        img = cv2.imread(str(png))
        if img is None:
            continue
        resized = cv2.resize(img, (TARGET, TARGET))
        out_jpg = out_map / (png.stem + ".jpg")
        cv2.imwrite(str(out_jpg), resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        txt = png.with_suffix(".txt")
        if txt.exists():
            (out_map / txt.name).write_text(txt.read_text())
        n += 1
    total += n
    print(f"{map_dir.name}: {n}")
print(f"\nTotal: {total} frames resized to {TARGET}×{TARGET} JPEG q={JPEG_QUALITY}")
print(f"Output: {OUT_DIR.resolve()}")
print(f"\nNext: zip -rq doom_data_416.zip data_416/ && upload to Drive/Colab")
