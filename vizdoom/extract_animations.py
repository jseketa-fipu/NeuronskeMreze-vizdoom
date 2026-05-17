"""Extract every animation sprite per enemy class from freedoom2.wad.

For each class:
  sprites/animations/<ClassName>/<SpriteName>.png   — every individual sprite (transparent)
  sprites/animations/<ClassName>_sheet.png          — composite grid of all sprites
"""
import os
from pathlib import Path
import numpy as np
import omg
import vizdoom as vzd
from PIL import Image, ImageDraw, ImageFont

SPRITE_PREFIXES = {
    "Zombieman":        "POSS",
    "ShotgunGuy":       "SPOS",
    "ChaingunGuy":      "CPOS",
    "DoomImp":          "TROO",
    "Demon":            "SARG",
    "Spectre":          "SARG",
    "LostSoul":         "SKUL",
    "Cacodemon":        "HEAD",
    "Fatso":            "FATT",
    "HellKnight":       "BOS2",
    "BaronOfHell":      "BOSS",
    "Arachnotron":      "BSPI",
    "PainElemental":    "PAIN",
    "Revenant":         "SKEL",
    "Archvile":         "VILE",
    "SpiderMastermind": "SPID",
    "Cyberdemon":       "CYBR",
}

OUT_DIR = Path("sprites") / "animations"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def mask_doom_transparency(rgba_img):
    arr = np.array(rgba_img)
    pink_mask = (arr[..., 0] > 200) & (arr[..., 1] < 80) & (arr[..., 2] > 200)
    arr[pink_mask, 3] = 0
    return Image.fromarray(arr)


def find_font():
    for fp in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if os.path.exists(fp):
            return ImageFont.truetype(fp, 11)
    return ImageFont.load_default()


wad = omg.WAD()
wad.from_file(os.path.join(os.path.dirname(vzd.__file__), "freedoom2.wad"))
font = find_font()

# Group sprite names by prefix to avoid scanning all sprites repeatedly.
all_names = list(wad.sprites.keys())

total_individual = 0
for class_name, prefix in SPRITE_PREFIXES.items():
    names = sorted([n for n in all_names if n.startswith(prefix)])
    if not names:
        print(f"  {class_name:18s} NO SPRITES")
        continue

    # 1. Save individual sprite PNGs.
    class_dir = OUT_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    for sname in names:
        sprite = wad.sprites[sname]
        img = mask_doom_transparency(sprite.to_Image().convert("RGBA"))
        # Doom uses ASCII chars after Z ('[', '\\', ']', '^', '_') as extended
        # frame letters; they're invalid in filenames. Map to ord-coded names.
        safe = "".join(c if c.isalnum() else f"_{ord(c):03d}_" for c in sname)
        img.save(class_dir / f"{safe}.png")
    total_individual += len(names)

    # 2. Build composite sheet: grid laid out alphabetically, fixed cell size.
    CELL = 96             # square cell, including padding for label
    COLS = 8              # columns of sprites
    LABEL_H = 14
    rows = (len(names) + COLS - 1) // COLS
    sheet_w = COLS * CELL
    sheet_h = rows * CELL + 20            # +20 for title bar
    sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 4), f"{class_name}  ({prefix} — {len(names)} sprites)",
              fill=(220, 220, 220), font=font)

    for i, sname in enumerate(names):
        col = i % COLS
        row = i // COLS
        cell_x = col * CELL
        cell_y = row * CELL + 20

        # Subtle cell border.
        draw.rectangle((cell_x, cell_y, cell_x + CELL - 1, cell_y + CELL - 1),
                       outline=(55, 55, 55), width=1)

        sprite = wad.sprites[sname]
        img = mask_doom_transparency(sprite.to_Image().convert("RGBA"))
        max_w = CELL - 6
        max_h = CELL - LABEL_H - 6
        scale = min(max_w / img.width, max_h / img.height, 2.0)
        if scale != 1.0:
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            img = img.resize(new_size,
                             Image.NEAREST if scale > 1 else Image.LANCZOS)
        img_x = cell_x + (CELL - img.width) // 2
        img_y = cell_y + (CELL - LABEL_H - img.height) // 2 + 2
        sheet.paste(img, (img_x, img_y), img)

        # Sprite-name label under the image.
        text_bbox = draw.textbbox((0, 0), sname, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_x = cell_x + (CELL - text_w) // 2
        text_y = cell_y + CELL - LABEL_H - 1
        draw.text((text_x, text_y), sname, fill=(180, 180, 180), font=font)

    sheet_path = OUT_DIR / f"{class_name}_sheet.png"
    sheet.save(sheet_path)
    print(f"  {class_name:18s} {len(names):3d} sprites -> {sheet_path}")

print(f"\nTotal individual sprite files: {total_individual}")
print(f"Output: {OUT_DIR.resolve()}")
