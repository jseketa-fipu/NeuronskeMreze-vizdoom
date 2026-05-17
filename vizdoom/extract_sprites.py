"""Extract enemy sprite reference images from freedoom2.wad.

Per ENEMY_CLASSES, finds the front-facing idle sprite (prefix + 'A1' / 'A0'),
saves an individual transparent PNG per class plus a composite grid showcase.
"""
import os
from pathlib import Path
import numpy as np
import omg
import vizdoom as vzd
from PIL import Image, ImageDraw, ImageFont


def mask_doom_transparency(rgba_img):
    """Doom's transparency-key color is magenta (palette index 247). PIL doesn't
    know it's meant to be transparent — appears as solid magenta until we mask it.
    Replace exact-magenta pixels with alpha=0."""
    arr = np.array(rgba_img)
    pink_mask = (arr[..., 0] > 200) & (arr[..., 1] < 80) & (arr[..., 2] > 200)
    arr[pink_mask, 3] = 0
    return Image.fromarray(arr)

# ENEMY_CLASSES → 4-letter sprite prefix (canonical Doom 2 sprite codes).
SPRITE_PREFIXES = {
    "Zombieman":        "POSS",
    "ShotgunGuy":       "SPOS",
    "ChaingunGuy":      "CPOS",
    "DoomImp":          "TROO",
    "Demon":            "SARG",
    "Spectre":          "SARG",   # same sprite as Demon; rendered semi-transparent at runtime
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

OUT_DIR = Path("sprites")
OUT_DIR.mkdir(exist_ok=True)

wad = omg.WAD()
wad.from_file(os.path.join(os.path.dirname(vzd.__file__), "freedoom2.wad"))

def find_sprite(prefix):
    """Find a usable front-facing sprite for an actor.
    Tries A1 (idle frame A, rotation 1=south), then A0 (omnidirectional),
    then B1/B0, then any sprite starting with the prefix."""
    for frame in "AB":
        for rot in "10":
            name = f"{prefix}{frame}{rot}"
            if name in wad.sprites:
                return name
    for name in wad.sprites.keys():
        if name.startswith(prefix):
            return name
    return None


individual_paths = {}
print("Extracting individual sprites:")
for class_name, prefix in SPRITE_PREFIXES.items():
    name = find_sprite(prefix)
    if name is None:
        print(f"  {class_name:18s} NOT FOUND ({prefix}?)")
        continue
    sprite = wad.sprites[name]
    img = sprite.to_Image().convert("RGBA")
    img = mask_doom_transparency(img)
    out = OUT_DIR / f"{class_name}.png"
    img.save(out)
    individual_paths[class_name] = out
    w, h = sprite.dimensions
    print(f"  {class_name:18s} {name:8s} {w}x{h} -> {out}")

# ---- Composite showcase grid ----
CELL = 220
COLS = 4
ROWS = (len(SPRITE_PREFIXES) + COLS - 1) // COLS
W = COLS * CELL
H = ROWS * CELL

composite = Image.new("RGB", (W, H), (30, 30, 30))
draw = ImageDraw.Draw(composite)

# Try a TrueType font with antialiasing; fall back to PIL's bitmap default.
font_paths = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
font = None
for fp in font_paths:
    if os.path.exists(fp):
        font = ImageFont.truetype(fp, 16)
        break
if font is None:
    font = ImageFont.load_default()

for i, class_name in enumerate(SPRITE_PREFIXES.keys()):
    col = i % COLS
    row = i // COLS
    cell_x = col * CELL
    cell_y = row * CELL

    # Cell separator (subtle border)
    draw.rectangle((cell_x, cell_y, cell_x + CELL - 1, cell_y + CELL - 1),
                   outline=(60, 60, 60), width=1)

    if class_name in individual_paths:
        sprite_img = Image.open(individual_paths[class_name]).convert("RGBA")
        label_height = 32
        max_w = CELL - 16
        max_h = CELL - label_height - 16
        # Scale up small sprites for visibility; NEAREST keeps pixel-art crisp.
        scale = min(max_w / sprite_img.width, max_h / sprite_img.height, 3.0)
        if scale > 1:
            new_size = (int(sprite_img.width * scale), int(sprite_img.height * scale))
            sprite_img = sprite_img.resize(new_size, Image.NEAREST)
        elif scale < 1:
            new_size = (int(sprite_img.width * scale), int(sprite_img.height * scale))
            sprite_img = sprite_img.resize(new_size, Image.LANCZOS)
        img_x = cell_x + (CELL - sprite_img.width) // 2
        img_y = cell_y + (CELL - label_height - sprite_img.height) // 2 + 4
        composite.paste(sprite_img, (img_x, img_y), sprite_img)

    # Class label at bottom
    text_bbox = draw.textbbox((0, 0), class_name, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_x = cell_x + (CELL - text_w) // 2
    text_y = cell_y + CELL - 24
    draw.text((text_x, text_y), class_name, fill=(220, 220, 220), font=font)

composite_path = OUT_DIR / "_showcase.png"
composite.save(composite_path)
print(f"\nComposite showcase ({W}x{H} px): {composite_path}")
print(f"Individual sprites: {len(individual_paths)}/17 in {OUT_DIR.resolve()}")
