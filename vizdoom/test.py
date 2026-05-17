"""Manual-play data capture for Doom enemy detection.

Spectator mode: you play freedoom2 in the Doom window with normal keyboard/mouse.
Every Nth tick, if enemies are visible, the frame and YOLO-format labels are saved
to data/<MAP>/<index>.png + .txt.

Controls in the cv2 window:
  q - quit (does NOT mark current map done; resume here next run)
  n - mark current map done and move to next
  m - toggle minimap mode: local rotating (default) <-> full-map north-up

Maps marked done are skipped on subsequent runs. To re-capture a map,
delete its data/<MAP>/.done marker.
"""
import os
import math
import collections
import cv2
import numpy as np
import omg
import vizdoom as vzd

# Stable ordering — these list indices are the integer class IDs baked into every
# saved label file. Don't reorder once you've captured data.
ENEMY_CLASSES = [
    # ZDoom actor class names (what state.labels reports) with freedoom2's
    # redesigned lore-names as comments. Never reorder/insert — old label files
    # bake the index as the class ID.
    "Zombieman",       # freedoom: (handgun zombie)
    "ShotgunGuy",      # freedoom: Shotgun Zombie
    "ChaingunGuy",     # freedoom: Minigun Zombie
    "DoomImp",         # freedoom: Serpentipede
    "Demon",           # freedoom: Flesh Worm
    "Spectre",         # freedoom: Stealth Worm
    "LostSoul",        # freedoom: Hatchling
    "Cacodemon",       # freedoom: Trilobite
    # ---- appended after initial capture began (IDs 8+) ----
    "Fatso",           # freedoom: Combat Slug    (Mancubus)
    "HellKnight",      # freedoom: Pain Bringer
    "Arachnotron",     # freedoom: Technospider
    "PainElemental",   # freedoom: Matribite
    "Revenant",        # freedoom: Octaminator    ("octopus legs")
    "BaronOfHell",     # freedoom: Pain Lord
    "Archvile",        # freedoom: Necromancer  (lowercase 'v' — ZDoom convention)
    "SpiderMastermind",# freedoom: Large Technospider
    "Cyberdemon",      # freedoom: Assault Tripod
]
CLASS_TO_ID = {name: i for i, name in enumerate(ENEMY_CLASSES)}

# All 32 freedoom2 maps in numeric order. Maps marked .done are skipped on subsequent
# runs, so adding the full set is safe — already-captured maps just queue-pass.
MAPS = [f"MAP{i:02d}" for i in range(1, 33)]
# Optional override for targeted sessions, e.g.:
#   ONLY_MAPS=MAP31           python test.py
#   ONLY_MAPS=MAP17,MAP31     python test.py
# Listed maps are played even if their .done marker exists.
if "ONLY_MAPS" in os.environ:
    MAPS = [m.strip() for m in os.environ["ONLY_MAPS"].split(",") if m.strip()]

OUTPUT_DIR = "data"
# Save every Nth tick. At 35 ticrate, FRAME_SKIP=10 -> ~3.5 saved frames/sec.
# Set higher than the obvious "as fast as possible": consecutive ticks during
# stand-and-aim moments are near-duplicates (same camera, same enemy, only the
# sprite's animation frame differs). Near-duplicates inflate the dataset without
# adding training signal and can leak the same pose across train/val splits.
FRAME_SKIP = 10
# Drop bboxes smaller than this. 32 was too lax (4×8 px specks); 200 was too strict
# (mid-distance Imps got filtered). 100 ≈ 10×10 — keeps the speck filter but admits
# anything you can actually see and identify on screen.
MIN_BBOX_AREA = 100
# Per-map episode cap. Bumped from 10 to 30 minutes for longer exploration.
EPISODE_TIMEOUT_TICKS = 35 * 60 * 30

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "classes.txt"), "w") as f:
    f.write("# AUTO-GENERATED from ENEMY_CLASSES in test.py — do not edit by hand.\n")
    f.write("# This file is overwritten on every run. Edit ENEMY_CLASSES instead.\n")
    f.write("\n".join(ENEMY_CLASSES) + "\n")

game = vzd.DoomGame()
game.set_doom_game_path(os.path.join(vzd.scenarios_path, "freedoom2.wad"))
game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
game.set_screen_format(vzd.ScreenFormat.BGR24)
game.set_mode(vzd.Mode.ASYNC_SPECTATOR)  # human plays in the Doom window
game.set_window_visible(True)
game.set_labels_buffer_enabled(True)
game.set_episode_timeout(EPISODE_TIMEOUT_TICKS)
for btn in (vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD,
            vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT,
            vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT,
            vzd.Button.TURN_LEFT_RIGHT_DELTA,
            vzd.Button.ATTACK, vzd.Button.USE, vzd.Button.SPEED):
    game.add_available_button(btn)
game.add_game_args("+set sv_cheats 1")
game.add_game_args("+set use_mouse 0")  # mouse capture is broken in this setup; keyboard only
game.set_sound_enabled(True)            # ViZDoom defaults to mute; enable for music + SFX
game.add_game_args("+set snd_musicvolume 0.2")
game.add_game_args("+set snd_sfxvolume 0.08")
# Game variables needed for our custom minimap (ZDoom's automap UI is suppressed
# in ViZDoom's spectator mode, so we render our own).
game.add_available_game_variable(vzd.GameVariable.POSITION_X)
game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
game.add_available_game_variable(vzd.GameVariable.ANGLE)
# All actors in the level (not just visible ones) — so the minimap shows enemies
# in adjacent rooms, not only those in our line of sight.
game.set_objects_info_enabled(True)
game.init()

W = game.get_screen_width()
H = game.get_screen_height()

# ---- WAD parsing for minimap geometry ----
# freedoom2.wad is shipped inside the vizdoom package, not in scenarios_path.
WAD = omg.WAD()
WAD.from_file(os.path.join(os.path.dirname(vzd.__file__), "freedoom2.wad"))

# Doom 2 linedef "action" specials we color on the minimap.
#   Locked doors (blue/red/yellow) — both keycard and skull variants use the same color.
#     Blue:   26, 32 (player open), 99, 133 (switch open, fast)
#     Red:    28, 33 (player open), 134, 135 (switch open, fast)
#     Yellow: 27, 34 (player open), 136, 137 (switch open, fast)
#   Exits — green for normal level exit, magenta for secret exit.
#     Normal exit: 11 (switch), 52 (walk-over)
#     Secret exit: 51 (switch), 124 (walk-over)
SPECIAL_LINE_COLORS = {
    26: (255, 0, 0), 32: (255, 0, 0), 99: (255, 0, 0), 133: (255, 0, 0),
    28: (0, 0, 255), 33: (0, 0, 255), 134: (0, 0, 255), 135: (0, 0, 255),
    27: (0, 255, 255), 34: (0, 255, 255), 136: (0, 255, 255), 137: (0, 255, 255),
    11: (0, 255, 0), 52: (0, 255, 0),
    51: (255, 0, 255), 124: (255, 0, 255),
}
EXIT_LINE_SPECIALS = {11, 51, 52, 124}

# Doom 2 monster Thing-type IDs → ENEMY_CLASSES names.
MONSTER_TYPES = {
    7: "SpiderMastermind", 9: "ShotgunGuy", 16: "Cyberdemon", 58: "Spectre",
    64: "Archvile", 65: "ChaingunGuy", 66: "Revenant", 67: "Fatso",
    68: "Arachnotron", 69: "HellKnight", 71: "PainElemental",
    3001: "DoomImp", 3002: "Demon", 3003: "BaronOfHell",
    3004: "Zombieman", 3005: "Cacodemon", 3006: "LostSoul",
}

def load_map_data(map_name):
    """Parse this map's WAD lumps for minimap rendering. Returns:
      lines:         [(x1, y1, x2, y2, color_or_None), ...]
      exits:         [(x, y), ...]   midpoints of exit linedefs
      player_start:  (x, y) or None  position of Thing type 1 (player spawn)
      bounds:        (min_x, min_y, max_x, max_y)  for full-map view scaling
      wad_enemy_counts: {class_name: count}  what's placed in this map
    """
    editor = omg.MapEditor(WAD.maps[map_name])
    lines = []
    exits = []
    for ld in editor.linedefs:
        v1 = editor.vertexes[ld.vx_a]
        v2 = editor.vertexes[ld.vx_b]
        action = getattr(ld, "action", 0)
        lines.append((v1.x, v1.y, v2.x, v2.y, SPECIAL_LINE_COLORS.get(action)))
        if action in EXIT_LINE_SPECIALS:
            exits.append(((v1.x + v2.x) / 2, (v1.y + v2.y) / 2))
    player_start = None
    wad_enemy_counts = {}
    for thing in editor.things:
        if thing.type == 1 and player_start is None:
            player_start = (thing.x, thing.y)
        name = MONSTER_TYPES.get(thing.type)
        if name in CLASS_TO_ID:
            wad_enemy_counts[name] = wad_enemy_counts.get(name, 0) + 1
    if lines:
        xs = [c for l in lines for c in (l[0], l[2])]
        ys = [c for l in lines for c in (l[1], l[3])]
        bounds = (min(xs), min(ys), max(xs), max(ys))
    else:
        bounds = (0, 0, 1, 1)
    return lines, exits, player_start, bounds, wad_enemy_counts


def count_captured_classes(map_dir):
    """Count per-class bbox instances across every .txt in a map directory.
    Used to populate the live capture-progress panel at map start."""
    counts = {name: 0 for name in ENEMY_CLASSES}
    for fname in os.listdir(map_dir):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(map_dir, fname)) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                try:
                    cid = int(parts[0])
                    if 0 <= cid < len(ENEMY_CLASSES):
                        counts[ENEMY_CLASSES[cid]] += 1
                except ValueError:
                    continue
    return counts

# ---- Minimap rendering ----
MINIMAP_SIZE = 220           # square, pixels
MINIMAP_MARGIN = 8           # px from frame edge
MINIMAP_RANGE = 1100         # world units shown from player to edge of minimap (local mode)
TRAIL_LENGTH = 800           # number of past player positions kept for the breadcrumb trail

# Doom keys (BGR colors for cv2). Both keycard and skull variants exist per color.
KEY_CLASSES = {
    "BlueCard":   (255,   0,   0),
    "BlueSkull":  (255,   0,   0),
    "RedCard":    (  0,   0, 255),
    "RedSkull":   (  0,   0, 255),
    "YellowCard": (  0, 255, 255),
    "YellowSkull":(  0, 255, 255),
}

def draw_class_panel(frame, captured_counts, wad_counts):
    """Top-left panel listing each enemy class for the current map with status:
       green   "<count>"   = at least one frame captured (discovered)
       cyan    "?"         = exists in this map but not yet captured
       grey    (dim)       = not present in this map
    """
    x, y = 8, 48
    line_h = 13
    cv2.putText(frame, "ENEMIES (this map)", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    y += line_h + 2
    for cls in ENEMY_CLASSES:
        expected = wad_counts.get(cls, 0)
        captured = captured_counts.get(cls, 0)
        if expected == 0:
            color = (90, 90, 90)
            text = f"  {cls}"
        elif captured == 0:
            color = (0, 200, 255)
            text = f"? {cls}"
        else:
            color = (0, 255, 0)
            text = f"  {cls} {captured}"
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        y += line_h


def draw_minimap(frame, lines, px, py, angle_deg,
                 enemy_positions, key_positions, trail,
                 exits, player_start, bounds, full_map=False):
    """Draw the minimap on `frame`. Two modes:
    - Local (default): 220×220 in top-right corner, rotating, player at center.
    - Full (full_map=True): fills the cv2 window, static north-up.
    Always shown: walls (grey), locked doors (colored), exit linedefs (green/magenta),
    trail (faint green line), enemies (yellow dots), keys (colored squares),
    exit markers ("E"), player-start marker ("S"), player (green arrow)."""
    h, w = frame.shape[:2]
    if full_map:
        size = min(w, h) - 2 * MINIMAP_MARGIN
        x0 = (w - size) // 2
        y0 = (h - size) // 2
    else:
        size = MINIMAP_SIZE
        x0 = w - size - MINIMAP_MARGIN
        y0 = MINIMAP_MARGIN
    cx = x0 + size // 2
    cy = y0 + size // 2

    # Background + border
    cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (20, 20, 20), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (200, 200, 200), 1)

    angle_rad = math.radians(angle_deg)
    if full_map:
        # No rotation; fit the whole map.
        mx0, my0, mx1, my1 = bounds
        map_cx = (mx0 + mx1) / 2
        map_cy = (my0 + my1) / 2
        scale = size / max(1.0, max(mx1 - mx0, my1 - my0)) * 0.95
        def w2m(wx, wy):
            return int(cx + (wx - map_cx) * scale), int(cy - (wy - map_cy) * scale)
    else:
        # Rotating local view: player's forward direction aligned with screen-up.
        scale = size / (2 * MINIMAP_RANGE)
        theta = math.pi / 2 - angle_rad
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        def w2m(wx, wy):
            dx, dy = wx - px, wy - py
            rx = dx * cos_t - dy * sin_t
            ry = dx * sin_t + dy * cos_t
            return int(cx + rx * scale), int(cy - ry * scale)

    rect = (x0, y0, size, size)
    # Plain walls first, colored special lines on top.
    for x1, y1, x2, y2, color in lines:
        if color is not None:
            continue
        in_rect, c1, c2 = cv2.clipLine(rect, w2m(x1, y1), w2m(x2, y2))
        if in_rect:
            cv2.line(frame, c1, c2, (160, 160, 160), 1)
    for x1, y1, x2, y2, color in lines:
        if color is None:
            continue
        in_rect, c1, c2 = cv2.clipLine(rect, w2m(x1, y1), w2m(x2, y2))
        if in_rect:
            cv2.line(frame, c1, c2, color, 2)

    # Breadcrumb trail (oldest → newest, segment-clipped).
    prev = None
    for tx, ty in trail:
        cur = w2m(tx, ty)
        if prev is not None:
            in_rect, c1, c2 = cv2.clipLine(rect, prev, cur)
            if in_rect:
                cv2.line(frame, c1, c2, (100, 180, 100), 1)
        prev = cur

    # Enemies (yellow dots).
    for ex, ey in enemy_positions:
        sx, sy = w2m(ex, ey)
        if x0 <= sx <= x0 + size and y0 <= sy <= y0 + size:
            cv2.circle(frame, (sx, sy), 3, (0, 255, 255), -1)

    # Keys (colored squares with white outline).
    for kx, ky, color in key_positions:
        sx, sy = w2m(kx, ky)
        if x0 <= sx <= x0 + size and y0 <= sy <= y0 + size:
            cv2.rectangle(frame, (sx - 4, sy - 4), (sx + 4, sy + 4), color, -1)
            cv2.rectangle(frame, (sx - 4, sy - 4), (sx + 4, sy + 4), (255, 255, 255), 1)

    # Player start "S" (grey).
    if player_start is not None:
        sx, sy = w2m(*player_start)
        if x0 <= sx <= x0 + size and y0 <= sy <= y0 + size:
            cv2.putText(frame, "S", (sx - 4, sy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 2)

    # Exit "E" markers (bright green).
    for ex_x, ex_y in exits:
        sx, sy = w2m(ex_x, ex_y)
        if x0 <= sx <= x0 + size and y0 <= sy <= y0 + size:
            cv2.putText(frame, "E", (sx - 5, sy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Player arrow: full-map = at projected position, rotated by facing direction;
    # local mode = at center, pointing up (player IS the orientation reference).
    if full_map:
        psx, psy = w2m(px, py)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        verts = []
        for vx, vy in [(7, 0), (-4, 4), (-4, -4)]:
            rx = vx * cos_a - vy * sin_a
            ry = vx * sin_a + vy * cos_a
            verts.append((int(psx + rx), int(psy - ry)))
        cv2.fillPoly(frame, [np.array(verts, dtype=np.int32)], (0, 255, 0))
    else:
        tri = np.array([(cx, cy - 6), (cx - 5, cy + 5), (cx + 5, cy + 5)], dtype=np.int32)
        cv2.fillPoly(frame, [tri], (0, 255, 0))

quit_requested = False
total_saved = 0
full_map_mode = False        # toggled with 'm' key; persists across maps

for map_name in MAPS:
    if quit_requested:
        break
    map_dir = os.path.join(OUTPUT_DIR, map_name)
    os.makedirs(map_dir, exist_ok=True)

    # Persistent per-map progress: a `.done` marker means the user has finished
    # capturing this map. We use a marker file (rather than e.g. "skip if any frames
    # exist") so a partial session can be resumed: pressing 'q' mid-map leaves the
    # map un-done, and the next run picks up where it stopped.
    done_marker = os.path.join(map_dir, ".done")
    # ONLY_MAPS overrides .done so the user can explicitly re-capture a finished map.
    if os.path.exists(done_marker) and "ONLY_MAPS" not in os.environ:
        print(f"{map_name}: skipping (marked done — delete {done_marker} to re-capture)")
        continue

    game.set_doom_map(map_name)
    game.new_episode()
    game.send_game_command("god")
    game.send_game_command("give plasmarifle")
    game.send_game_command("give cell 600")
    game.send_game_command("selectweapon PlasmaRifle")

    # Load this map's geometry + metadata once for the minimap.
    map_lines, map_exits, map_player_start, map_bounds, map_wad_counts = load_map_data(map_name)
    # Per-episode breadcrumb trail (reset on each new map).
    player_trail = collections.deque(maxlen=TRAIL_LENGTH)
    # Per-class capture progress for this map (counts bbox instances across all .txt
    # files already on disk; we increment in-memory as we save more during this session).
    captured_counts = count_captured_classes(map_dir)

    # Resume index if files already exist (so you can do multiple sessions per map)
    existing = sorted(f for f in os.listdir(map_dir) if f.endswith(".png"))
    saved = int(existing[-1].split(".")[0]) + 1 if existing else 0

    tick = 0
    next_pressed = False
    unrecognized_seen = set()  # diagnostic: label names not in CLASS_TO_ID
    while not game.is_episode_finished():
        tick += 1
        state = game.get_state()
        if state is None:
            game.advance_action()
            continue

        enemies = [l for l in state.labels
                   if l.object_name in CLASS_TO_ID
                   and l.width * l.height >= MIN_BBOX_AREA]
        # Track unknown labels so we can spot freedoom2 emitting non-standard class names.
        unknowns = [l for l in state.labels if l.object_name not in CLASS_TO_ID]
        for lab in unknowns:
            if lab.object_name not in unrecognized_seen:
                unrecognized_seen.add(lab.object_name)
                print(f"  [{map_name}] new unrecognized label: {lab.object_name!r}")

        # Live preview: green = recognized enemy (saved to dataset).
        # Unrecognized labels are still tracked for the stdout summary, but no longer
        # drawn — the diagnostic phase is over, all 17 classes have been confirmed.
        frame_view = state.screen_buffer.copy()
        for lab in enemies:
            cv2.rectangle(frame_view, (lab.x, lab.y),
                          (lab.x + lab.width, lab.y + lab.height), (0, 255, 0), 1)
            cv2.putText(frame_view, lab.object_name, (lab.x, max(lab.y - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # HUD: map + saved counts + time remaining until episode auto-ends.
        et = int(game.get_episode_time())
        remaining = max(0, EPISODE_TIMEOUT_TICKS - et) // 35  # seconds
        mins, secs = divmod(remaining, 60)
        cv2.putText(frame_view, f"{map_name}  saved={saved}  total={total_saved}  T-{mins}:{secs:02d}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Minimap overlay: top-right corner.
        px = game.get_game_variable(vzd.GameVariable.POSITION_X)
        py = game.get_game_variable(vzd.GameVariable.POSITION_Y)
        pa = game.get_game_variable(vzd.GameVariable.ANGLE)
        enemy_positions = [(o.position_x, o.position_y) for o in (state.objects or [])
                           if o.name in CLASS_TO_ID]
        key_positions = [(o.position_x, o.position_y, KEY_CLASSES[o.name])
                         for o in (state.objects or []) if o.name in KEY_CLASSES]
        player_trail.append((px, py))
        draw_minimap(frame_view, map_lines, px, py, pa,
                     enemy_positions, key_positions, player_trail,
                     map_exits, map_player_start, map_bounds,
                     full_map=full_map_mode)
        draw_class_panel(frame_view, captured_counts, map_wad_counts)

        cv2.imshow("doom capture", frame_view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            quit_requested = True
            break
        if key == ord('n'):
            next_pressed = True
            break
        if key == ord('m'):
            full_map_mode = not full_map_mode

        # Save (frame.png + labels.txt) when enemies are present and frame-skip aligns
        if enemies and tick % FRAME_SKIP == 0:
            stem = f"{saved:06d}"
            cv2.imwrite(os.path.join(map_dir, stem + ".png"), state.screen_buffer)
            with open(os.path.join(map_dir, stem + ".txt"), "w") as f:
                for lab in enemies:
                    cls_id = CLASS_TO_ID[lab.object_name]
                    cx = (lab.x + lab.width / 2) / W
                    cy = (lab.y + lab.height / 2) / H
                    bw = lab.width / W
                    bh = lab.height / H
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                    captured_counts[lab.object_name] += 1
            saved += 1
            total_saved += 1

        game.advance_action()

    # Identify which exit condition fired, for the per-map log line.
    finished = game.is_episode_finished()
    if next_pressed:
        reason = "user pressed 'n'"
    elif quit_requested:
        reason = "user pressed 'q'"
    elif finished and game.get_episode_time() >= EPISODE_TIMEOUT_TICKS - 1:
        reason = "timeout"
    elif finished:
        reason = "level exit"
    else:
        reason = "unknown"

    # Mark map done if the user explicitly said so (n) or the episode ended naturally
    # (level exit / timeout). 'q' deliberately does NOT mark done — it's a "pause".
    if next_pressed or finished:
        open(done_marker, "w").close()
        print(f"{map_name}: saved {saved} cumulative; marked DONE ({reason})", flush=True)
    else:
        print(f"{map_name}: saved {saved} cumulative; left un-done ({reason})", flush=True)
    if unrecognized_seen:
        print(f"  unrecognized labels this session: {sorted(unrecognized_seen)}", flush=True)

cv2.destroyAllWindows()
game.close()
print(f"\nTotal new frames saved this run: {total_saved}")
print(f"Dataset root: {os.path.abspath(OUTPUT_DIR)}")
