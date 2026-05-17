"""compare.py — Play Doom with engine ground-truth AND model predictions overlaid.

Same spectator-mode loop as test.py, but does NOT save any frames. Instead:
  - Engine ground-truth boxes (from state.labels) drawn in GREEN, as in test.py.
  - Stage-6 model predictions (stage6_best.pt) drawn in CYAN, with confidence.

The two overlays let you visually see what the model is hallucinating,
missing, or confidently nailing — the kind of inspection that an mAP
number can never give you.

Controls:
  q   quit
  n   next map
  m   toggle local rotating minimap <-> full-map north-up
  p   pause / resume model inference (keep last predictions on screen)
  [   lower CONF threshold (less strict — see more boxes)
  ]   raise CONF threshold (more strict — see fewer boxes)

ONLY_MAPS=MAP17,MAP31 python compare.py    # restrict to specific maps
WEIGHTS=stage8_best.pt python compare.py   # compare a different checkpoint
"""
import os
import math
import time
import collections
from pathlib import Path

import cv2
import numpy as np
import omg
import torch
import vizdoom as vzd

from stage3 import (
    ANCHORS_PX, NUM_CLASSES, GRID_SIZE, INPUT_SIZE,
    CONF_THRESH, NMS_IOU,
    decode_predictions, predictions_to_detections, nms_per_class,
)
from stage4 import IMAGENET_MEAN, IMAGENET_STD
from stage5 import FineTunedYOLO


# Same ENEMY_CLASSES as test.py — must match what the model was trained on.
ENEMY_CLASSES = [
    "Zombieman", "ShotgunGuy", "ChaingunGuy", "DoomImp", "Demon", "Spectre",
    "LostSoul", "Cacodemon", "Fatso", "HellKnight", "Arachnotron",
    "PainElemental", "Revenant", "BaronOfHell", "Archvile",
    "SpiderMastermind", "Cyberdemon",
]
CLASS_TO_ID = {name: i for i, name in enumerate(ENEMY_CLASSES)}

MAPS = [f"MAP{i:02d}" for i in range(1, 33)]
if "ONLY_MAPS" in os.environ:
    MAPS = [m.strip() for m in os.environ["ONLY_MAPS"].split(",") if m.strip()]

WEIGHTS = Path(os.environ.get("WEIGHTS", "stage6_best.pt"))
INFER_EVERY_N_TICKS = int(os.environ.get("INFER_EVERY", "2"))  # CPU-friendly
MIN_BBOX_AREA = 100
EPISODE_TIMEOUT_TICKS = 35 * 60 * 30


# ---------- Model setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Weights: {WEIGHTS}")
if not WEIGHTS.exists():
    raise SystemExit(f"ERROR: weights file {WEIGHTS} not found in cwd")

model = FineTunedYOLO().to(device)
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval()
anchors = ANCHORS_PX.to(device)

MEAN_T = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
STD_T  = torch.tensor(IMAGENET_STD,  device=device).view(1, 3, 1, 1)


@torch.no_grad()
def model_predict(frame_bgr, conf_thresh):
    """Run model on a single Doom BGR frame; return [(cls, conf, x1, y1, x2, y2), ...]
    in the *original* frame's pixel coordinates."""
    frame_h, frame_w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    img_n = (img_t - MEAN_T) / STD_T
    preds = model(img_n)
    cx, cy, w, h, obj, cls = decode_predictions(
        preds, anchors, GRID_SIZE, INPUT_SIZE, NUM_CLASSES, device)
    dets = predictions_to_detections(cx, cy, w, h, obj, cls,
                                     conf_thresh, frame_w, frame_h)[0]
    return nms_per_class(dets, NMS_IOU)


# ---------- Game setup (same as test.py, minus capture-specific bits) ----------
game = vzd.DoomGame()
game.set_doom_game_path(os.path.join(vzd.scenarios_path, "freedoom2.wad"))
game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
game.set_screen_format(vzd.ScreenFormat.BGR24)
game.set_mode(vzd.Mode.ASYNC_SPECTATOR)
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
game.add_game_args("+set use_mouse 0")
game.set_sound_enabled(True)
game.add_game_args("+set snd_musicvolume 0.2")
game.add_game_args("+set snd_sfxvolume 0.08")
game.add_available_game_variable(vzd.GameVariable.POSITION_X)
game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
game.add_available_game_variable(vzd.GameVariable.ANGLE)
game.set_objects_info_enabled(True)
game.init()

W = game.get_screen_width()
H = game.get_screen_height()

WAD = omg.WAD()
WAD.from_file(os.path.join(os.path.dirname(vzd.__file__), "freedoom2.wad"))

SPECIAL_LINE_COLORS = {
    26: (255, 0, 0), 32: (255, 0, 0), 99: (255, 0, 0), 133: (255, 0, 0),
    28: (0, 0, 255), 33: (0, 0, 255), 134: (0, 0, 255), 135: (0, 0, 255),
    27: (0, 255, 255), 34: (0, 255, 255), 136: (0, 255, 255), 137: (0, 255, 255),
    11: (0, 255, 0), 52: (0, 255, 0),
    51: (255, 0, 255), 124: (255, 0, 255),
}
EXIT_LINE_SPECIALS = {11, 51, 52, 124}

MONSTER_TYPES = {
    7: "SpiderMastermind", 9: "ShotgunGuy", 16: "Cyberdemon", 58: "Spectre",
    64: "Archvile", 65: "ChaingunGuy", 66: "Revenant", 67: "Fatso",
    68: "Arachnotron", 69: "HellKnight", 71: "PainElemental",
    3001: "DoomImp", 3002: "Demon", 3003: "BaronOfHell",
    3004: "Zombieman", 3005: "Cacodemon", 3006: "LostSoul",
}

KEY_CLASSES = {
    "BlueCard":    (255,   0,   0), "BlueSkull":   (255,   0,   0),
    "RedCard":     (  0,   0, 255), "RedSkull":    (  0,   0, 255),
    "YellowCard":  (  0, 255, 255), "YellowSkull": (  0, 255, 255),
}

MINIMAP_SIZE   = 220
MINIMAP_MARGIN = 8
MINIMAP_RANGE  = 1100
TRAIL_LENGTH   = 800


def load_map_data(map_name):
    editor = omg.MapEditor(WAD.maps[map_name])
    lines, exits = [], []
    for ld in editor.linedefs:
        v1 = editor.vertexes[ld.vx_a]; v2 = editor.vertexes[ld.vx_b]
        action = getattr(ld, "action", 0)
        lines.append((v1.x, v1.y, v2.x, v2.y, SPECIAL_LINE_COLORS.get(action)))
        if action in EXIT_LINE_SPECIALS:
            exits.append(((v1.x + v2.x) / 2, (v1.y + v2.y) / 2))
    player_start, wad_counts = None, {}
    for thing in editor.things:
        if thing.type == 1 and player_start is None:
            player_start = (thing.x, thing.y)
        name = MONSTER_TYPES.get(thing.type)
        if name in CLASS_TO_ID:
            wad_counts[name] = wad_counts.get(name, 0) + 1
    if lines:
        xs = [c for l in lines for c in (l[0], l[2])]
        ys = [c for l in lines for c in (l[1], l[3])]
        bounds = (min(xs), min(ys), max(xs), max(ys))
    else:
        bounds = (0, 0, 1, 1)
    return lines, exits, player_start, bounds, wad_counts


def draw_class_panel(frame, wad_counts):
    """Map-enemy roster. We don't track captured counts here (no saving)."""
    x, y = 8, 48
    line_h = 13
    cv2.putText(frame, "ENEMIES (this map)", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    y += line_h + 2
    for cls in ENEMY_CLASSES:
        expected = wad_counts.get(cls, 0)
        if expected == 0:
            color, text = (90, 90, 90), f"  {cls}"
        else:
            color, text = (0, 200, 255), f"  {cls} x{expected}"
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        y += line_h


def draw_minimap(frame, lines, px, py, angle_deg,
                 enemy_positions, key_positions, trail,
                 exits, player_start, bounds, full_map=False):
    h, w = frame.shape[:2]
    if full_map:
        size = min(w, h) - 2 * MINIMAP_MARGIN
        x0 = (w - size) // 2; y0 = (h - size) // 2
    else:
        size = MINIMAP_SIZE
        x0 = w - size - MINIMAP_MARGIN; y0 = MINIMAP_MARGIN
    cx = x0 + size // 2; cy = y0 + size // 2

    cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (20, 20, 20), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (200, 200, 200), 1)

    angle_rad = math.radians(angle_deg)
    if full_map:
        mx0, my0, mx1, my1 = bounds
        map_cx = (mx0 + mx1) / 2; map_cy = (my0 + my1) / 2
        scale = size / max(1.0, max(mx1 - mx0, my1 - my0)) * 0.95
        def w2m(wx, wy):
            return int(cx + (wx - map_cx) * scale), int(cy - (wy - map_cy) * scale)
    else:
        scale = size / (2 * MINIMAP_RANGE)
        theta = math.pi / 2 - angle_rad
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        def w2m(wx, wy):
            dx, dy = wx - px, wy - py
            rx = dx * cos_t - dy * sin_t
            ry = dx * sin_t + dy * cos_t
            return int(cx + rx * scale), int(cy - ry * scale)

    rect = (x0, y0, size, size)
    for x1, y1, x2, y2, color in lines:
        if color is not None: continue
        ok, c1, c2 = cv2.clipLine(rect, w2m(x1, y1), w2m(x2, y2))
        if ok: cv2.line(frame, c1, c2, (160, 160, 160), 1)
    for x1, y1, x2, y2, color in lines:
        if color is None: continue
        ok, c1, c2 = cv2.clipLine(rect, w2m(x1, y1), w2m(x2, y2))
        if ok: cv2.line(frame, c1, c2, color, 2)

    prev = None
    for tx, ty in trail:
        cur = w2m(tx, ty)
        if prev is not None:
            ok, c1, c2 = cv2.clipLine(rect, prev, cur)
            if ok: cv2.line(frame, c1, c2, (100, 180, 100), 1)
        prev = cur

    for ex, ey in enemy_positions:
        sx, sy = w2m(ex, ey)
        if x0 <= sx <= x0 + size and y0 <= sy <= y0 + size:
            cv2.circle(frame, (sx, sy), 3, (0, 255, 255), -1)
    for kx, ky, color in key_positions:
        sx, sy = w2m(kx, ky)
        if x0 <= sx <= x0 + size and y0 <= sy <= y0 + size:
            cv2.rectangle(frame, (sx - 4, sy - 4), (sx + 4, sy + 4), color, -1)
            cv2.rectangle(frame, (sx - 4, sy - 4), (sx + 4, sy + 4), (255, 255, 255), 1)
    if player_start is not None:
        sx, sy = w2m(*player_start)
        if x0 <= sx <= x0 + size and y0 <= sy <= y0 + size:
            cv2.putText(frame, "S", (sx - 4, sy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 2)
    for ex_x, ex_y in exits:
        sx, sy = w2m(ex_x, ex_y)
        if x0 <= sx <= x0 + size and y0 <= sy <= y0 + size:
            cv2.putText(frame, "E", (sx - 5, sy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if full_map:
        psx, psy = w2m(px, py)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        verts = []
        for vx, vy in [(7, 0), (-4, 4), (-4, -4)]:
            rx = vx * cos_a - vy * sin_a; ry = vx * sin_a + vy * cos_a
            verts.append((int(psx + rx), int(psy - ry)))
        cv2.fillPoly(frame, [np.array(verts, dtype=np.int32)], (0, 255, 0))
    else:
        tri = np.array([(cx, cy - 6), (cx - 5, cy + 5), (cx + 5, cy + 5)], dtype=np.int32)
        cv2.fillPoly(frame, [tri], (0, 255, 0))


# ---------- Overlay helpers ----------

GT_COLOR    = (0, 255, 0)      # green — engine ground truth
PRED_COLOR  = (255, 255, 0)    # cyan  — model prediction


def draw_engine_truth(frame, labels):
    for lab in labels:
        cv2.rectangle(frame, (lab.x, lab.y),
                      (lab.x + lab.width, lab.y + lab.height), GT_COLOR, 1)
        cv2.putText(frame, lab.object_name, (lab.x, max(lab.y - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 1)


def draw_model_preds(frame, detections):
    h = frame.shape[0]
    for cls_id, conf, x1, y1, x2, y2 in detections:
        x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), PRED_COLOR, 2)
        name = ENEMY_CLASSES[cls_id] if 0 <= cls_id < len(ENEMY_CLASSES) else f"cls{cls_id}"
        label = f"{name} {conf*100:.0f}%"
        # Place model label BELOW the box so it doesn't collide with engine label above.
        ly = min(y2i + 14, h - 4)
        cv2.putText(frame, label, (x1i, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, PRED_COLOR, 1)


def draw_legend(frame):
    """Top-right (below minimap area is occupied; use bottom-left)."""
    h, w = frame.shape[:2]
    y = h - 38
    cv2.rectangle(frame, (8, y - 4), (180, h - 6), (0, 0, 0), -1)
    cv2.rectangle(frame, (12, y),  (28, y + 12), GT_COLOR, -1)
    cv2.putText(frame, "engine truth", (34, y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, GT_COLOR, 1)
    cv2.rectangle(frame, (12, y + 16), (28, y + 28), PRED_COLOR, -1)
    cv2.putText(frame, "model pred", (34, y + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, PRED_COLOR, 1)


# ---------- Main loop ----------

quit_requested = False
full_map_mode  = False
inference_on   = True
conf_thresh    = CONF_THRESH

for map_name in MAPS:
    if quit_requested:
        break
    print(f"\n=== {map_name} ===", flush=True)

    game.set_doom_map(map_name)
    game.new_episode()
    game.send_game_command("god")
    game.send_game_command("give plasmarifle")
    game.send_game_command("give cell 600")
    game.send_game_command("selectweapon PlasmaRifle")

    map_lines, map_exits, map_player_start, map_bounds, map_wad_counts = load_map_data(map_name)
    player_trail = collections.deque(maxlen=TRAIL_LENGTH)

    tick = 0
    last_dets = []         # held over between inference ticks
    last_inf_ms = 0.0
    next_pressed = False

    while not game.is_episode_finished():
        tick += 1
        state = game.get_state()
        if state is None:
            game.advance_action()
            continue

        frame = state.screen_buffer
        engine_enemies = [l for l in state.labels
                          if l.object_name in CLASS_TO_ID
                          and l.width * l.height >= MIN_BBOX_AREA]

        # Throttled inference: every Nth tick, reuse last detections in between.
        if inference_on and (tick % INFER_EVERY_N_TICKS == 0):
            t0 = time.perf_counter()
            last_dets = model_predict(frame, conf_thresh)
            last_inf_ms = (time.perf_counter() - t0) * 1000

        view = frame.copy()
        draw_engine_truth(view, engine_enemies)
        draw_model_preds(view, last_dets)

        # HUD: counts and inference time, top-left strip
        cv2.putText(view,
                    f"{map_name}  engine={len(engine_enemies)}  model={len(last_dets)}  "
                    f"conf>={conf_thresh:.2f}  inf={last_inf_ms:.0f}ms"
                    f"{'' if inference_on else '  [PAUSED]'}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Minimap
        px = game.get_game_variable(vzd.GameVariable.POSITION_X)
        py = game.get_game_variable(vzd.GameVariable.POSITION_Y)
        pa = game.get_game_variable(vzd.GameVariable.ANGLE)
        enemy_positions = [(o.position_x, o.position_y) for o in (state.objects or [])
                           if o.name in CLASS_TO_ID]
        key_positions = [(o.position_x, o.position_y, KEY_CLASSES[o.name])
                         for o in (state.objects or []) if o.name in KEY_CLASSES]
        player_trail.append((px, py))
        draw_minimap(view, map_lines, px, py, pa,
                     enemy_positions, key_positions, player_trail,
                     map_exits, map_player_start, map_bounds,
                     full_map=full_map_mode)
        draw_class_panel(view, map_wad_counts)
        draw_legend(view)

        cv2.imshow("doom compare (engine vs model)", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            quit_requested = True
            break
        if key == ord('n'):
            next_pressed = True
            break
        if key == ord('m'):
            full_map_mode = not full_map_mode
        if key == ord('p'):
            inference_on = not inference_on
        if key == ord('['):
            conf_thresh = max(0.05, conf_thresh - 0.05)
        if key == ord(']'):
            conf_thresh = min(0.95, conf_thresh + 0.05)

        game.advance_action()

    print(f"  finished ({'user-next' if next_pressed else 'episode-end'})", flush=True)

cv2.destroyAllWindows()
game.close()
