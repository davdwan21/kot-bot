from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
import os
import cv2
import numpy as np
from tracker import SimpleTracker
import csv
from typing import Optional
from math import hypot
import json
from pathlib import Path

# CSV config
INPUTS_CSV = "inputs_test.csv"
STATE_CSV = "states_test.csv"
LABELS_CSV = "labels_test.csv"
TRAIN_CSV = "train_test.csv"
FRAMES_CSV = "frames_test.csv"

TRAP_CLASSES = {"saw", "bullet", "rg"}
K_TRAPS = 2

# CV Model config
load_dotenv()
MODEL_ID = "simple-kot/5"
CONF_THRES = 0.5
STRIDE = 1

IN_VID = "vids/capture2.mp4"
OUT_VID = "vids/burner.mp4"

API_URL = "http://localhost:9001" # Docker required for local host
API_KEY = os.getenv("ROBOFLOW_KEY")

# Font config
font_path = "Helvetica.ttc"
font_size = 14
try:
    font = ImageFont.truetype(font_path, font_size)
except IOError:
    print(f"could not load font: {font_path}. Using default font.")
    font = ImageFont.load_default() 

# Load client
CLIENT = InferenceHTTPClient(
    api_url=API_URL,
    api_key=API_KEY
)
DATA_ROOT = "data"

# Returns all raw run directories: data/raw/<level>/<run>/
def list_raw_runs(data_root: str = DATA_ROOT):
    raw_root = Path(data_root) / "raw"
    if not raw_root.exists():
        raise FileNotFoundError(f"Missing raw root: {raw_root}")

    runs = []
    for level_dir in sorted(raw_root.iterdir()):
        if not level_dir.is_dir():
            continue
        for run_dir in sorted(level_dir.iterdir()):
            if run_dir.is_dir():
                # require expected input files
                if (run_dir / "capture.mp4").exists() and (run_dir / "frames.csv").exists() and (run_dir / "inputs.csv").exists():
                    runs.append(run_dir)
    return runs

def pick_run_interactive(data_root: str = DATA_ROOT) -> Path:
    runs = list_raw_runs(data_root)
    if not runs:
        raise RuntimeError(f"No runs found under {Path(data_root)/'raw'}")

    print("\nAvailable raw runs:\n")
    for i, p in enumerate(runs):
        # p is path = data/raw/<level>/<run>
        level_id = p.parent.name
        run_id = p.name
        print(f"[{i}] {level_id}/{run_id}")

    while True:
        s = input("\nPick a run index: ").strip()
        try:
            idx = int(s)
            if 0 <= idx < len(runs):
                return runs[idx]
        except ValueError:
            pass
        print("Invalid index, try again.")

# Given raw run dir: data/raw/<level>/<run>/
# Create processed dir: data/processed/<level>/<run>/
def resolve_paths_from_run_dir(run_dir: Path, data_root: str = DATA_ROOT) -> dict:
    level_id = run_dir.parent.name
    run_id = run_dir.name

    capture = run_dir / "capture.mp4"
    frames = run_dir / "frames.csv"
    inputs = run_dir / "inputs.csv"
    meta = run_dir / "meta.json"

    for p in [capture, frames, inputs]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    out_dir = Path(data_root) / "processed" / level_id / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    return {
        "level_id": level_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "capture": str(capture),
        "frames": str(frames),
        "inputs": str(inputs),
        "meta": str(meta) if meta.exists() else None,
        "out_dir": str(out_dir),
        "state": str(out_dir / "state.csv"),
        "labels": str(out_dir / "labels.csv"),
        "train": str(out_dir / "train.csv"),
        "annotated": str(out_dir / "annotated.mp4"),
    }


def load_frame_times(frames_csv_path: str) -> dict[int, float]:
    t_by_frame = {}
    with open(frames_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t_by_frame[int(row["frame_idx"])] = float(row["t"])
    return t_by_frame


def load_keydowns(csv_path: str, key_name: str = "space") -> list[float]:
    times = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("key") == key_name and row.get("event") == "keydown":
                times.append(float(row["t"]))
    times.sort()
    return times

def interval_has_keydown(keydowns: list[float], start_t: float, end_t: float, idx_ptr: int) -> tuple[int, int]:
    label = 0
    n = len(keydowns)

    while idx_ptr < n and keydowns[idx_ptr] <= start_t:
        idx_ptr += 1

    label = 0
    if idx_ptr < n and keydowns[idx_ptr] <= end_t:
        label = 1
        # handle all keydowns up to end_t (safeguard)
        while idx_ptr < n and keydowns[idx_ptr] <= end_t:
            idx_ptr += 1

    return label, idx_ptr
        
def pick_best_track(tracks, cls: str):
    cands = [tr for tr in tracks if tr.cls == cls and tr.last_xy is not None]
    
    if not cands:
        return None
    
    # Pick freshest (least missed) track
    cands.sort(key=lambda tr: (tr.missed, -tr.age))
    return cands[0]

def dist_xy(a, b) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])

def pick_k_nearest_tracks(tracks, player_xy: Optional[tuple[float, float]], k: int = 2):
    trap_tracks = [tr for tr in tracks if tr.cls in TRAP_CLASSES and tr.last_xy is not None]
    if not trap_tracks:
        return []
    
    # Return most recent traps if no player detection
    if player_xy is None:
        trap_tracks.sort(key=lambda tr: (tr.missed, -tr.age))
        return trap_tracks[:k]
    
    # Choose K closest traps to player
    trap_tracks.sort(key=lambda tr: (dist_xy(tr.last_xy, player_xy), tr.missed, -tr.age))
    
    return trap_tracks[:k]

def build_state_row(frame_idx: int, t: float, prev_t: Optional[float], tracks) -> dict:
    dt = 0.0 if prev_t is None else float(t - prev_t)
    
    player = pick_best_track(tracks, "player")
    goal = pick_best_track(tracks, "goal")
    
    player_present = 1 if player is not None else 0
    goal_present = 1 if goal is not None else 0
    
    if player_present:
        px, py = player.last_xy
        pvx, pvy = player.last_v if player.last_v is not None else (0.0, 0.0)
        player_xy = (px, py)
    else:
        px = 0.0
        py = 0.0
        pvx = 0.0
        pvy = 0.0
        player_xy = None
        
    if goal_present:
        gx, gy = goal.last_xy
    else:
        gx = 0.0
        gy = 0.0
        
    goal_dx = gx - px if (player_present and goal_present) else 0.0
    goal_dy = gy - py if (player_present and goal_present) else 0.0
    
    traps = pick_k_nearest_tracks(tracks, player_xy, K_TRAPS)
    
    row = {
        "frame_idx": frame_idx,
        "t": float(t),
        "dt": float(dt),
        
        "player_x": float(px),
        "player_y": float(py),
        "player_vx": float(pvx),
        "player_vy": float(pvy),
        "player_present": int(player_present),
        
        "goal_x": float(gx),
        "goal_y": float(gy),
        "goal_dx": float(goal_dx),
        "goal_dy": float(goal_dy),
        "goal_present": int(goal_present),
    }
    
    for i in range(K_TRAPS):
        key = i + 1
        if i < len(traps):
            tr = traps[i]
            tx, ty = tr.last_xy
            tvx, tvy = tr.last_v if tr.last_v is not None else (0.0, 0.0)
            
            row[f"trap{key}_x"] = float(tx)
            row[f"trap{key}_y"] = float(ty)
            row[f"trap{key}_vx"] = float(tvx)
            row[f"trap{key}_vy"] = float(tvy)
            
            row[f"trap{key}_dx"] = float(tx - px) if player_present else 0.0
            row[f"trap{key}_dy"] = float(ty - py) if player_present else 0.0
            row[f"trap{key}_present"] = 1
        else:
            row[f"trap{key}_x"] = 0.0
            row[f"trap{key}_y"] = 0.0
            row[f"trap{key}_vx"] = 0.0
            row[f"trap{key}_vy"] = 0.0
            
            row[f"trap{key}_dx"] = 0.0
            row[f"trap{key}_dy"] = 0.0
            row[f"trap{key}_present"] = 0
    
    return row

""" Draw Roboflow predictions on image and return it """
def draw_boxes(img: Image.Image, result: dict | list[dict]) -> Image.Image:
    # img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    preds = result.get("predictions", [])
    for p in preds:
        if p["confidence"] < CONF_THRES:
            continue

        x = p["x"]
        y = p["y"]
        w = p["width"]
        h = p["height"]

        left = x - w / 2
        top = y - h / 2
        right = x + w / 2
        bottom = y + h / 2

        label = f'{p.get("class","?")} {p.get("confidence",0):.2f}'
        draw.rectangle([left, top, right, bottom], width=3, outline="red")
        draw.text((left, top - 16), label, fill="black", font=font)

    return img

def draw_tracks(img: Image.Image, tracks, v_scale=0.3, a_scale=0.02) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for tr in tracks:
        if tr.last_box_xyxy is None:
            continue
        
        x1, y1, x2, y2 = tr.last_box_xyxy
        draw.rectangle([x1, y1, x2, y2], width=3, outline="green")
        draw.text((x1, y1 - 16), f"id={tr.track_id} {tr.cls}", fill="green", font=font)
        
        cx, cy = tr.last_xy
        
        if tr.last_v is not None:
            # print(f"id: {tr.track_id} has velo {tr.last_v}")
            draw_arrow(draw, (cx, cy), tr.last_v, scale=v_scale, color="blue", width=3)
            
        if tr.last_a is not None:
            # print(f"id: {tr.track_id} has accel {tr.last_v}")
            
            # Don't draw accel arrow (too noisy)
            # draw_arrow(draw, (cx, cy), tr.last_a, scale=a_scale, color="red", width=3)
            
        # print(f"id {tr.track_id} is class {tr.cls}")

    return img

def draw_arrow(draw, start, vec, scale=1.0, color="blue", width=3):
    x0, y0 = start
    vx, vy = vec
    x1 = x0 + vx * scale
    y1 = y0 + vy * scale
    draw.line([x0, y0, x1, y1], fill=color, width=width)

""" Make inference on single frame (image) """
def infer_frame(frame_path: str) -> dict | list[dict]:
    return CLIENT.infer(frame_path, model_id=MODEL_ID)


""" Run predictions on the video and save an annotated copy """
def process_video(in_path: str, out_path: str, frames_csv: str, inputs_csv: str,
                  state_csv: str, labels_csv: str, train_csv: str):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {in_path}")
    
    t_by_frame = load_frame_times(frames_csv)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    keydowns = load_keydowns(inputs_csv, "space")
    kd_ptr = 0
    
    tracker = SimpleTracker(
        max_age=10,
        conf_thres=CONF_THRES,
        use_iou_gate=True,
        min_iou=0.1
    )
    
    state_fieldnames = [
        "frame_idx","t","dt",
        "player_x","player_y","player_vx","player_vy","player_present",
        "goal_x","goal_y","goal_dx","goal_dy","goal_present",
        "trap1_x","trap1_y","trap1_vx","trap1_vy","trap1_dx","trap1_dy","trap1_present",
        "trap2_x","trap2_y","trap2_vx","trap2_vy","trap2_dx","trap2_dy","trap2_present",
    ]
    labels_fieldnames = ["frame_idx", "t", "jump_next"]
    train_fieldnames = state_fieldnames + ["jump_next"]
    
    state_f = open(state_csv, "w", newline="")
    labels_f = open(labels_csv, "w", newline="")
    train_f = open(train_csv, "w", newline="")

    state_writer = csv.DictWriter(state_f, fieldnames=state_fieldnames)
    labels_writer = csv.DictWriter(labels_f, fieldnames=labels_fieldnames)
    train_writer = csv.DictWriter(train_f, fieldnames=train_fieldnames)

    state_writer.writeheader()
    labels_writer.writeheader()
    train_writer.writeheader()
    
    pending_state = None
    pending_t = None
    pending_frame_idx = None
    
    prev_infer_t = None
    
    frame_idx = 0
    last_result = None
    last_tracks = []
    
    print("Beginning inference")
    try:
        while True:
            print(f"Reading frame {frame_idx}")
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % STRIDE == 0:
                temp_frame_path = "temp_frame.png"
                cv2.imwrite(temp_frame_path, frame_bgr)
                result = infer_frame(temp_frame_path)
                last_result = result
                
                # Updated t get due to potential frame <--> time mismatching
                t = t_by_frame.get(frame_idx)
                if t is None:
                    # Fallback, ideally never used
                    t = frame_idx / (fps or 30.0)

                last_tracks = tracker.update(result, t)

                # Last state row is dropped (no next frame to predict)
                state_row = build_state_row(frame_idx=frame_idx, t=t, prev_t=prev_infer_t, tracks=last_tracks)
                
                if pending_state is not None and pending_t is not None:
                    jump_label, kd_ptr = interval_has_keydown(keydowns, pending_t, t, kd_ptr)
                    
                    state_writer.writerow(pending_state)
                    labels_writer.writerow({"frame_idx": pending_frame_idx, "t": pending_t, "jump_next": jump_label})
                    
                    train_row = dict(pending_state)
                    train_row["jump_next"] = jump_label
                    train_writer.writerow(train_row)
                    
                pending_state = state_row
                pending_t = t
                pending_frame_idx = frame_idx
                prev_infer_t = t
            else:
                result = last_result

            if result is not None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)

                # Draw boxes for per-frame labeling, tracks is more recent
                # img = draw_boxes(img, result)
                img = draw_tracks(img, last_tracks)

                out_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                writer.write(out_rgb)
            else:
                writer.write(frame_bgr)

            frame_idx += 1
    finally:
        cap.release()
        writer.release()
        state_f.close()
        labels_f.close()
        train_f.close()
        
    print(f"Saved: {out_path}")
    print(f"Wrote to {state_csv}, {labels_csv}, and {train_csv}")
    

if __name__ == "__main__":
    run_dir = pick_run_interactive(DATA_ROOT)
    paths = resolve_paths_from_run_dir(run_dir, DATA_ROOT)

    print("\nSelected run:")
    print(f"  raw:  {paths['run_dir']}")
    print(f"  out:  {paths['out_dir']}\n")

    process_video(
        in_path=paths["capture"],
        out_path=paths["annotated"],
        frames_csv=paths["frames"],
        inputs_csv=paths["inputs"],
        state_csv=paths["state"],
        labels_csv=paths["labels"],
        train_csv=paths["train"],
    )
