from mss import mss
import Quartz
import time
import cv2
import numpy as np
import csv
import threading
from pynput import keyboard
from pathlib import Path
from datetime import datetime
import json

# Config
FPS = 30
SECONDS = 5
OUT_PATH = "capture_test.mp4"
INPUTS_PATH = "inputs_test.csv"
FRAMES_PATH = "frames_test.csv"

WINDOW = "King of Thieves"
JUMP_KEY = keyboard.Key.space

DATA_ROOT = "data"
LEVEL_ID = "test" # Adjust per level
    
def find_window_region(owner_target: str):
    options = Quartz.kCGWindowListExcludeDesktopElements | Quartz.kCGWindowListOptionOnScreenOnly
    windows = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)

    for w in windows:
        owner = w.get("kCGWindowOwnerName", "")
        name = w.get("kCGWindowName", "")
        wid = w.get("kCGWindowNumber")

        if owner != owner_target:
            continue

        bounds = w.get(Quartz.kCGWindowBounds)

        region = {
            "left": int(bounds["X"]),
            "top": int(bounds["Y"]),
            "width": int(bounds["Width"]),
            "height": int(bounds["Height"]),
        }
        print(f"Using window id={wid}, owner={owner}, title='{name}', region={region}")
        return region

    raise RuntimeError(f"window not found for owner='{owner_target}'")

# Creates data_root/raw/<level_id>/<run_id>/
def make_raw_run_dir(data_root: str, level_id: str):
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(data_root) / "raw" / level_id / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    return {
        "level_id": level_id,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "capture_path": str(run_dir / "capture.mp4"),
        "inputs_path": str(run_dir / "inputs.csv"),
        "frames_path": str(run_dir / "frames.csv"),
        "meta_path": str(run_dir / "meta.json"),
    }
    
def write_meta(meta_path: str, payload: dict):
    with open(meta_path, "w") as f:
        json.dump(payload, f, indent=2)

class InputLogger:
    def __init__(self, out_csv_path = str, jump_key = keyboard.Key.space):
        self.out_csv_path = out_csv_path
        self.jump_key = jump_key
        self._t0 = None
        self._listener = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._csv_file = None
        self._writer = None
        self._jump_down = False
        
    def start(self, t0: float):
        self._t0 = t0
        self._csv_file = open(self.out_csv_path, "w")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(["t", "event", "key"])
        self._csv_file.flush()
        
        def on_press(key):
            if self._stop_event.is_set():
                return False
            if key == self.jump_key:
                if not self._jump_down:
                    self._jump_down = True
                    self._write_event("keydown", "space" if key == keyboard.Key.space else str(key))

        def on_release(key):
            if self._stop_event.is_set():
                return False
            if key == self.jump_key:
                self._jump_down = False
                self._write_event("keyup", "space" if key == keyboard.Key.space else str(key))
                
        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        
    def _write_event(self, event_type: str, key_name: str):
        t = time.perf_counter() - self._t0
        with self._lock:
            self._writer.writerow([f"{t:.6f}", event_type, key_name])
            print(f"key input at with details {t:.6f}", event_type, key_name)
            self._csv_file.flush()

    def stop(self):
        self._stop_event.set()
        if self._listener is not None:
            self._listener.stop()
            self._listener.join(timeout=1.0)
        if self._csv_file is not None:
            self._csv_file.close()
    
def record_region(region, out_path: str, inputs_path: str, frames_path: str, meta_path: str, level_id: str, run_id: str):
    width, height = region["width"], region["height"]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (width, height))

    if not writer.isOpened():
        raise RuntimeError("failed to open videowriter")
    
    frame_count = int(FPS * SECONDS)
    frame_time = 1.0 / FPS
    
    # Shared timebase
    t0 = time.perf_counter()
    
    input_logger = InputLogger(inputs_path, jump_key=JUMP_KEY)
    input_logger.start(t0)
    
    frames_file = open(frames_path, "w", newline="")
    frames_writer = csv.writer(frames_file)
    frames_writer.writerow(["frame_idx", "t"])
    frames_file.flush()
    
    meta_payload = {
        "level_id": level_id,
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fps_target": FPS,
        "seconds_target": SECONDS,
        "window_owner_target": WINDOW,
        "region": {
            "left": region["left"],
            "top": region["top"],
            "width": region["width"],
            "height": region["height"],
        },
        "notes": ""
    }
    write_meta(meta_path, meta_payload)

    print("recording started (video + inputs)")
    try:
        with mss() as sct:
            t_next = t0
            for i in range(frame_count):
                shot = sct.grab(region)

                frame = np.array(shot)[:, :, :3] # BGR
                writer.write(frame)

                t_frame = time.perf_counter() - t0
                frames_writer.writerow([i, f"{t_frame:.6f}"])
                if i % 30 == 0:
                    frames_file.flush()

                t_next += frame_time
                sleep_for = t_next - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
    finally:
        writer.release()
        frames_file.close()
        input_logger.stop()
        
    print(f"saved video: {out_path}")
    print(f"saved inputs: {inputs_path}")
    print(f"saved frames: {frames_path}")
    print(f"saved meta:   {meta_path}")


if __name__ == "__main__":
    print(f"starting {SECONDS} second recording in 3 seconds")
    time.sleep(3)

    region = find_window_region(owner_target=WINDOW)

    paths = make_raw_run_dir(DATA_ROOT, LEVEL_ID)
    print(f"\nRun directory: {paths['run_dir']}\n")

    record_region(
        region=region,
        out_path=paths["capture_path"],
        inputs_path=paths["inputs_path"],
        frames_path=paths["frames_path"],
        meta_path=paths["meta_path"],
        level_id=paths["level_id"],
        run_id=paths["run_id"],
    )