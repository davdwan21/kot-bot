from mss import mss
import Quartz
import time
import cv2
import numpy as np
import csv
import threading
from pynput import keyboard

# Config
FPS = 30
SECONDS = 10
OUT_PATH = "capture_test.mp4"
INPUTS_PATH = "inputs_test.csv"
FRAMES_PATH = "frames_test.csv"

WINDOW = "King of Thieves"
JUMP_KEY = keyboard.Key.space

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
    
def record_region(region, out_path: str, inputs_path: str, frames_path: str):
    width, height = region["width"], region["height"]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (width, height))

    if not writer.isOpened():
        raise RuntimeError("failed to open videowriter")
    
    frame_count = int(FPS * SECONDS)
    frame_time = 1.0 / FPS
    
    t0 = time.perf_counter()
    input_logger = InputLogger(inputs_path, jump_key=JUMP_KEY)
    input_logger.start(t0)
    
    frames_file = open(frames_path, "w", newline="")
    frames_writer = csv.writer(frames_file)
    frames_writer.writerow(["frame_idx", "t"])
    frames_file.flush()

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

if __name__ == "__main__":
    print("starting recording in 3 seconds")
    time.sleep(3)
    
    region = find_window_region(owner_target=WINDOW)
    record_region(region, out_path=OUT_PATH, inputs_path=INPUTS_PATH, frames_path=FRAMES_PATH)