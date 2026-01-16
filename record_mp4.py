from mss import mss
import Quartz
import time
import cv2
import numpy as np

# Config
FPS = 30
SECONDS = 10
OUT_PATH = "capture2.mp4"
WINDOW = "King of Thieves"

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

def record_region(region, out_path: str):
    width, height = region["width"], region["height"]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (width, height))

    if not writer.isOpened():
        raise RuntimeError("failed to open videowriter")
    
    frame_count = int(FPS * SECONDS)
    frame_time = 1.0 / FPS

    print("recording started")
    with mss() as sct:
        t_next = time.perf_counter()
        for i in range(frame_count):
            shot = sct.grab(region)

            frame = np.array(shot)
            frame = frame[:, :, :3]
            writer.write(frame)

            t_next += frame_time
            sleep_for = t_next - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

    writer.release()
    print(f"saved video: {out_path}")

if __name__ == "__main__":
    print("starting recording in 3 seconds")
    time.sleep(3)
    
    region = find_window_region(owner_target=WINDOW)
    record_region(region, out_path=OUT_PATH)