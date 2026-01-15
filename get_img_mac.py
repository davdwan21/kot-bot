from mss import mss
from PIL import Image
import Quartz
import time
import numpy as np

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

region = find_window_region("King of Thieves")
print(region)

with mss() as sct:
    print("taking screenshots in 5 seconds")
    time.sleep(5)
    for i in range(90,92):
        screenshot = sct.grab(region)

        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        img.save(f"imgs/test - {i}.png")
        print(f"image test - {i}.png saved successfully")
        time.sleep(1)