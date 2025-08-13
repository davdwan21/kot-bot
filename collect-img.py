import pygetwindow as gw
import pyautogui as pag
from PIL import ImageGrab
import time

print(gw.getAllTitles())

target_window = gw.getWindowsWithTitle("BlueStacks App Player")[0]
bbox = (target_window.left, target_window.top, target_window.right, target_window.bottom)
pag.click(target_window.left + 10, target_window.top + 10)
# screenshot = ImageGrab.grab(bbox)
# screenshot.save("imgs/test_img.png")

for i in range(5, 0, -1):
    print(f"starting to take screenshots in {i}...")
    time.sleep(1)
    
counter = 81
while True:
    screenshot = ImageGrab.grab(bbox)
    file_name = f"imgs/hc-bg-bh/08-12-image-{counter}.png"
    screenshot.save(file_name)
    print(f"screenshot saved: {file_name}")
    
    time.sleep(1)
    counter += 1
    
    if counter == 101:
        break