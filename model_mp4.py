from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
import os
import cv2
import numpy as np

# model config
load_dotenv()
MODEL_ID = "simple-kot/5"
CONF_THRES = 0.5
STRIDE = 1

IN_VID = "vids/capture2.mp4"
OUT_VID = "vids/annotated2.mp4"

API_URL = "http://localhost:9001" # Docker required for local host
API_KEY = os.getenv("ROBOFLOW_KEY")

# font config
font_path = "Helvetica.ttc"
font_size = 14
try:
    font = ImageFont.truetype(font_path, font_size)
except IOError:
    print(f"could not load font: {font_path}. Using default font.")
    font = ImageFont.load_default() 

# load client
CLIENT = InferenceHTTPClient(
    api_url=API_URL,
    api_key=API_KEY
)

""" draw roboflow predictions on image and return it """
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
        draw.text((left, top - 17), label, fill="black", font=font)

    return img

""" make inference on single frame (image) """
def infer_frame(frame_path: str) -> dict | list[dict]:
    return CLIENT.infer(frame_path, model_id=MODEL_ID)

def calculate_motion(result: dict | list[dict]):
    pass

""" run predictions on the video and save an annotated copy """
def process_video(in_path: str, out_path: str):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {in_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx = 0
    last_result = None
    
    print("beginning inference")
    while True:
        print(f"reading frame {frame_idx}")
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx % STRIDE == 0:
            temp_frame_path = "temp_frame.png"
            cv2.imwrite(temp_frame_path, frame_bgr)
            result = infer_frame(temp_frame_path)
            last_result = result
        else:
            result = last_result

        if result is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            img = draw_boxes(img, result)

            out_rgb = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            writer.write(out_rgb)
        else:
            writer.write(frame_bgr)

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved: {out_path}")
        
if __name__ == "__main__":
    process_video(IN_VID, OUT_VID)