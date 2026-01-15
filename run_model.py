from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
import os

load_dotenv()

def draw_boxes(image_path, result, out_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    preds = result.get("predictions", [])
    for p in preds:
        if p["confidence"] < 0.8:
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

    img.save(out_path)
    return out_path

img_path = "imgs/test - 67.png"
font_path = "Helvetica.ttc" 
font_size = 14
try:
    font = ImageFont.truetype(font_path, font_size)
except IOError:
    print(f"could not load font: {font_path}. Using default font.")
    font = ImageFont.load_default() 


CLIENT = InferenceHTTPClient(
    # api_url="https://serverless.roboflow.com",
    api_url="http://localhost:9001",
    api_key=os.getenv("ROBOFLOW_KEY")
)

result = CLIENT.infer(img_path, model_id="simple-kot/2")

print(result)

annotated_path = draw_boxes(img_path, result, "out2.png")