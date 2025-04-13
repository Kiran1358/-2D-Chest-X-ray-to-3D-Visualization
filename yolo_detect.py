import torch
import numpy as np
from PIL import Image
import cv2

# Load YOLOv5 model (path to your trained weights or use a pretrained one for testing)
model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")  # replace 'best.pt' with your path
model.conf = 0.4  # confidence threshold

def detect_tb_regions(image: Image.Image) -> Image.Image:
    # Convert PIL to numpy
    img_np = np.array(image)

    # Run YOLOv5
    results = model(img_np)

    # Draw results on image
    results.render()  # updates results.imgs with boxes drawn
    output_img = Image.fromarray(results.ims[0])

    return output_img
