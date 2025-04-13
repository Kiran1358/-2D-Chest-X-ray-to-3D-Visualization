import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def plot_depth_map(image):
    # Load MiDaS model
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)

    # Transform image
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    input_image = transform(image).to(device)

    # Predict depth
    with torch.no_grad():
        prediction = midas(input_image)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize for visualization
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_vis = (255 * (depth_map - depth_min) / (depth_max - depth_min)).astype(np.uint8)

    # Plot using matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(depth_vis, cmap='plasma')
    ax.set_title('Estimated Depth Map')
    ax.axis('off')

    return fig
