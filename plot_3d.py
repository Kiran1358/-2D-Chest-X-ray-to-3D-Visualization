import torch
import cv2
import numpy as np
import plotly.graph_objs as go
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from scipy.ndimage import zoom

# Load MiDaS
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
midas.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# MiDaS transform
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

def estimate_depth(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = Image.fromarray(image)
    
    input_tensor = transform(input_image).to(device)
    with torch.no_grad():
        prediction = midas(input_tensor.unsqueeze(0))
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()
    
    return prediction

def plot_depth_map(depth_map):
    # Normalize and resize depth
    depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-8)
    depth_map = np.clip(depth_map, 0, 1)
    
    max_size = 200
    h, w = depth_map.shape
    factor = min(max_size / h, max_size / w, 1.0)
    depth_map = zoom(depth_map, zoom=factor, order=3)
    
    h, w = depth_map.shape
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)

    fig = go.Figure(data=[go.Surface(
        z=depth_map,
        x=xx,
        y=yy,
        colorscale='Inferno',
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.4, roughness=0.3),
        lightposition=dict(x=100, y=200, z=50),
        showscale=False,
        opacity=1.0
    )])

    fig.update_layout(
        title="Realistic 3D Visualization of TB-Affected Region",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(title='Depth', showgrid=False),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        scene_camera=dict(eye=dict(x=1.4, y=1.4, z=0.8))
    )

    return fig
