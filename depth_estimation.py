import torch
from torchvision import transforms
from PIL import Image

# Load the MiDaS model (only once globally)
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
midas.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize(384),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def estimate_depth(image_path):
    """
    Estimate depth from a 2D image using the MiDaS DPT_Hybrid model.

    Args:
        image_path (str or PIL.Image.Image): Path to image or PIL image object.

    Returns:
        np.ndarray: 2D array representing the estimated depth.
    """
    # Support both file path or already loaded PIL Image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path.convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = midas(input_tensor)

    # Resize to original image size
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=image.size[::-1],  # (H, W)
        mode="bicubic",
        align_corners=False
    ).squeeze()

    return prediction.cpu().numpy()
