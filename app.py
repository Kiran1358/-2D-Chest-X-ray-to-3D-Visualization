import streamlit as st
from PIL import Image
import torch
import numpy as np
from plot_3d import plot_depth_map  # Your 3D plotting function
from yolo_detect import detect_tb_regions  # Your YOLOv5 object detection function

st.title("2D Chest X-ray to 3D TB Visualizer")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Step 1: Run TB detection using YOLOv5
    st.subheader("Detecting TB-affected regions...")
    try:
        detected_img = detect_tb_regions(image)
        st.image(detected_img, caption="TB Regions Detected", use_column_width=True)
    except Exception as e:
        st.error(f"TB Detection Error: {e}")

    # Step 2: Run Depth Estimation (MiDaS)
    st.subheader("Estimating Depth (3D)...")
    try:
        depth_map = plot_depth_map(image)  # returns matplotlib fig or image
        st.pyplot(depth_map)  # or st.image() depending on what plot_depth_map returns
    except Exception as e:
        st.error(f"3D Visualization Error: {e}")
