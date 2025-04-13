import streamlit as st
from PIL import Image
from depth_estimation import estimate_depth
from plot_3d import plot_depth_map

st.set_page_config(page_title="2D to 3D Chest X-ray Visualizer", layout="wide")
st.title("🩻 2D Chest X-ray to 3D Visualization")

# Load sample image
image_path = r"C:\Users\shett\Desktop\TB-Xray-3D-Visualizer\sample.png"
image = Image.open(image_path)
st.image(image, caption="Original NIH Chest X-ray", use_column_width=True)

# Button to trigger 3D Visualization
if st.button("Generate 3D Visualization"):
    with st.spinner("Estimating depth and generating 3D plot..."):
        depth_map = estimate_depth(image_path)  # Estimate depth using MiDaS
        fig = plot_depth_map(depth_map)  # Plot depth map in 3D
        st.plotly_chart(fig, use_container_width=True)
    st.success("3D visualization complete!")
