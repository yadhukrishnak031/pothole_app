import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load model (cached so it loads once)
@st.cache_resource
def load_model():
    return YOLO(r"C:\Users\krish\OneDrive\Desktop\pothole_app\best.pt")

model = load_model()

st.title("Pothole Detection App")
st.write("Upload an image to detect potholes and classify by size")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image_np = np.array(image)

    results = model(image_np)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    count = len(boxes)

    small = 0
    medium = 0
    large = 0

    h, w, _ = image_np.shape
    image_area = h * w

    for box in boxes:
        x1, y1, x2, y2 = box
        box_area = (x2 - x1) * (y2 - y1)
        ratio = box_area / image_area

        if ratio < 0.02:
            small += 1
        elif ratio < 0.15:
            medium += 1
        else:
            large += 1

    annotated_frame = results[0].plot()

    st.image(annotated_frame, caption="Detected Image", use_column_width=True)

    st.write(f"### Total Potholes: {count}")
    st.write(f"Small: {small}")
    st.write(f"Medium: {medium}")
    st.write(f"Large: {large}")