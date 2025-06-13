import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Load classification model
model = YOLO("runs/classify/train/weights/best.pt")  # Must be trained in classification mode

st.title("🧬 Retinal Disease Classification")
st.write("Upload a retinal image to classify diseases.")

uploaded_file = st.file_uploader("📷 Choose a retinal image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    st.write("🔍 Analyzing...")
    results = model.predict(source=img_np, conf=0.2)

    # Show top class
    top_result = results[0]
    class_id = int(top_result.probs.top1)
    class_conf = float(top_result.probs.top1conf)
    class_name = model.names[class_id]

    st.success(f"✅ Predicted Class: **{class_name}** ({class_conf*100:.2f}% confidence)")
    st.image(img, caption="Retinal Image", use_container_width=True)
