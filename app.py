import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ocr import LiscencePlateExtractor  # your OCR class
from yolo import LicensePlateDetector
# from cascade import LiscencePlateDetector

# Page config
st.set_page_config(page_title="License Plate OCR", layout="wide")
st.title("🚘 License Plate Detection & Reader")

# Object detection model
detector = LicensePlateDetector()
# OCR model
reader = LiscencePlateExtractor()

# Top row: Upload + preview side-by-side
col1, col2, col3 = st.columns(3)

with col1:
    uploaded_file = st.file_uploader("Upload Vehicle Image", type=["jpg", "jpeg", "png"])

with col3:
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)

# Process image and show intermediate steps
if uploaded_file and st.button("🔍 Process License Plate"):
    with st.spinner("Processing..."):

        # Convert image to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        plate = detector.detect_highest_conf_plate(img_cv)
        if plate is not None:
            chars, intermediates = reader.extract(plate)
            if chars!=None:

                # 2 rows × 3 columns grid for intermediate images
                rows = [st.columns(3), st.columns(3)]
                for i in range(6):
                    col = rows[i // 3][i % 3]
                    with col:
                        st.image(intermediates[i], caption='image', use_container_width=True, channels="GRAY")

                text = reader.predict(chars)

                # Output text below
                st.markdown("### 📛 Detected License Plate Text")
                st.success(text if text else "No text detected.")

            else:
                st.success("Enough Characters not Found")
        else:
            st.success("Number Plate Not Found")