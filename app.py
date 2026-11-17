
from pathlib import Path
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image
from ultralytics import YOLO


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "best.pt"

IMAGE_SIZE = 512

st.set_page_config(
    page_title="Pothole Detection System",
    page_icon="🚧",
    layout="wide"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("Pothole Detection System")

st.markdown(
    """
    Upload a road image to detect potholes using a
    YOLOv8 deep learning model.
    """
)

st.divider()


# --------------------------------------------------
# MODEL CHECK
# --------------------------------------------------

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: {MODEL_PATH}\n\n"
        "Copy your selected best.pt checkpoint into the models folder."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

st.sidebar.header("Detection Settings")

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
    help="Minimum confidence required to display a detection."
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05,
    help="Controls Non-Maximum Suppression."
)

st.sidebar.write("Model: YOLOv8n")
st.sidebar.write("Input size: 512 × 512")
st.sidebar.write("Detection class: Pothole")


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        original_image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Input Image")

        st.image(
            original_image,
            caption="Uploaded road image",
            use_container_width=True
        )

        # ------------------------------------------
        # DETECTION
        # ------------------------------------------

        if st.button("Detect Potholes", type="primary"):

            with st.spinner("Analyzing image..."):

                results = model.predict(
                    source=original_image,
                    imgsz=IMAGE_SIZE,
                    conf=confidence,
                    iou=iou_threshold,
                    max_det=300,
                    verbose=False
                )

                result = results[0]

                # Number of detections
                boxes = result.boxes

                detection_count = (
                    len(boxes) if boxes is not None else 0
                )

                # Annotated image
                annotated_bgr = result.plot()

                annotated_rgb = annotated_bgr[:, :, ::-1]

                annotated_image = Image.fromarray(
                    annotated_rgb
                )

                # ----------------------------------
                # DISPLAY RESULTS
                # ----------------------------------

                st.divider()
                st.subheader("Detection Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Detected Potholes",
                        detection_count
                    )

                with col2:
                    st.metric(
                        "Confidence Threshold",
                        f"{confidence:.2f}"
                    )

                if detection_count > 0:

                    st.success(
                        f"{detection_count} pothole detection(s) found."
                    )

                else:

                    st.info(
                        "No potholes detected at the selected "
                        "confidence threshold."
                    )

                # ----------------------------------
                # ANNOTATED IMAGE
                # ----------------------------------

                st.subheader("Annotated Image")

                st.image(
                    annotated_image,
                    caption="Pothole detection output",
                    use_container_width=True
                )

                # ----------------------------------
                # DETECTION DETAILS
                # ----------------------------------

                if detection_count > 0:

                    detection_data = []

                    for index, box in enumerate(boxes):

                        coordinates = (
                            box.xyxy[0].cpu().numpy()
                        )

                        score = float(
                            box.conf[0].cpu().item()
                        )

                        detection_data.append({
                            "Detection": index + 1,
                            "Class": "Pothole",
                            "Confidence": round(score, 4),
                            "X1": round(float(coordinates[0]), 2),
                            "Y1": round(float(coordinates[1]), 2),
                            "X2": round(float(coordinates[2]), 2),
                            "Y2": round(float(coordinates[3]), 2)
                        })

                    df = pd.DataFrame(detection_data)

                    st.subheader("Detection Details")

                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )

                # ----------------------------------
                # DOWNLOAD ANNOTATED IMAGE
                # ----------------------------------

                image_buffer = BytesIO()

                annotated_image.save(
                    image_buffer,
                    format="PNG"
                )

                st.download_button(
                    label="Download Annotated Image",
                    data=image_buffer.getvalue(),
                    file_name="pothole_detection_result.png",
                    mime="image/png"
                )

    except Exception as error:

        st.error(
            f"An error occurred while processing the image: {error}"
        )
