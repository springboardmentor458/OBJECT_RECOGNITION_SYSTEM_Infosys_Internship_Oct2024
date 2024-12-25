import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import torch
import numpy as np

def load_model():
    return YOLO(r"C:\Users\harsh\Downloads\flask\yolov5s.pt")  # Use pre-trained YOLOv5s model
model = load_model()


# Function to process images
def process_image(image):
    results = model(image)
    annotated_image = results[0].plot()
    return annotated_image

# Function to process video
def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    return temp_output.name

# Real-time prediction with webcam
def real_time_prediction():
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Camera not found.")
            break
        results = model(frame)
        annotated_frame = results[0].plot()

        # Convert BGR to RGB for Streamlit display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame, channels="RGB", use_column_width=True)

    cap.release()

# Streamlit App
st.title("YOLOv5 Object Detection App")

# Sidebar
st.sidebar.title("Features")
feature = st.sidebar.radio(
    "Select a feature:",
    ("Image Upload and Prediction", "Real-Time Prediction", "Video Upload and Prediction")
)

# Feature: Image Upload and Prediction
if feature == "Image Upload and Prediction":
    st.header("Image Upload and Prediction")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        annotated_image = process_image(image)
        st.image(annotated_image, channels="BGR", caption="Predicted Image")

# Feature: Real-Time Prediction
elif feature == "Real-Time Prediction":
    st.header("Real-Time Prediction")
    if st.button("Start Real-Time Prediction"):
        real_time_prediction()

# Feature: Video Upload and Prediction
elif feature == "Video Upload and Prediction":
    st.header("Video Upload and Prediction")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(uploaded_video.read())
        st.write("Processing video...")
        output_video_path = process_video(temp_input.name)
        st.video(output_video_path)
