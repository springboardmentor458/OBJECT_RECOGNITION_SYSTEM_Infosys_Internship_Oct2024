import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
import tempfile
import os

# Custom CSS Styling
st.markdown(
    """
    <style>
        .stApp { background-color: #000000 !important; }
        h1, h2, h3, h4, h5, h6 { color: #FFD700 !important; }
        p, span { color: #FFD700 !important; font-size: 16px; }
        .stButton>button { 
            background-color: #000000 !important; 
            color: #FFD700 !important; 
            font-weight: bold; 
            border-radius: 8px; 
            border: 2px solid #FFD700; 
        }
        .stButton>button:hover {
            background-color: #222222 !important;
            color: #FFD700 !important;
        }
        [data-testid="stSidebar"] { background-color: #FFFFFF !important; }
        [data-testid="stSidebar"] * { color: #000000 !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# Load YOLOv5 Model
@st.cache_resource
def load_model():
    model = torch.hub.load('yolov5', 'yolov5s', source='local')  # Ensure YOLOv5 repo is in the same directory
    return model

model = load_model()

# Streamlit UI
st.title("Object Detection using YOLOv5")
st.sidebar.title("Options")
choice = st.sidebar.radio("Select an option:", ["Webcam", "Image Upload", "Video Upload"])

# Function to process and detect objects on an image
def detect_objects_image(image):
    results = model(image)  # Perform object detection
    img = np.squeeze(results.render())  # Render results
    return img

# Function to process video and detect objects frame by frame
def detect_objects_video(video_path):
    cap = cv2.VideoCapture(video_path)
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output.name, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)  # Perform detection
        frame = np.squeeze(results.render())  # Render results
        out.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    out.release()
    return temp_output.name  # Return path of the processed video

# Webcam Option
if choice == "Webcam":
    st.sidebar.write("Webcam Object Detection")
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    if run:
        webcam = cv2.VideoCapture(0)
        while True:
            _, frame = webcam.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame)
            frame = np.squeeze(results.render())
            FRAME_WINDOW.image(frame)
    else:
        st.warning("Click 'Start Webcam' to run real-time object detection.")

# Image Upload Option
elif choice == "Image Upload":
    st.sidebar.write("Upload an Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Detecting objects...")
        processed_image = detect_objects_image(image)
        st.image(processed_image, caption="Processed Image with Objects", use_column_width=True)

# Video Upload Option
elif choice == "Video Upload":
    st.sidebar.write("Upload a Video")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()

        st.video(uploaded_video)

        st.write("Processing video...")
        processed_video_path = detect_objects_video(tfile.name)
        st.success("Video processing complete!")

        # Display the processed video
        st.video(processed_video_path)

        # Provide a download button for the processed video
        with open(processed_video_path, "rb") as video_file:
            st.download_button(
                label="Download Processed Video",
                data=video_file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
