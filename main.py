import torch
from PIL import Image
import cv2
import numpy as np
import tempfile
import streamlit as st
import time

# Set page configuration for theme and layout
st.set_page_config(
    page_title="Object Detection System",
    layout="wide"
)

# CSS for theme and to control button and text appearance
st.markdown(
    """
    <style>
        body {
            background-color: #1e1e1e;
            color: white;
            font-family: 'Roboto', sans-serif;
            font-size: 18px;
        }
        .stApp h1 {
            text-align: center;
            font-size: 3rem;
            color: #0f69fa;
            font-weight: bold;
        }
        .stButton > button {
            background-color: #0f69fa;
            color: black;
            font-size: 18px;
            border-radius: 8px;
            transition: all 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #0f69fa;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the YOLOv5 model, utilizing GPU if available
@st.cache_resource
def load_model():
    # Load the YOLOv5 medium model; use CUDA if GPU is available, else CPU
    return torch.hub.load('ultralytics/yolov5', 'yolov5m', device='cuda' if torch.cuda.is_available() else 'cpu')

# Load the model into memory
model = load_model()

# Function to process a single frame or image
def process_frame(frame):
    """
    Process a single frame for object detection System.
    Args:
        frame: Input frame (numpy array, BGR format).
    Returns:
        Processed frame with YOLOv5 detections rendered.
    """
    # Convert BGR frame to RGB and create a PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = model(image)
    results.render()
    # Convert the processed frame back to BGR format for OpenCV compatibility
    processed_frame = results.ims[0]
    return cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)

# Streamlit UI - Title and instructions
st.title("Object Detection System")
st.write("Upload a video, an image, or use your webcam for real-time object detection.")

# Sidebar configuration for selecting input options
st.sidebar.title("Panel")
use_webcam = st.sidebar.checkbox("Use Webcam") 
uploaded_video = st.sidebar.file_uploader("Upload a video")
uploaded_image = st.sidebar.file_uploader("Upload an image")  

# Handle image upload for object detection
if uploaded_image is not None and not use_webcam and not uploaded_video:
    st.write("Processing Image...")

    # Load the uploaded image and convert to numpy array
    input_image = Image.open(uploaded_image)
    input_frame = np.array(input_image)

    # Process the frame for object detection
    processed_image = process_frame(input_frame)

    # Create two columns for displaying 
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(input_image, use_container_width=True)

    with col2:
        st.subheader("Processed Image")
        st.image(processed_image, use_container_width=True)

    # Save the processed image to a temporary file and allow download
    img_buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(processed_image).save(img_buffer, format="PNG")
    img_buffer.close()

    st.download_button(
        label="Download Processed Image",
        data=open(img_buffer.name, "rb"),
        file_name="processed_image.png",
        mime="image/png"
    )

# Handle video upload for object detection
if uploaded_video is not None and not use_webcam:
    st.write("Processing Video...")

    # Save uploaded video to a temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_video.read())
    temp_video.close()

    # Open video using OpenCV
    cap = cv2.VideoCapture(temp_video.name)

    # Get video properties for processing
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  

    # Prepare a temporary file for the processed video
    temp_processed_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_processed_video.name, fourcc, fps, (width, height))

    # Initialize a Streamlit container for displaying video frames
    stframe = st.empty()
    frame_count = 0
    start_time = time.time()

    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)

        out.write(processed_frame)

        # Display the processed frame in the Streamlit app
        stframe.image(processed_frame, channels="BGR", use_container_width=True)

        # FPS tracking for performance monitoring
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            st.write(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds (FPS: {frame_count / elapsed_time:.2f})")

    cap.release()
    out.release()

    st.write("Video processing complete.")
    with open(temp_processed_video.name, "rb") as video_file:
        st.download_button(
            label="Download Processed Video",
            data=video_file,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )
if use_webcam:
    st.write("Capturing from webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        st.error("Could not access the webcam. Please check your device.")
    else:
        stframe = st.empty()  
        stop_webcam_button = st.button("Stop Webcam", key="stop_webcam")  

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame.")
                break

            # Mirror the frame for natural webcam display
            frame = cv2.flip(frame, 1)

            processed_frame = process_frame(frame)

            # Display the processed frame in the placeholder
            stframe.image(processed_frame, channels="BGR", use_container_width=True)

            if stop_webcam_button:
                cap.release()
                stframe.empty()
                st.write("Webcam feed stopped.")
                break

    cap.release()  
