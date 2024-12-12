import torch
from PIL import Image
import cv2
import time
import numpy as np
import io
import streamlit as st

# Set page config for the futuristic theme
st.set_page_config(
    page_title="AI Object Detection - YOLOv5",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for futuristic theme and image container styling
st.markdown("""
<style>
    /* Background */
    body {
        background-color: #121212;
        color: white;
        font-family: 'Roboto', sans-serif;
    }

    /* Header Styling */
    .stApp h1 {
        text-align: center;
        font-size: 3rem;
        color: #00f5d4;
        font-weight: bold;
        text-shadow: 0px 0px 20px rgba(0, 255, 255, 0.7);
    }

    /* Button Styling */
    .stButton > button {
        background-color: #00f5d4;
        color: black;
        font-size: 18px;
        border-radius: 8px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease-in-out;
    }

    .stButton > button:hover {
        background-color: #00b8a9;
        box-shadow: 0px 8px 16px rgba(0, 255, 255, 0.5);
        text-shadow: 0px 0px 15px rgba(0, 255, 255, 0.7);
    }

    /* Image container with padding */
    .image-container {
        # display: inline-block;
        # padding: 10px;
        border: 2px solid #00f5d4;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0, 255, 255, 0.5);
    }

    /* Neon border for images */
    .stImage {
        border-radius: 8px;
        box-shadow: 0px 4px 10px rgba(0, 255, 255, 0.5);
        transition: all 0.3s ease-in-out;
    }

    .stImage:hover {
        box-shadow: 0px 0px 20px rgba(0, 255, 255, 0.7);
    }

    /* Glow effect for drag-and-drop area */
    .stFileUploader {
        border: 2px dashed #00f5d4;
        padding: 20px;
        text-align: center;
        color: #00f5d4;
        border-radius: 12px;
        transition: all 0.3s ease;
    }

    /* Hover effect for glowing drag-and-drop area */
    .stFileUploader:hover {
        box-shadow: 0px 0px 20px rgba(0, 255, 255, 0.7);
    }
    
</style>
""", unsafe_allow_html=True)

# Display only the second file uploader
st.sidebar.title("Control Panel")
use_webcam = st.sidebar.checkbox("Use Webcam")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Load the pretrained YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5m')  # Load medium YOLOv5 model

model = load_model()

# Function to process image frame
def process_image_frame(image):
    # Perform YOLOv5 inference
    results = model(image)

    # Render the results
    results.render()  # Updates results.ims with rendered images
    img_with_labels = results.ims[0]  # Get the annotated image

    return img_with_labels

# Streamlit UI setup
st.title("AI Object Detection - YOLOv5")
st.write("Use your webcam or upload an image to detect objects using the YOLOv5 model. Powered by AI!")

# Image Upload Handling
if uploaded_file is not None and not use_webcam:
    image = Image.open(uploaded_file)

    # Displaying the images side by side using st.columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)  # Start container for uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)  # End container for uploaded image

    # Process and display the image
    img_with_labels = process_image_frame(image)
    img_with_labels = Image.fromarray(img_with_labels)

    with col2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)  # Start container for processed image
        st.image(img_with_labels, caption="Processed Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)  # End container for processed image

    # Allow the user to download the processed image
    buffer = io.BytesIO()
    img_with_labels.save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        label="Download Processed Image",
        data=buffer,
        file_name="processed_image.png",
        mime="image/png"
    )

# Webcam Handling
if use_webcam:
    st.write("Capturing from webcam...")

    # Start the webcam
    video_capture = cv2.VideoCapture(0)

    # Check if the webcam is opened
    if not video_capture.isOpened():
        st.error("Could not access the webcam. Please check your device.")
    else:
        frame_placeholder = st.empty()
        stop_button = st.button("Stop Webcam", key="stop_webcam", help="Click to stop the webcam feed", use_container_width=True)

        # Add custom class to the Stop Webcam button
        if stop_button:
            video_capture.release()
            st.write("Webcam feed stopped.")
        else:
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    st.error("Failed to capture frame.")
                    break

                # Process the frame and perform YOLOv5 inference
                img_with_labels = process_image_frame(frame)

                # Convert the annotated frame to a PIL image
                img_with_labels = Image.fromarray(img_with_labels)

                # Display processed frames directly
                frame_placeholder.image(img_with_labels, caption="Processed Webcam Frame", use_container_width=True)

                # Add a small delay to manage frame updates
                time.sleep(0.05)
