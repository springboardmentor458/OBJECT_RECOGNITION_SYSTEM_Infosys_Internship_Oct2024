import torch
from PIL import Image
import cv2
import time
import numpy as np
import tempfile
import streamlit as st
import base64
import os

# Set page config for the futuristic theme
port = int(os.environ.get('PORT', 8501))
st.set_page_config(
    page_title="Object Detection System",
    page_icon="🤖",
    layout="wide"
)
# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Set background image
def set_background_image(image_path):
    img_base64 = image_to_base64(image_path)
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url('data:image/jpg;base64,{img_base64}');
                background-size: cover;
                background-position: center;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the path to your background image
background_image_path = "./bg3.jpg"

# Apply the background image
set_background_image(background_image_path)

# Custom CSS for futuristic theme and to control video size
st.markdown(
    """
    <style>
        body {
            margin: 0px;
            padding: 0px;
            background-color: #121212;  
            color: white;
            font-family: 'Roboto', sans-serif;
        }
        .main-title {
            font-size: 3rem;  
            text-align: center;
            color: #00f5d4;
            font-weight: bold;
        }
        .st-emotion-cache-1mw54nq h1{
            color: #00f5d4;
            text-align: center;
            font-size: 28px;
            margin-bottom: 15px
        }
        h1#object-detection-system.main-title{
            color: #00f5d4;
        }
        .stApp p{
            text-align: center;
            color: white;
        }
        .custom-text {
            font-size: 20px;
            text-align: center;
            color: white;
        }
        .stButton > button {
            background-color: #00f5d4;
            color: black;
            font-size: 18px;
            border-radius: 8px;
            transition: all 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #00b8a9;
        }
        .stImage, .stVideo, .stWebcam {
            width: 80%;  /* Set the width */
            height: 80%;  /* Maintain aspect ratio */
        }
        .st-emotion-cache-1ibsh2c{
            padding: 30px
        }
        .st-emotion-cache-12fmjuu {
            display: none;
        }
        [data-testid="stSidebar"] {
            background-color: #000046;  
        }
        .stFileUploader {
            background-color: #000056;  /* Replace with your desired color */
            padding: 10px;
            border-radius: 5px;
            border: 1px dashed #00f5d4;
        }
        .st-emotion-cache-taue2i {
            background-color: #000069;
            color: #e6e6e6;
            font-size: 15px;
        }
        .st-emotion-cache-1aehpvj {
            color: rgba(204, 204, 204, 0.7);
            font-size: 14px;
        }
        .st-emotion-cache-zaw6nw {
            color: #00004d;
            background-color: #00f5d4;
            border: 1px solid rgba(49, 51, 63, 0.2);
        }
        .center-button {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: 20px;
            width: 100%;
        }
         .center-button button {
            color:  #00004d;
        }
        .st-emotion-cache-1v45yng .es2srfl9 {
            width: 100%;
            color: #e6e6e6;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the pretrained YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5n')

model = load_model()

# Function to process frames
def process_frame(frame):
    # Convert frame (numpy array) to PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform YOLOv5 inference
    results = model(image)

    # Render the results on the frame
    results.render()
    processed_frame = results.ims[0]  # Annotated frame
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR

    return processed_frame

# Streamlit UI setup
st.markdown('<h1 class="main-title">Object Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="custom-text">Upload a video, use your webcam, or upload an image for real-time object detection.</p>', unsafe_allow_html=True)

# Sidebar for input options
st.sidebar.title("Control Panel")
use_webcam = st.sidebar.checkbox("Use Webcam")
uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Initialize webcam capture variable
video_capture = None

# Handle image upload
if uploaded_image is not None and not use_webcam and not uploaded_video:
    st.write("Processing uploaded image...")

    # Load and process the image
    input_image = Image.open(uploaded_image)
    processed_image = process_frame(np.array(input_image))

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='color: #00f5d4;'>Input Image</h3>", unsafe_allow_html=True)  # Add color to the header
        st.image(input_image, use_container_width=True)

    # Processed Image
    with col2:
        st.markdown("<h3 style='color: #00f5d4;'>Processed Image</h3>", unsafe_allow_html=True)  # Add color to the header
        st.image(processed_image, use_container_width=True)

    # Allow the user to download the processed image
    img_buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(processed_image).save(img_buffer, format="PNG")
    img_buffer.close()


# Handle video upload
if uploaded_video is not None and not use_webcam:
    st.write("Processing uploaded video...")

    # Save uploaded video to a temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_video.read())
    temp_video.close()

    # Open video using OpenCV
    cap = cv2.VideoCapture(temp_video.name)

    # Define output video settings
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video

    # Temporary file for processed video
    temp_processed_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_processed_video.name, fourcc, fps, (width, height))

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        processed_frame = process_frame(frame)

        # Write the processed frame to the output video
        out.write(processed_frame)

        # Display the processed frame
        stframe.image(processed_frame, channels="BGR", use_container_width=True)

    cap.release()
    out.release()

    st.write("Video processing complete.")

# Handle webcam input
if use_webcam:
    st.write("Capturing from webcam...")

    # Start the webcam if not already started
    if video_capture is None:
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 500)  # Set width
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 350)  # Set height

    if not video_capture.isOpened():
        st.error("Could not access the webcam. Please check your device.")
    else:
        stframe = st.empty()  # Placeholder for displaying frames
        st.markdown('<div class="center-button">', unsafe_allow_html=True)
        stop_webcam_button = st.button("Stop Webcam", key="stop_webcam")
        st.markdown('</div>', unsafe_allow_html=True)  # Button outside the loop

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                st.error("Failed to capture frame.")
                break

            # Remove mirroring
            frame = cv2.flip(frame, 1)

            # Process the frame using YOLOv5 inference
            processed_frame = process_frame(frame)

            # Display the processed frame in the placeholder
            stframe.image(processed_frame, channels="BGR", use_container_width=True)

            # Stop webcam if the button is pressed or checkbox is unchecked
            if stop_webcam_button or not use_webcam:

                video_capture.release()  # Release the webcam
                stframe.empty()  # Clear the frame
                st.write("Webcam feed stopped.")
                break

# Release the webcam if it's still open
if video_capture and video_capture.isOpened():
    video_capture.release()
