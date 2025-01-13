import torch
from PIL import Image
import cv2
import time
import numpy as np
import tempfile
import streamlit as st

# Set page config for the application
st.set_page_config(
    page_title="Object Detection App",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #0e2431;
            color: #f4f4f9;
            font-family: 'Arial', sans-serif;
        }
        .stApp h1 {
            text-align: center;
            font-size: 2.8rem;
            color: #ffa62b;
            font-weight: bold;
        }
        .stButton > button {
            background-color: #ffa62b;
            color: black;
            font-size: 16px;
            border-radius: 5px;
            transition: all 0.3s ease-in-out;
        }
        .stButton > button:hover {
            background-color: #cc8400;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the pretrained YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')

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
st.title("AI Object Detection Application")
st.write("Upload a video, use your webcam, or upload an image for real-time object detection.")

# Sidebar for input options
st.sidebar.title("Input Options")
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

    # Display input and processed images side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(input_image, use_container_width=True)

    with col2:
        st.subheader("Processed Image")
        st.image(processed_image, use_container_width=True)

    # Allow the user to download the processed image
    img_buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(processed_image).save(img_buffer, format="PNG")
    img_buffer.close()

    st.download_button(
        label="Download Processed Image",
        data=open(img_buffer.name, "rb"),
        file_name="processed_image.png",
        mime="image/png"
    )

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
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video

    # Temporary file for processed video
    temp_processed_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_processed_video.name, fourcc, fps, (width, height))

    stframe = st.empty()

    frame_duration = 1 / fps  # Time per frame in seconds

    while cap.isOpened():
        start_time = time.time()  # Start timer

        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        processed_frame = process_frame(frame)

        # Write the processed frame to the output video
        out.write(processed_frame)

        # Display the processed frame
        stframe.image(processed_frame, channels="BGR", use_container_width=True)

        # Ensure consistent playback speed
        elapsed_time = time.time() - start_time
        delay = max(0, frame_duration - elapsed_time)  # Adjust delay to match frame rate
        time.sleep(delay)

    cap.release()
    out.release()

    st.write("Video processing complete.")

    # Allow the user to download the processed video
    with open(temp_processed_video.name, "rb") as video_file:
        st.download_button(
            label="Download Processed Video",
            data=video_file,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

# Handle webcam input
if use_webcam:
    st.write("Capturing from webcam...")

    # Start the webcam if not already started
    if video_capture is None:
        video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        st.error("Could not access the webcam. Please check your device.")
    else:
        stframe = st.empty()  # Placeholder for displaying frames
        stop_webcam_button = st.button("Stop Webcam", key="stop_webcam")  # Button outside the loop

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

