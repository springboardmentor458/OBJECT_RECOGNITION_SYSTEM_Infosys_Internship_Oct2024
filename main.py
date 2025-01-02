import torch
import cv2
import streamlit as st
import tempfile
from PIL import Image
import numpy as np

# Configure the Streamlit app
st.set_page_config(
    page_title="Object Detection with YOLOv5",
    page_icon="🎯",
    layout="centered"
)

# Custom styles for the app
st.markdown(
    """
    <style>
        body {
            background-color: #f0f2f6;
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        .stApp h1 {
            text-align: center;
            color: #3366ff;
            font-size: 2.5rem;
        }
        .stApp h3 {
            text-align: center;
            color: #444;
        }
        .stButton > button {
            background-color: #3366ff;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            padding: 0.5rem 1rem;
            cursor: pointer;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background-color: #254eda;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5m.pt', force_reload=True)

model = load_model()

# App title and introduction
st.title("YOLOv5 Object Detection")
st.write("📷 Upload an image, video, or select the webcam to detect objects. The model will annotate with bounding boxes and labels.")

# Sidebar for selecting input type
input_type = st.sidebar.radio("Select Input Type", ("Image", "Video", "Webcam"))

# Handle image input
if input_type == "Image":
    uploaded_image = st.file_uploader("Upload your image (JPEG/PNG):", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.write("🔄 **Processing your image...**")

        # Open and display the uploaded image
        input_image = Image.open(uploaded_image)
        st.image(input_image, caption="Original Image", use_container_width=True)

        # YOLOv5 inference
        results = model(input_image)

        # Render the results
        results.render()
        processed_image = Image.fromarray(results.ims[0])  # Extract annotated image

        st.write("✅ **Processing complete!**")

        # Display the processed image
        st.image(processed_image, caption="Annotated Image", use_container_width=True)

        # Provide a download button for the processed image
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        processed_image.save(temp_file.name, format="PNG")

        st.download_button(
            label="📥 Download Annotated Image",
            data=open(temp_file.name, "rb"),
            file_name="annotated_image.png",
            mime="image/png"
        )

# Handle video input
elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload your video (MP4/AVI):", type=["mp4", "avi"])

    if uploaded_video is not None:
        st.write("🔄 **Processing your video...**")

        # Save the uploaded video to a temporary file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_video.read())
        temp_video.close()

        # Open video using OpenCV
        cap = cv2.VideoCapture(temp_video.name)
        
        # Get video properties (frame width, height, and FPS)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define output video settings
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
        temp_processed_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out = cv2.VideoWriter(temp_processed_video.name, fourcc, fps, (width, height))

        # Stream the processed video
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference using YOLOv5
            results = model(frame)  # Frame is already a numpy array

            # Render bounding boxes and labels
            results.render()  # Render the bounding boxes and labels on the image
            processed_frame = results.ims[0]  # Get the annotated frame
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

            # Write the processed frame to the output video
            out.write(processed_frame)

            # Display the processed frame
            stframe.image(processed_frame, channels="BGR", use_container_width=True)

        # Release resources
        cap.release()
        out.release()

        st.write("✅ **Video processing complete!**")

        # Provide a download link for the processed video
        with open(temp_processed_video.name, "rb") as video_file:
            st.download_button(
                label="📥 Download Processed Video",
                data=video_file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

# Handle webcam input
elif input_type == "Webcam":
    st.write("🔄 **Accessing webcam...**")

    # Open webcam using OpenCV
    cap = cv2.VideoCapture(0)  # Open default webcam

    if not cap.isOpened():
        st.error("Unable to access the webcam. Please check your device settings.")
    else:
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference using YOLOv5
            results = model(frame)

            # Render bounding boxes and labels
            results.render()
            processed_frame = results.ims[0]  # Get the annotated frame
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

            # Display the processed frame
            stframe.image(processed_frame, channels="BGR", use_container_width=True)

        cap.release()
