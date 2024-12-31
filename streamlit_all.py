import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
import tempfile

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')

model = load_model()

st.title("YOLO Object Detection")

# Sidebar for mode selection
mode = st.sidebar.selectbox("Choose Mode", ["Image Detection", "Video Upload", "Webcam Detection"])

if mode == "Image Detection":
    # Image uploader
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        # Load image with PIL
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert image to numpy array
        image_array = np.array(image)

        # Perform object detection
        st.write("Processing image, please wait...")
        results = model(image_array)

        # Render detections on the image
        rendered_image = results.render()[0]
        rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)

        # Display the processed image
        st.image(rendered_image, caption="Detected Image", use_column_width=True)

elif mode == "Video Upload":
    # Video file uploader
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file:
        # Save the uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        st.write("Processing video, please wait...")

        # Open video and process frame-by-frame
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            st.error("Could not process the uploaded video.")
        else:
            # Prepare output display
            output_frames = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert frame to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Perform object detection
                results = model(rgb_frame)

                # Render detections on the frame
                rendered_frame = results.render()[0]
                rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)

                # Display the processed frame
                output_frames.image(rendered_frame, channels="BGR")

            cap.release()

elif mode == "Webcam Detection":
    # Webcam detection mode
    st.write("Press 'q' in the webcam window to stop detection.")

    # Access the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access the webcam.")
    else:
        # Placeholder for displaying frames
        frame_window = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam.")
                break

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform object detection
            results = model(rgb_frame)

            # Render detections on the frame
            rendered_frame = results.render()[0]
            rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)

            # Update Streamlit frame
            frame_window.image(rendered_frame, channels="BGR")

            # Stop detection with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
