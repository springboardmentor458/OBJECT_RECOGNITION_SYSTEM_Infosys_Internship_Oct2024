import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
from time import time

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to process an image
def process_image(image):
    # Convert the image to RGB (YOLOv5 works with RGB images)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(rgb_image)

    # Get the results
    detections = results.pandas().xyxy[0]  # Bounding boxes with confidence and labels

    # Draw bounding boxes and labels on the image
    for _, row in detections.iterrows():
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = row['confidence']

        # Draw rectangle
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Draw label and confidence
        cv2.putText(image, f"{label} {confidence:.2f}", (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

# Streamlit UI
st.title("YOLOv5 Object Detection")
st.sidebar.title("Settings")
source = st.sidebar.radio("Select Source", ("Image", "Video", "Webcam"))

# Upload and process an image
if source == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, channels="BGR", caption="Uploaded Image")
        processed_image = process_image(image)
        st.image(processed_image, channels="BGR", caption="Detected Objects")

# Upload and process a video
elif source == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        video_path = f"temp_video.{uploaded_video.name.split('.')[-1]}"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.video(video_path)
        cap = cv2.VideoCapture(video_path)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_image(frame)
            stframe.image(processed_frame, channels="BGR", use_column_width=True)

        cap.release()

# Process webcam feed
elif source == "Webcam":
    stframe = st.empty()
    cap = cv2.VideoCapture(0)  # Use webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Unable to access webcam")
            break

        start_time = time()

        # Process the frame
        processed_frame = process_image(frame)

        # Calculate FPS
        fps = 1 / (time() - start_time)

        # Display FPS on the frame
        cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        stframe.image(processed_frame, channels="BGR", use_column_width=True)

    cap.release()
