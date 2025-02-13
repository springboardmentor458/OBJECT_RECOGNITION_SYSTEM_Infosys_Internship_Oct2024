import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import tempfile
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("YOLOv5_Detection")

# Set page configuration
st.set_page_config(page_title="YOLOv5 Object Detection", layout="wide")

# Add custom CSS for styling
def add_custom_css():
    st.markdown(
        """
        <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f7f9fc;
            color: #333;
        }
        h1, h2, h3, h4 { color: #0056b3; }
        .stButton > button {
            background-color: #0056b3;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 8px 16px;
        }
        .stButton > button:hover { background-color: #003f8a; }
        .stSidebar {
            background-color: #ffffff;
            border-right: 1px solid #ddd;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_resource
def load_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        model.conf = 0.25
        model.iou = 0.45
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def process_frame(model, frame):
    """Process a single frame for object detection"""
    try:
        results = model(frame)
        detections = []
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            label = results.names[int(cls)]
            detections.append({
                'class': label,
                'confidence': float(conf),
                'bbox': [x1, y1, x2, y2],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        annotated_frame = results.render()[0]
        return detections, annotated_frame
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return [], frame

class WebcamDetector:
    def __init__(self, model):
        self.model = model

    def start(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Unable to access webcam. Please check permissions and connections.")
            return

        # Initialize session state for controlling webcam
        if "webcam_running" not in st.session_state:
            st.session_state["webcam_running"] = True

        # Create placeholders for frames and statistics
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()

        # Stop button (placed outside the loop to avoid re-creation)
        stop_button_clicked = st.button("Stop Webcam", key="unique_stop_button")

        try:
            all_detections = []
            while st.session_state["webcam_running"]:
                # Handle stop button
                if stop_button_clicked:
                    st.session_state["webcam_running"] = False
                    break

                # Read frame from webcam
                ret, frame = cap.read()
                if not ret:
                    break

                # Process the frame for detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections, annotated_frame = process_frame(self.model, frame_rgb)
                all_detections.extend(detections)

                # Update frame and statistics in the UI
                frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

                if all_detections:
                    df = pd.DataFrame(all_detections)
                    stats_placeholder.dataframe(
                        df[['class', 'confidence', 'timestamp']].sort_values('timestamp', ascending=False),
                        use_container_width=True
                    )

        except Exception as e:
            logger.error(f"Webcam detection error: {str(e)}")
            st.error(f"Webcam detection error: {str(e)}")
        finally:
            cap.release()
            # Clear placeholders after stopping webcam
            frame_placeholder.empty()
            stats_placeholder.empty()
            st.session_state["webcam_running"] = False

def process_video_realtime(model, video_file):
    """Process video in real-time"""
    try:
        # Create temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Error opening video file")
            return

        # Create placeholders
        frame_placeholder = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, annotated_frame = process_frame(model, frame_rgb)
            
            # Display frame
            frame_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        logger.error(f"Error processing video: {str(e)}")
    finally:
        cap.release()
        frame_placeholder.empty()

def main():
    add_custom_css()
    
    model = load_model()
    if not model:
        st.error("Failed to load model. Please refresh the page.")
        return

    st.title("YOLOv5 Object Detection")
    
    st.sidebar.header("Detection Settings")
    detection_type = st.sidebar.selectbox("Detection Mode", ["Image", "Webcam", "Video"])
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.25, 
        step=0.01
    )
    model.conf = confidence_threshold

    if detection_type == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                if len(image_np.shape) == 3 and image_np.shape[2] == 4:
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_container_width=True)

                with st.spinner('Detecting objects...'):
                    detections, annotated_image = process_frame(model, image_np)

                with col2:
                    st.subheader("Detected Objects")
                    st.image(annotated_image, use_container_width=True)

                if detections:
                    st.subheader("Detection Details")
                    df = pd.DataFrame(detections)
                    st.dataframe(df[['class', 'confidence']], use_container_width=True)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

    elif detection_type == "Webcam":
        st.subheader("Webcam Object Detection")
        
        # Using a container to manage button state
        start_button_container = st.empty()
        
        if start_button_container.button("Start Webcam", key="unique_start_button"):
            start_button_container.empty()  # Remove the start button
            detector = WebcamDetector(model)
            detector.start()
            # Recreate the start button after webcam stops
            start_button_container.button("Start Webcam", key="unique_start_button_after")

    elif detection_type == "Video":
        st.subheader("Video Object Detection")
        uploaded_video = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])

        if uploaded_video:
            process_video_realtime(model, uploaded_video)

if __name__ == "__main__":
    main()
