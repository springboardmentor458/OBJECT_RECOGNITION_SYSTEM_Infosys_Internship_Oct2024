import torch  # type: ignore
from PIL import Image
import cv2
import numpy as np
import tempfile
import streamlit as st

# Set page config with a unique title and modern layout
st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS for a more creative and unique design
st.markdown(
    """
    <style>
        /* Background */
        body {
            background: linear-gradient(135deg, #4f8cff, #2a3d8d);
            color: #fff;
            font-family: 'Arial', sans-serif;
        }

        /* Title Style */
        .stApp h1 {
            font-size: 4rem;
            font-weight: 900;
            text-align: center;
            color: #fff;
            text-transform: uppercase;
            letter-spacing: 5px;
            margin-top: 50px;
        }

        .stApp h2 {
            font-size: 1.8rem;
            font-weight: bold;
            text-align: center;
            color: #dbe7f0;
        }

        /* Buttons */
        .stButton > button {
            background-color: #00e6e6;
            color: #111;
            font-size: 18px;
            font-weight: 600;
            border-radius: 30px;
            padding: 12px 28px;
            border: none;
            transition: transform 0.3s, background-color 0.3s;
        }

        .stButton > button:hover {
            background-color: #00b8b8;
            transform: scale(1.1);
        }

        /* Sidebar */
        .stSidebar {
            background-color: #0e0d0d;
            color: #fff;
            padding: 20px;
            border-radius: 20px;
        }

        .stSidebar .stTitle {
            font-size: 1.5rem;
            font-weight: 600;
            color: #00e6e6;
            text-align: center;
            margin-bottom: 15px;
        }

        .stSidebar .stCheckbox > div {
            font-size: 16px;
            color: #c8c8c8;
        }

        /* Images */
        .stImage {
            border-radius: 20px;
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.4);
            margin-bottom: 20px;
        }

        /* Video */
        .stVideo {
            border-radius: 20px;
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.4);
        }

        /* Download button */
        .stDownloadButton > button {
            background-color: #00e6e6;
            color: #111;
            font-size: 16px;
            border-radius: 30px;
            padding: 10px 25px;
            border: none;
            transition: transform 0.3s, background-color 0.3s;
        }

        .stDownloadButton > button:hover {
            background-color: #00b8b8;
            transform: scale(1.1);
        }

        /* Overall container */
        .stApp {
            padding-bottom: 50px;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .stApp h1 {
                font-size: 2.5rem;
            }
            .stApp h2 {
                font-size: 1.5rem;
            }
            .stButton > button {
                font-size: 16px;
            }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load YOLOv5 model
def load_yolo_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5m')

model = load_yolo_model()

# Process frames for detection
def process_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = model(image)
    results.render()
    return cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)

# Streamlit UI setup
st.title("🚀 YOLOv5 Object Detection")
st.write("Welcome to the futuristic object detection tool powered by YOLOv5. Upload your images, videos or use the webcam to see object detection in action!")

# Sidebar for input options
st.sidebar.title("Control Panel")
use_webcam = st.sidebar.checkbox("Use Webcam")
uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Webcam capture initialization
video_capture = None

# Handle uploaded image
if uploaded_image and not use_webcam and not uploaded_video:
    st.write("Processing uploaded image...")
    input_image = Image.open(uploaded_image)
    processed_image = process_frame(np.array(input_image))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(input_image, use_container_width=True, caption="Original Image")

    with col2:
        st.subheader("Detected Image")
        st.image(processed_image, use_container_width=True, caption="Processed Image")

    img_buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    Image.fromarray(processed_image).save(img_buffer, format="PNG")
    st.download_button(
        "Download Processed Image",
        open(img_buffer.name, "rb"),
        file_name="processed_image.png",
        mime="image/png",
        key="download_img"
    )

# Handle uploaded video
if uploaded_video and not use_webcam:
    st.write("Processing uploaded video...")
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_video.read())
    temp_video.close()

    cap = cv2.VideoCapture(temp_video.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    temp_processed_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_processed_video.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(process_frame(frame))
        stframe.image(process_frame(frame), channels="BGR", use_container_width=True, caption="Processed Video")

    cap.release()
    out.release()
    st.write("Video processed successfully.")

    st.download_button(
        "Download Processed Video",
        open(temp_processed_video.name, "rb"),
        file_name="processed_video.mp4",
        mime="video/mp4",
        key="download_vid"
    )

# Handle webcam input
if use_webcam:
    st.write("Using Webcam... Let's detect objects in real-time.")
    video_capture = video_capture or cv2.VideoCapture(0)

    if not video_capture.isOpened():
        st.error("Unable to access the webcam.")
    else:
        stframe = st.empty()
        stop_webcam_button = st.sidebar.button("Stop Webcam")

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                st.error("Failed to capture frame.")
                break
            stframe.image(process_frame(cv2.flip(frame, 1)), channels="BGR", use_container_width=True, caption="Real-time Detection")
            if stop_webcam_button:
                video_capture.release()
                stframe.empty()
                break

if video_capture and video_capture.isOpened():
    video_capture.release()
