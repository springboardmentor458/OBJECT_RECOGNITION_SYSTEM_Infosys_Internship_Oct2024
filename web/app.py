import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, session, url_for, Response, flash
from flask_session import Session
from ultralytics import YOLO
from PIL import Image
import base64
import tempfile

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Ensure this path is correct

# Functions for inference on images
def infer_image(image):
    """
    Run YOLOv8 inference on an image and return annotated image and detections.
    """
    results = model.predict(source=image, save=False, conf=0.25)
    annotated_image = results[0].plot()
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        detections.append({
            'class': results[0].names[class_id],
            'confidence': round(confidence * 100, 2),
            'bounding_box': [x1, y1, x2, y2]
        })

    # Encode the annotated image to base64
    _, buffer = cv2.imencode('.png', annotated_image)
    img_data = base64.b64encode(buffer).decode('utf-8')
    return img_data, detections

# Function to process video frames and save output video
def process_video(video_path):
    """
    Run YOLOv8 inference on each frame of the video and save the annotated video.
    """
    cap = cv2.VideoCapture(video_path)
    frame_detections = []

    # Define output path
    temp_output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)  # Default FPS to 25 if not readable
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, save=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        # Collect detections
        frame_detection = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            frame_detection.append({
                'class': results[0].names[class_id],
                'confidence': round(confidence * 100, 2),
                'bounding_box': [x1, y1, x2, y2]
            })
        frame_detections.append(frame_detection)

    cap.release()
    out.release()
    return temp_output_path, frame_detections

# Generate live video feed for camera detection
def generate_frames():
    """
    Yield frames from the camera with YOLOv8 annotations for live video feed.
    """
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(source=frame, save=False)
        annotated_frame = results[0].plot()

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_media():
    """
    Handle image and video uploads for processing.
    """
    if 'detections' not in session:
        session['detections'] = []

    uploaded_files = []
    for file in request.files.getlist('files'):
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = np.array(Image.open(file.stream).convert('RGB'))
            img_data, detections = infer_image(image)
            uploaded_files.append({'type': 'image', 'image_data': img_data, 'detections': detections})
        elif file.filename.lower().endswith(('.mp4', '.avi', '.mkv')):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp_path = tmp.name
                file.save(tmp_path)
            try:
                video_output_path, frame_detections = process_video(tmp_path)
                uploaded_files.append({
                    'type': 'video',
                    'video_path': url_for('static', filename=os.path.basename(video_output_path)),
                    'detections': frame_detections
                })
            finally:
                os.unlink(tmp_path)
        else:
            flash('Unsupported file format. Please upload an image or video.', 'error')

    session['detections'] = uploaded_files
    return redirect(url_for('display_results'))

@app.route('/results')
def display_results():
    """
    Display results for uploaded images/videos.
    """
    detections = session.get('detections', [])
    return render_template('results.html', detections=detections)

@app.route('/reset', methods=['GET'])
def reset_app():
    """
    Reset the app session.
    """
    session.clear()
    flash('The app has been reset.', 'info')
    return redirect(url_for('index'))

@app.route('/camera')
def camera_detection():
    """
    Render camera detection page.
    """
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    """
    Stream video feed with YOLOv8 annotations.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about_page():
    return render_template('about.html')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/how_to_use')
def how_to_use():
    return render_template('howtouse.html')

if  __name__  == '_main_':
    app.run(debug=True)