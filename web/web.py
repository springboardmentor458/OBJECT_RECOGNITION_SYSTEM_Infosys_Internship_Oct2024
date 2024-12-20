import os 
import cv2
import numpy as np
from flask import Flask, request, render_template_string, redirect, session, url_for, Response, flash
from flask_session import Session
from ultralytics import YOLO
from PIL import Image
import tempfile

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load YOLO model
model = YOLO("yolov8n.pt")  # Adjust the model path if needed

# HTML template for uploading video and streaming results
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <title>YOLOv8 Video Object Detection</title>
  </head>
  <body>
    <h1>Upload Video for Object Detection</h1>
    <form method="post" enctype="multipart/form-data" action="/upload">
      <input type="file" name="video" accept="video/*" required>
      <input type="submit" value="Upload and Process">
    </form>
    {% if video_path %}
    <h2>Processed Video Stream</h2>
    <img src="{{ url_for('video_feed') }}" width="640" height="480">
    <br>
    <a href="/reset">Reset to Home</a>
    {% endif %}
  </body>
</html>
"""

# Function to generate video frames with detection and replay support
def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set FPS to process frames faster
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Replay video by resetting position
            continue
        
        # Resize frame to improve speed
        frame_resized = cv2.resize(frame, (640, 480))
        
        # Perform object detection
        results = model(frame_resized, verbose=False)  # Disable verbose for speed
        annotated_frame = results[0].plot()  # Draw detections on the frame
        
        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Increase frame counter
        current_frame += 1
        if current_frame >= frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
            current_frame = 0
        
    cap.release()

# Route to upload video
@app.route('/', methods=['GET'])
def index():
    video_path = session.get('video_path', None)
    return render_template_string(HTML_TEMPLATE, video_path=video_path)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['video']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    # Save uploaded video temporarily
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    file.save(temp_video.name)
    session['video_path'] = temp_video.name
    
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    video_path = session.get('video_path', None)
    if video_path is None:
        return "No video uploaded", 404
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reset', methods=['GET'])
def reset():
    session.pop('video_path', None)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
