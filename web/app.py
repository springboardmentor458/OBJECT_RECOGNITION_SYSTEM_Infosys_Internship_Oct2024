import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, session, url_for, Response, flash, send_file
from flask_session import Session
from ultralytics import YOLO
from PIL import Image
import tempfile
import base64

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load YOLO model
model = YOLO("yolov8n.pt")  # Ensure the YOLO model path is correct

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')  # Home page for uploading files and navigation

@app.route('/upload', methods=['POST'])
def upload_media():
    if 'files' not in request.files:
        flash('No files uploaded.', 'error')
        return redirect(url_for('index'))

    uploaded_files = []
    for file in request.files.getlist('files'):
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = np.array(Image.open(file.stream).convert('RGB'))
            img_data, detections = infer_image(image)
            uploaded_files.append({'type': 'image', 'data': img_data, 'detections': detections})
        elif file.filename.lower().endswith(('.mp4', '.avi', '.mkv')):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp_path = tmp.name
                file.save(tmp_path)
            frame_detections, processed_video_path = process_video(tmp_path)
            uploaded_files.append({'type': 'video', 'processed_video': processed_video_path, 'detections': frame_detections})
        else:
            flash(f'Unsupported file format: {file.filename}', 'error')
            continue

    session['detections'] = uploaded_files
    return redirect(url_for('results_view'))

@app.route('/results')
def results_view():
    detections = session.get('detections', [])
    return render_template('results.html', detections=detections)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download_video/<int:video_index>')
def download_video(video_index):
    detections = session.get('detections', [])
    if 0 <= video_index < len(detections) and 'processed_video' in detections[video_index]:
        processed_video_path = detections[video_index]['processed_video']
        return send_file(processed_video_path, mimetype='video/mp4', as_attachment=True)
    flash('Invalid video index or video not processed.', 'error')
    return redirect(url_for('results_view'))

@app.route('/reset', methods=['GET'])
def reset_app():
    session.clear()
    flash('Application has been reset.', 'info')
    return redirect(url_for('index'))

@app.route('/camera')
def camera_detection():
    return render_template('camera.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how_to_use')
def how_to_use():
    return render_template('howtouse.html')

# YOLO Processing Functions
def infer_image(image):
    results = model.predict(source=image, save=False, conf=0.25, verbose=False)
    annotated_image = results[0].plot()
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        detections.append({
            'class': results[0].names[class_id],
            'confidence': round(confidence, 2),
            'bounding_box': [x1, y1, x2, y2]
        })
    _, buffer = cv2.imencode('.png', annotated_image)
    img_data = base64.b64encode(buffer).decode('utf-8')
    return img_data, detections

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    temp_processed_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_processed_video.name, fourcc, fps, (width, height))
    frame_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, save=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        frame_detection = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            frame_detection.append({
                'class': results[0].names[class_id],
                'confidence': round(confidence, 2),
                'bounding_box': [x1, y1, x2, y2]
            })
        frame_detections.append(frame_detection)

    cap.release()
    out.release()
    return frame_detections, temp_processed_video.name

def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(source=frame, save=False, conf=0.5)  # Adjust confidence threshold if needed

        # Annotate the frame with detections
        annotated_frame = results[0].plot()

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue

        frame_data = buffer.tobytes()

        # Yield frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()



@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
