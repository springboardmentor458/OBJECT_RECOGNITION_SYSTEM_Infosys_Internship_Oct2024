from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os
import cv2
from werkzeug.utils import secure_filename
import torch
import numpy as np
from PIL import Image

video_app = Flask(__name__)
CORS(video_app)

# Configure upload and results folders
UPLOAD_FOLDER = './uploads'
RESULTS_FOLDER = './results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

video_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 model
CLASS_NAMES = model.names  # Mapping class IDs to labels


@video_app.route('/')
def home():
    return "Object Detection Flask App is Running!"


### IMAGE IDENTIFICATION
@video_app.route('/api/image', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the uploaded image
        filename = secure_filename(file.filename)
        file_path = os.path.join(video_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform object detection
        image = Image.open(file_path)
        results = model(image)

        # Render results on the image
        results.render()
        annotated_image = Image.fromarray(results.imgs[0])

        # Save the annotated image
        annotated_path = os.path.join(RESULTS_FOLDER, f"annotated_{filename}")
        annotated_image.save(annotated_path)

        # Return the URL of the processed image
        result_image_url = f"http://{request.host}/results/{os.path.basename(annotated_path)}"
        response = {
            "message": "Image processed successfully",
            "result_image": result_image_url
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


### VIDEO IDENTIFICATION (EXISTING)
@video_app.route('/api/video', methods=['POST'])
def classify_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the uploaded video
        filename = secure_filename(file.filename)
        file_path = os.path.join(video_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Open the video file for processing
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return jsonify({"error": "Could not open the video file. Please check the format."}), 400

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 FPS if invalid

        # Prepare the output video
        output_filename = f'annotated_{os.path.splitext(filename)[0]}.mp4'
        output_path = os.path.join(RESULTS_FOLDER, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB for YOLOv5
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform object detection
            results = model(rgb_frame)

            # Render results on the frame
            rendered_frame = results.render()[0]
            rendered_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)

            # Write the processed frame to the output video
            out.write(rendered_frame)

        cap.release()
        out.release()

        # Convert video to H.264 format using FFmpeg
        converted_output_filename = f'converted_{output_filename}'
        converted_output_path = os.path.join(RESULTS_FOLDER, converted_output_filename)
        ffmpeg_command = f'ffmpeg -y -i "{output_path}" -vcodec libx264 -crf 22 "{converted_output_path}"'
        os.system(ffmpeg_command)

        # Return the URL of the processed video
        result_video_url = f"http://{request.host}/results/{converted_output_filename}"
        response = {
            "message": "Video processed successfully",
            "result_video": result_video_url
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


### WEBCAM IDENTIFICATION
@video_app.route('/api/webcam', methods=['GET'])
def classify_webcam():
    def generate():
        cap = cv2.VideoCapture(0)  # Open webcam
        if not cap.isOpened():
            yield b"Webcam not accessible"
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform object detection
            results = model(frame)

            # Render results on the frame
            rendered_frame = results.render()[0]
            _, buffer = cv2.imencode('.jpg', rendered_frame)

            # Yield the frame as a response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Serve processed files
@video_app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory(RESULTS_FOLDER, filename)


if __name__ == "__main__":
    video_app.run(debug=True, host='0.0.0.0', port=5001)  # Run on port 5001
