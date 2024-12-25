from flask import Flask, request, jsonify, send_file
import os
import cv2
import torch
from pathlib import Path
import sys

# Add YOLOv5 directory to sys.path
sys.path.append(r'C:\Users\harsh\Downloads\flask\yolov5')  # Adjust path as needed

# Import YOLOv5 utilities
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

app = Flask(__name__)

# Load YOLOv5 Model
weights_path = 'yolov5s.pt'  # Pretrained weights
device = select_device('')  # Auto-select CPU/GPU
model = DetectMultiBackend(weights_path, device=device, dnn=False)
model.eval()

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found in request'}), 400

    image_file = request.files['image']

    # Validate file type
    if not image_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({'error': 'Invalid file type. Only jpg, jpeg, png are allowed.'}), 400

    img_path = os.path.join('uploads', image_file.filename)
    os.makedirs('uploads', exist_ok=True)
    image_file.save(img_path)

    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (640, 640))[:, :, ::-1].copy()  # Convert BGR to RGB and ensure array is contiguous
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    preds = model(img_tensor)
    preds = non_max_suppression(preds)[0]

    if preds is None or len(preds) == 0:
        return jsonify({'message': 'No objects detected'}), 200

    for *xyxy, conf, cls in preds:
        label = f"{model.names[int(cls)]} {conf:.2f}"
        xyxy = [int(coord) for coord in xyxy]
        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(img, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_path = os.path.join('outputs', image_file.filename)
    os.makedirs('outputs', exist_ok=True)
    cv2.imwrite(output_path, img)

    return send_file(output_path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
