# OBJECT_RECOGNITION_SYSTEM_Infosys_Internship_Oct2024
The object recognition project aims to develop a system capable of identifying objects in real-time using machine learning algorithms. This system will enable the identification of various objects captured by a laptop's camera.

# Object Detection with YOLO and Flask

A web-based object detection application using YOLOv8 and Flask. This project allows users to upload images or use a live video feed for real-time object detection.

## Features
- Detect objects in uploaded images.
- Real-time object detection using a webcam.
- Highlight objects with customized bounding box colors for different object types.
- Lightweight and easy to deploy.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ashutosh-76/Object-Detection-Yolo-Flask.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Object-Detection-Yolo-Flask
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open the application in your browser:
   ```
   http://127.0.0.1:5000/
   ```

3. **Object Detection Options**:
   - **Image Upload**: Use the home page to upload an image for object detection.
   - **Live Video Feed**: Navigate to the `/video` endpoint to enable real-time detection using your laptop's webcam.

## Project Structure

```
Object-Detection-Yolo-Flask/
├── templates/
│   ├── index.html       # Page for image upload and detection
│   ├── video.html       # Page for live video feed detection
├── static/
│   └── uploads/         # Folder to temporarily store uploaded images
├── app.py               # Main Flask application
├── requirements.txt     # Python dependencies
```

## Requirements

- Python 3.8+
- Flask
- OpenCV
- YOLOv8 model weights
- Ultralytics package

## How It Works

1. The `app.py` initializes a YOLOv8 model for object detection.
2. Users can upload images or access the video stream through the web interface.
3. Detected objects are highlighted with bounding boxes and labeled with their names and confidence scores.

## Contributing

Contributions are welcome! If you'd like to add features or improve the project, feel free to create a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

- **YOLOv8**: For providing a robust object detection framework.
- **Flask**: For enabling easy web application development.

---
*Developed as part of the Infosys Internship Program (October 2024).*

