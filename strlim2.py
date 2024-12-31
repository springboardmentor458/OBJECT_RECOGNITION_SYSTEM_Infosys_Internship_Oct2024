import streamlit as st
import requests
import os

# Define the Flask API endpoint
FLASK_API_URL = "http://127.0.0.1:5001/api/video"  # Replace with your Flask server's URL

# Streamlit app title
st.title("YOLO Video Detection")

# Instructions
st.write("""
Upload a video file, and the YOLO object detection model will process it.
Once completed, the annotated video will be displayed below.
""")

# File uploader for videos
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

# Placeholder for displaying processed video
video_placeholder = st.empty()

# Process the video
if uploaded_file:
    # Save the uploaded file temporarily
    temp_file_path = os.path.join("temp_uploads", uploaded_file.name)
    os.makedirs("temp_uploads", exist_ok=True)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Display upload status
    st.write("Video uploaded successfully!")

    # Call the Flask API to process the video
    st.write("Processing video, please wait...")
    with open(temp_file_path, "rb") as f:
        files = {'video': f}
        response = requests.post(FLASK_API_URL, files=files)

    # Handle the response
    if response.status_code == 200:
        result = response.json()
        video_url = result.get("result_video")
        
        if video_url:
            st.success("Video processed successfully!")
            st.video(video_url)
        else:
            st.error("Error: Processed video URL not found.")
    else:
        st.error(f"Error: {response.status_code} - {response.text}")

    # Clean up temporary file
    os.remove(temp_file_path)
