import os
import json
import cv2
import requests
import numpy as np
from PIL import Image



# Part 1: Download and Annotate Images
def download_and_annotate_images(json_path, output_folder):
    with open(json_path, 'r') as file:
        data = json.load(file)

    os.makedirs(output_folder, exist_ok=True)

    for item in data['images']:
        image_url = item['url']
        image_name = os.path.basename(image_url)
        image_response = requests.get(image_url)

        image = np.asarray(bytearray(image_response.content), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        for annotation in item['annotations']:
            x, y, w, h = annotation['bbox']
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        output_image_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_image_path, image)


# Part 2: Process the Images
def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        resized_image = cv2.resize(thresholded_image, (224, 224))

        final_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
        output_image_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_image_path, final_image)
