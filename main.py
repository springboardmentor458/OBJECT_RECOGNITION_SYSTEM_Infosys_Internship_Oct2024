import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, Resize

# Paths
json_path = "dataset/Object Recognition Dataset/annotations/instances_val2017.json"  # Path to JSON file
image_folder = "dataset/Object Recognition Dataset/images"  # Folder containing images
annotated_folder = "annotated_images"  # Folder to save annotated images
processed_folder = "processed_images"  # Folder to save processed images

# Define augmentations using albumentations
augmentation = Compose([
    HorizontalFlip(p=0.5),  # 50% chance to flip image horizontally
    RandomBrightnessContrast(p=0.2),  # 20% chance to apply random brightness and contrast
    Resize(224, 224)  # Resize image to 224x224
])

# Part 1: Annotate Images
def annotate_images(json_path, image_folder, output_folder, max_images=5):
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Mapping annotations to images based on image_id
    image_annotations = {item['image_id']: [] for item in data['annotations']}
    for annotation in data['annotations']:
        image_annotations[annotation['image_id']].append(annotation)

    os.makedirs(output_folder, exist_ok=True)

    image_count = 0

    for image_info in data['images']:
        if image_count >= max_images:
            break

        image_name = image_info['file_name']
        image_path = os.path.join(image_folder, image_name)

        if not os.path.exists(image_path):
            print(f"Image {image_name} not found. Skipping...")
            continue

        image = cv2.imread(image_path)

        annotations = image_annotations.get(image_info['id'], [])

        for annotation in annotations:
            bbox = annotation['bbox']
            if len(bbox) == 4:
                x, y, w, h = map(int, bbox)
                if isinstance(x, int) and isinstance(y, int) and isinstance(w, int) and isinstance(h, int):
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        output_image_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_image_path, image)
        print(f"Annotated image saved at: {output_image_path}")

        image_count += 1

# Part 2: Process the Images (with Augmentation)
def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error loading image: {image_name}")
            continue

        # Apply augmentations
        augmented = augmentation(image=image)
        augmented_image = augmented['image']

        # Convert to grayscale
        gray_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray_image, 100, 200)

        # Convert back to BGR for consistency
        final_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Save processed image
        output_image_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_image_path, final_image)
        print(f"Processed image saved at: {output_image_path}")

# Part 3: Display an Image
def display_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

def main():
    annotate_images(json_path, image_folder, annotated_folder)
    process_images(annotated_folder, processed_folder)

    # Display an example image
    processed_images = os.listdir(processed_folder)
    if processed_images:
        example_image_path = os.path.join(processed_folder, processed_images[0])
        print(f"Displaying image: {example_image_path}")
        display_image(example_image_path)
    else:
        print("No processed images to display.")

if __name__ == "__main__":
    main()
