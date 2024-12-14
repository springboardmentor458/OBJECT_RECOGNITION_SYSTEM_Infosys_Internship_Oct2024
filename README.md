# Infosys Springboard Internship (Nov 24 - Dec 24)

## Overview

This repository contains the code and models developed during my Infosys Springboard Internship from November 24 to December 24. The internship focused on object detection using various state-of-the-art deep learning models, including **Faster R-CNN**, **Mask R-CNN**, and **YOLOv5**. The goal was to train and evaluate these models on different datasets for object recognition and detection tasks.

## Technologies Used

- **Python 3.10**: Programming language used for implementation.
- **PyTorch 2.5.1**: Deep learning framework used to build and train models.
- **CUDA 12.4**: For GPU acceleration in model training.
- **OpenCV**: Library for image processing and manipulation.
- **Roboflow**: Tool used for dataset creation and augmentation.
- **YOLOv5**: Object detection model architecture.

## Installation

1. Clone the repository to your local machine.

2. Install the necessary dependencies by following the requirements provided in the `requirements.txt` file.

3. Ensure that **PyTorch** with **CUDA 12.4** support is installed on your system. Use the following command to install PyTorch:

    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    ```

4. Download and place the datasets in the appropriate folders for each model (e.g., VOC2012, COCO, etc.).

## How to Run

- **Faster R-CNN**: The code provided allows for training and evaluating a Faster R-CNN model on your chosen dataset.
- **Mask R-CNN**: Code for training and testing the Mask R-CNN model on your dataset.
- **YOLOv5**: Set up the dataset and configuration files to train the YOLOv5 model on your data.

## Datasets

- **VOC 2012**: A widely used dataset for object detection, used for training Faster R-CNN and Mask R-CNN models.
- **COCO**: A large-scale dataset used for training YOLOv5 and other object detection models.
- **Roboflow Dataset**: A custom dataset created for animal detection and other object recognition tasks.

## Results

- The results of training and testing processes are saved in the respective output folders.
- Visualizations, model predictions, and evaluation metrics can be found in the output directories.

## Future Work

- Further fine-tuning of models to improve detection accuracy.
- Exploration of additional datasets to enhance object recognition.
- Implementation of other object detection architectures and evaluation techniques.

## License

This project is licensed under the MIT License.
