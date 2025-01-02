# Infosys Springboard Internship (November 24 - December 24)


# Project Overview
This repository documents the work completed during my Infosys Springboard Internship from November 24 to December 24. The focus of this internship was on advancing object detection capabilities using cutting-edge deep learning models such as Faster R-CNN, Mask R-CNN, and YOLOv5. The primary objective was to train, evaluate, and compare these models across diverse datasets to achieve accurate object recognition and detection results.

# Technologies and Tools
Python 3.10: Programming language utilized for all implementations.
PyTorch 2.5.1: Framework for developing and training deep learning models.
CUDA 12.4: Leveraged for GPU-accelerated model training.
OpenCV: Used for image processing and preprocessing tasks.
Roboflow: Tool for dataset generation and augmentation.
YOLOv5: State-of-the-art object detection architecture.


# Setup and Installation
Clone the repository to your local machine.
Install required dependencies by referring to the requirements.txt file.
Ensure that PyTorch with CUDA 12.4 support is correctly installed. Use the following command to set up PyTorch:
bash

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  

Download the necessary datasets (e.g., VOC2012, COCO) and place them in the respective folders for model training and evaluation.


# Model Execution
Faster R-CNN: Provides training and evaluation scripts for object detection tasks using this architecture.
Mask R-CNN: Includes tools for training and testing the Mask R-CNN model on your dataset.
YOLOv5: Supports dataset setup, configuration, and training workflows tailored for YOLOv5.


# Datasets Utilized
VOC 2012: A standard benchmark dataset for object detection, used for Faster R-CNN and Mask R-CNN training.
COCO: A large-scale dataset ideal for training YOLOv5 and other deep learning models.
Custom Roboflow Dataset: Specialized dataset curated for tasks like animal detection and other domain-specific object recognition challenges.


# Project Outputs
All training logs, model predictions, and evaluation metrics are stored in the designated output directories.
Visual representations of model performance, including detection results and evaluation graphs, are available for review.


# Future Directions
Further optimization of models to enhance detection accuracy.
Expansion to additional datasets to explore diverse object recognition scenarios.
Integration and testing of new object detection algorithms and methodologies.


# License
This project is distributed under the MIT License, allowing for reuse and modification in alignment with its terms.
