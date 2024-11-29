import os
from subprocess import run

# Paths
yolov5_dir = r"C:\Users\vrunda\Desktop\new\yolov5"  # Path to YOLOv5 repo
dataset_dir = r"C:\Users\vrunda\Desktop\new\dataset"  # Path to your dataset
data_yaml = os.path.join(dataset_dir, "data.yaml")  # Dataset configuration file

# Train YOLOv5
run([
    "python", os.path.join(yolov5_dir, "train.py"),
    "--img", "640",                # Image size
    "--batch", "16",               # Batch size
    "--epochs", "10",              # Number of epochs
    "--data", data_yaml,           # Dataset YAML file
    "--weights", "yolov5s.pt",     # Pretrained weights (YOLOv5 small model)
    "--device", "0"                # Use GPU (0) or CPU (-1)
])

# python detect.py --weights runs/train/exp/weights/best.pt --source C:\Users\vrunda\Desktop\new\e.jpg --img 640 --device 0
# python detect.py --weights runs/train/exp/weights/best.pt --source C:\Users\vrunda\Desktop\new\e.jpg --img 640 --device 0 --conf-thres 0.3
python detect.py --weights runs/train/exp/weights/best.pt --source C:\Users\vrunda\Desktop\new\zebra.jpeg --img 640 --device 0 --conf-thres 0.3
