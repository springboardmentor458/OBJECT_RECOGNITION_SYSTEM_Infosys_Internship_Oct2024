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



# OTHER ON 100 trainng images nad  50 validation n 50 testing 
# import json
# import os
# import shutil
# from tqdm import tqdm

# # Define paths
# data_folder = r"C:\Users\vrunda\Desktop\final\val2017"  # Directory containing image files
# json_file = r"C:\Users\vrunda\Desktop\final\annotations\instances_val2017.json"  # Path to COCO JSON file
# output_directory = r"C:\Users\vrunda\Desktop\final\ds"  # Directory for YOLO dataset output

# # Create the required directory structure for YOLO
# os.makedirs(f"{output_directory}/train/images", exist_ok=True)
# os.makedirs(f"{output_directory}/train/labels", exist_ok=True)
# os.makedirs(f"{output_directory}/val/images", exist_ok=True)
# os.makedirs(f"{output_directory}/val/labels", exist_ok=True)
# os.makedirs(f"{output_directory}/test/images", exist_ok=True)
# os.makedirs(f"{output_directory}/test/labels", exist_ok=True)

# # Define the YOLO-compatible COCO classes
# yolo_coco_classes = [
#     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
#     "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
#     "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
#     "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
#     "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
#     "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
#     "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
#     "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
#     "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
# ]
# category_mapping = {cat['id']: idx for idx, cat in enumerate(json.load(open(json_file))["categories"]) if cat['name'] in yolo_coco_classes}

# # Load COCO JSON annotations
# with open(json_file, "r") as f:
#     coco_data = json.load(f)

# # Map image metadata
# image_data = {
#     img["id"]: {"file_name": img["file_name"], "width": img["width"], "height": img["height"], "bboxes": []}
#     for img in coco_data["images"]
# }

# # Filter annotations to include YOLO-supported classes
# for ann in coco_data["annotations"]:
#     if ann["category_id"] in category_mapping:
#         class_id = category_mapping[ann["category_id"]]
#         bbox = ann["bbox"]  # x, y, width, height
#         image_data[ann["image_id"]]["bboxes"].append((class_id, bbox))

# # Prepare the dataset for YOLO
# for i, (image_id, data) in tqdm(enumerate(image_data.items()), desc="Preparing YOLO dataset"):
#     width, height = data["width"], data["height"]
#     name = os.path.splitext(data["file_name"])[0]
#     src_image = os.path.join(data_folder, data["file_name"])

#     if i < 100:  # Training set
#         dst_image = f"{output_directory}/train/images/{name}.jpg"
#         dst_label = f"{output_directory}/train/labels/{name}.txt"
#     elif i < 150:  # Validation set
#         dst_image = f"{output_directory}/val/images/{name}.jpg"
#         dst_label = f"{output_directory}/val/labels/{name}.txt"
#     elif i < 200:  # Test set
#         dst_image = f"{output_directory}/test/images/{name}.jpg"
#         dst_label = f"{output_directory}/test/labels/{name}.txt"
#     else:
#         continue

#     if not os.path.exists(src_image):
#         print(f"Image not found: {src_image}. Skipping...")
#         continue

#     shutil.copyfile(src_image, dst_image)

#     with open(dst_label, "w") as label_file:
#         for class_id, bbox in data["bboxes"]:
#             x, y, w, h = bbox
#             x_center = (x + w / 2) / width
#             y_center = (y + h / 2) / height
#             norm_width = w / width
#             norm_height = h / height
#             label_file.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")

# # Create the data.yaml file
# data_yaml_content = f"""train: ../datayolo/train/images
# val: ../datayolo/val/images

# nc: 80
# names: {yolo_coco_classes}
# """
# with open(f"{output_directory}/data.yaml", "w") as yaml_file:
#     yaml_file.write(data_yaml_content)

# print("Dataset preparation completed successfully!")


# to detect 
# python detect.py --weights C:\Users\vrunda\Desktop\final\yolov5\runs\train\yolov5_custom3\weights\best.pt --img 640 --conf 0.25 --source C:\Users\vrunda\Desktop\final\ds\test\images\000000008021.jpg

