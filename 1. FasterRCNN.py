
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms

# Pascal VOC class names (21 classes: 20 classes + 1 background)
class_names = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
    'sofa', 'train', 'tvmonitor'
]

# Define the transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the trained model
model = fasterrcnn_resnet50_fpn(pretrained=False)  # Set pretrained=False since you're loading your custom-trained model
num_classes = 21  # 20 classes + background

# Modify the model's classifier (as done during training)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load the model weights (adjust the path to where you saved your model)
model.load_state_dict(torch.load('fasterrcnn_voc2012_1000_images.pth'))
model.eval()  # Set model to evaluation mode

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Load a sample image (use the path to your image)
image_path = "VOCdevkit/VOC2012/JPEGImages/2007_000241.jpg"  # Update with your image path
image = Image.open(image_path).convert("RGB")

# Apply the same transformations as during training
image_tensor = transform(image).unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    prediction = model(image_tensor)

# Get prediction details
boxes = prediction[0]['boxes']
labels = prediction[0]['labels']
scores = prediction[0]['scores']

# Filter predictions based on a score threshold (e.g., 0.5)
threshold = 0.5
filtered_boxes = boxes[scores > threshold]
filtered_labels = labels[scores > threshold]

# Visualize the image and bounding boxes
fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(image)

# Draw bounding boxes on the image
for box, label in zip(filtered_boxes, filtered_labels):
    xmin, ymin, xmax, ymax = box.tolist()
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Map label to class name and display it
    class_name = class_names[label.item()]
    ax.text(xmin, ymin, class_name, fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))

plt.show()


