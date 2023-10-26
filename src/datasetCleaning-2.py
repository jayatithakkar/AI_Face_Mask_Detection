"""
Dataset Cleaning with Face Detection and Feature Extraction Script.

This script contains methods to process images using face detection and ResNet 
feature extraction to identify and remove undesirable images from a dataset directory.

@author: Dhruv Nareshkumar Panchal
"""

# from google.colab import drive
import os
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np

# Drive mounting and dataset extraction (specific to Google Colab)
# drive.mount('/content/drive')
# !unzip /content/drive/MyDrive/FINAL-DATASET-2.zip -d /content/

# Installing necessary packages
# !pip install facenet-pytorch torchvision torch

# MTCNN for face detection
mtcnn = MTCNN(select_largest=False, post_process=False)

# Threshold for feature variance
threshold = 0.15

# Load pre-trained ResNet and set to evaluation mode
model = resnet50(pretrained=True).eval().cuda()

# Feature extractor
feature_extractor = nn.Sequential(*list(model.children())[:-1]).cuda()

# Image preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def process_image(image_path):
    """Process an image and remove it if it doesn't meet criteria."""
    # Load image
    img = Image.open(image_path)

    # Detect face and crop
    face = mtcnn(img)
    if face is None:
        print(f"No face detected in {image_path}")
        os.remove(image_path)
        return

    # Extract features using ResNet
    face_img = transforms.ToPILImage()(face)
    tensor = preprocess(face_img).unsqueeze(0).cuda()
    with torch.no_grad():
        features = feature_extractor(tensor)

    # Use threshold to decide image's quality
    variance = np.var(features.cpu().numpy())
    if variance < threshold:
        print(f"Low feature variance in {image_path}")
        os.remove(image_path)
        return

    # Resize image to 224x224
    img.resize((128, 128)).save(image_path)


def main(root_directory):
    """Main function to iterate over dataset directory and perform image processing."""
    for split_name in ["train", "test", "val"]:
        split_path = os.path.join(root_directory, split_name)
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    if image_name.endswith((".png", ".jpg", ".jpeg")):
                        process_image(os.path.join(class_path, image_name))


if __name__ == "__main__":
    dataset_dir = "./FINAL-DATASET-2"
    main(dataset_dir)
