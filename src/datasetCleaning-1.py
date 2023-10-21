"""
Dataset Cleaning Script.

This script contains methods to identify and remove corrupt, blank, and
inconsistent images from a dataset directory.

@author: Dhruv Nareshkumar Panchal
"""

import os
from PIL import Image
import numpy as np


def is_image_corrupted(image_path):
    """Check if an image is corrupted."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return False
    except:
        return True


def has_consistent_channels(image_path, expected_channels=3):
    """Check if an image has the expected number of channels (default: 3 for RGB)."""
    with Image.open(image_path) as img:
        return img.mode == ("RGB" if expected_channels == 3 else "L")


def is_image_blank(image_path, threshold=10):
    """Check if an image is blank or nearly blank."""
    with Image.open(image_path) as img:
        variance = np.var(np.array(img))
    return variance < threshold


def quality_check_and_delete_if_needed(image_path):
    """Comprehensive quality control for images. Deletes the image if it doesn't pass checks."""
    messages = []
    if is_image_corrupted(image_path):
        os.remove(image_path)
        messages.append(f"{image_path} was corrupted.")
    elif not has_consistent_channels(image_path):
        os.remove(image_path)
        messages.append(f"{image_path} didn't have consistent channels (expected RGB).")
    elif is_image_blank(image_path):
        os.remove(image_path)
        messages.append(f"{image_path} was blank or nearly blank.")
    else:
        messages.append(f"{image_path} is good.")
    return messages


def main(root_directory):
    """Main function to iterate over dataset directory and perform quality checks."""
    messages = []
    for split_name in ["train", "test", "val"]:
        split_path = os.path.join(root_directory, split_name)
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    if image_name.endswith((".png", ".jpg", ".jpeg")):
                        image_path = os.path.join(class_path, image_name)
                        messages.extend(quality_check_and_delete_if_needed(image_path))
    for message in messages:
        print(message)


if __name__ == "__main__":
    dataset_dir = "FINAL-DATASET-2"
    main(dataset_dir)
