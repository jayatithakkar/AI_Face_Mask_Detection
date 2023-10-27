"""
Image Categorization Utility

This script categorizes images based on labels from a CSV and
moves them to their respective directories.

Author:
    Jayati Thakkar
"""

import os
import shutil
import pandas as pd


def categorize_images(folder_path: str, csv_path: str):
    """
    Categorize images based on labels from a CSV and move them to respective folders.
    
    Args:
        folder_path (str): Directory path containing the images to be categorized.
        csv_path (str): Path to the CSV file containing image labels.

    Returns:
        None
    """
    # Define the subfolders
    engagement_folder = os.path.join(folder_path, "Engagement")
    boredom_folder = os.path.join(folder_path, "Boredom")
    
    # Ensure the subfolders exist or create them
    for folder in [engagement_folder, boredom_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Build a dictionary of image names to labels
    df = pd.read_csv(csv_path)
    image_to_label = dict(zip(df["ImageName"], df["Label"]))

    # Iterate over all images and move them to the appropriate folder
    for image_name in os.listdir(folder_path):
        full_image_path = os.path.join(folder_path, image_name)
        
        # Skip if it's a directory
        if os.path.isdir(full_image_path):
            continue

        label = image_to_label.get(image_name.split(".")[0])
        
        if label == "Engagement":
            shutil.move(full_image_path, engagement_folder)
        elif label == "Boredom":
            shutil.move(full_image_path, boredom_folder)


if __name__ == "__main__":
    categorize_images("./FINAL-DATASET-4", "./NewLabels.csv")