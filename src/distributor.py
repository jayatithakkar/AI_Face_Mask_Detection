"""
Distributor Script to Organize Datasets.

This script provides methods to read and process different datasets,
re-organizing them under a unified directory structure and generating 
corresponding label files.

@author: Dhruvil Patel
"""

import pandas as pd
import os
import shutil


def process_dataset(df, image_column, label_column, base_path, label_mapping=None):
    """
    Process the provided dataset dataframe and distribute images.

    Args:
    - df (DataFrame): Input dataframe with image paths and labels.
    - image_column (int): Column index or name for the image path in the dataframe.
    - label_column (int): Column index or name for the label in the dataframe.
    - base_path (str): Base directory path for the dataset.
    - label_mapping (dict, optional): Mapping of numeric labels to string labels.

    Returns:
    None
    """
    global counter

    for _, row in df.iterrows():
        old_path = os.path.join(os.getcwd(), base_path, row[image_column])

        label = row[label_column]
        if label_mapping:
            label = label_mapping.get(label, "unknown")

        new_dir = os.path.join(new_root_path, label)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        new_filename = f"image{counter:05d}.jpg"
        new_path = os.path.join(new_dir, new_filename)
        shutil.copy(old_path, new_path)

        new_data.append([counter, new_filename, label])
        counter += 1


# Define the root path for the new images
new_root_path = os.path.join(os.getcwd(), "FINAL-DATASET-1")

# List to store new paths and labels
new_data = []
counter = 1

if not os.path.exists(new_root_path):
    os.makedirs(new_root_path)

# Dataset-1
df1 = pd.read_csv("./affectnet-training-data/labels.csv")
process_dataset(df1, 1, 2, "affectnet-training-data")

# Dataset-2A and 2B label mapping
label_mapping = {
    1: "neutral",
    2: "happy",
    3: "sad",
    4: "surprise",
    5: "fear",
    6: "disgust",
    7: "anger",
    8: "contempt",
}

# Dataset-2A
df2a = pd.read_csv("./affectnet-sid/AffectNet/train.csv")
process_dataset(
    df2a,
    3,
    2,
    os.path.join("affectnet-sid", "AffectNet", "train_images"),
    label_mapping,
)

# Dataset-2B
df2b = pd.read_csv("./affectnet-sid/AffectNet/valid.csv")
process_dataset(
    df2b, 3, 2, os.path.join("affectnet-sid", "AffectNet", "val_images"), label_mapping
)

# Saving the consolidated labels
new_df = pd.DataFrame(new_data, columns=["Index", "ImageName", "Label"])
new_df.to_csv(os.path.join(new_root_path, "new_labels.csv"), index=False)

print("Done!")
