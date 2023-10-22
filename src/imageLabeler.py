"""
ImageLabeler Script for Generating CSVs of Dataset Splits.

This script provides methods to scan dataset directories and generate
CSV files containing image paths and their corresponding labels for 
train, test, and val splits.

@author: Dhruvil Patel
"""

import os
import pandas as pd


class ImageLabeler:
    def _init_(self, destination_root):
        """
        Initialize the ImageLabeler with the destination root directory.

        Args:
        - destination_root (str): Root directory where train, val, test folders are saved.
        """
        self.destination_root = destination_root

    def generate_csv_for_split(self, split_name):
        """
        Generate a CSV file for the provided dataset split (train, test, or val).

        Args:
        - split_name (str): The name of the split ('train', 'test', or 'val').
        """
        split_dir = os.path.join(self.destination_root, split_name)
        data = self._fetch_data_from_split(split_dir)
        self._save_data_to_csv(data, split_dir, split_name)

    def _fetch_data_from_split(self, split_dir):
        """
        Fetch image names and labels from the given directory split.

        Args:
        - split_dir (str): Directory of the dataset split.

        Returns:
        - list: A list of tuples containing the index, image name, and label.
        """
        indices = []
        image_names = []
        labels = []
        idx = 1

        for label in os.listdir(split_dir):
            label_dir = os.path.join(split_dir, label)
            if os.path.isdir(label_dir):
                for image_name in os.listdir(label_dir):
                    image_path = os.path.join(label_dir, image_name)
                    if os.path.isfile(image_path):
                        indices.append(idx)
                        image_names.append(image_name)
                        labels.append(label)
                        idx += 1

        return indices, image_names, labels

    def _save_data_to_csv(self, data, split_dir, split_name):
        """
        Save data to a CSV file.

        Args:
        - data (tuple): A tuple of lists containing indices, image names, and labels.
        - split_dir (str): Directory of the dataset split.
        - split_name (str): The name of the split ('train', 'test', or 'val').
        """
        indices, image_names, labels = data
        df = pd.DataFrame({"Index": indices, "ImageName": image_names, "Label": labels})
        csv_path = os.path.join(split_dir, f"{split_name}_labels.csv")
        df.to_csv(csv_path, index=False)
        print(f"CSV generated for {split_name} at {csv_path}")


if _name_ == "_main_":
    labeler = ImageLabeler("FINAL-DATASET-2")
    for split in ["train", "test", "val"]:
        labeler.generate_csv_for_split(split)