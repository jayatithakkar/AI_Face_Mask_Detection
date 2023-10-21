"""
StratifiedSplit Script for Dividing Dataset into Train, Validation, and Test Splits.

This script provides methods to perform a stratified split on a given dataset directory
and move the images to train, val, and test directories based on the split results.

@author: Jayati Thakkar
"""

import os
import shutil
from sklearn.model_selection import train_test_split


class StratifiedSplitter:
    def __init__(self, source_root, destination_root):
        """
        Initialize the StratifiedSplitter with source and destination directories.

        Args:
        - source_root (str): Root directory of the source dataset.
        - destination_root (str): Root directory where train, val, and test folders will be saved.
        """
        self.source_root = source_root
        self.destination_root = destination_root

        # Make directories for train, test, and val sets
        for split in ["train", "val", "test"]:
            os.makedirs(os.path.join(self.destination_root, split), exist_ok=True)

    def split_and_move(self):
        """
        Perform stratified split and move files to the corresponding directories.
        """
        file_paths, labels = self._collect_files_and_labels()
        train_paths, val_paths, test_paths, train_labels, val_labels, test_labels = self._perform_stratified_split(file_paths, labels)

        # Move files to corresponding directories
        self._move_files(train_paths, train_labels, "train")
        self._move_files(val_paths, val_labels, "val")
        self._move_files(test_paths, test_labels, "test")

        print("Dataset split and moved successfully!")

    def _collect_files_and_labels(self):
        """
        Collect file paths and their corresponding labels from the source directory.

        Returns:
        - tuple: A tuple of lists containing file paths and their corresponding labels.
        """
        file_paths = []
        labels = []

        for label_str in os.listdir(self.source_root):
            class_folder = os.path.join(self.source_root, label_str)
            if os.path.isdir(class_folder):
                for filename in os.listdir(class_folder):
                    file_paths.append(os.path.join(class_folder, filename))
                    labels.append(label_str)

        return file_paths, labels

    def _perform_stratified_split(self, file_paths, labels):
        """
        Perform a stratified split of the dataset.

        Args:
        - file_paths (list): List of file paths.
        - labels (list): Corresponding labels of the file paths.

        Returns:
        - tuple: Lists containing train, validation, and test file paths and labels.
        """
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            file_paths, labels, stratify=labels, test_size=0.20, random_state=42
        )

        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, stratify=temp_labels, test_size=0.50, random_state=42
        )

        return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels

    def _move_files(self, file_paths, labels, split_name):
        """
        Move files to the specified split directory.

        Args:
        - file_paths (list): List of file paths.
        - labels (list): Corresponding labels of the file paths.
        - split_name (str): Name of the split directory (train, val, or test).
        """
        destination = os.path.join(self.destination_root, split_name)
        for file_path, label in zip(file_paths, labels):
            dest_folder = os.path.join(destination, label)
            os.makedirs(dest_folder, exist_ok=True)
            shutil.move(file_path, os.path.join(dest_folder, os.path.basename(file_path)))


if __name__ == "__main__":
    source_dir = "./FINAL-DATASET-1"
    dest_dir = "./FINAL-DATASET-2"
    splitter = StratifiedSplitter(source_dir, dest_dir)
    splitter.split_and_move()
