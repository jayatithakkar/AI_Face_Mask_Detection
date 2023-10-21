# Dataset Processing Scripts

This repository contains a series of scripts designed for processing, cleaning, and managing image datasets. These scripts are specifically tailored to work with facial image datasets and include functionalities such as moving images based on CSV metadata, performing a stratified split, cleaning datasets, and more.

### Author Information:
- Dhruvil Patel
- Jayati Thakkar
- Dhruv Nareshkumar Panchal

---

## 1. `distributor.py`

This script is designed to read images and their labels from given CSV files, and then copy and organize these images into a new directory structure based on their labels.

**Functionality:**
- Reads image paths and labels from a CSV file.
- Creates directories for each label if not already present.
- Copies images to the new directory structure.
- Generates a new CSV file containing the paths and labels of the moved images.

---

## 2. `imageLabeler.py`

This script is meant to generate CSV files containing the image paths and labels for a given dataset split (train, test, or val).

**Functionality:**
- Iterates through a dataset directory structure.
- Collects image names and labels.
- Generates a CSV file for each dataset split.

---

## 3. `stratifiedSplit.py`

This script organizes images into train, test, and validation sets using a stratified split.

**Functionality:**
- Performs a stratified split on the dataset.
- Moves images into train, test, and validation directories.

---

## 4. `datasetCleaning-1.py`

This script identifies and removes corrupted, blank, and inconsistent images from a dataset directory.

**Functionality:**
- Checks for corrupted images.
- Checks for images with inconsistent channels.
- Identifies and removes blank or nearly blank images.

---

## 5. `datasetCleaning-2.py`

This script uses face detection and feature extraction from a ResNet model to identify and process facial images in a dataset.

**Functionality:**
- Detects faces in images using MTCNN.
- Extracts features using a ResNet model.
- Determines image quality based on feature variance.
- Resizes and saves processed images.

---

## Usage

To run these scripts, execute the corresponding Python file in an environment that satisfies the required dependencies:

```
python <script_name>.py
```

### Dependencies:

- pandas
- os
- shutil
- sklearn
- PIL
- numpy
- facenet-pytorch
- torchvision
- torch

---

## Note

These scripts are tailored for specific use-cases. To apply them to different datasets or directory structures, adjustments may be necessary.

---

