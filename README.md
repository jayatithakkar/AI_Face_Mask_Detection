# A.I.ducation Analytics

### Team ID: AK_ 12

### Repository: 
[Github AI_Facial_Expression_Detection](https://github.com/Dhruvil189/AI_Facial_Expression_Detection)



This repository currently contains a series of scripts designed for processing, cleaning, and managing image datasets. These scripts are specifically tailored to work with facial image datasets and include functionalities such as moving images based on CSV metadata, performing a stratified split, cleaning datasets, and more.

### Author Information:
1. **Dhruvil Patel (40226179) – Data Specialist** 
2. **Dhruv Nareshkumar Panchal (40234693) – Training Specialist**
3. **Jayati Thakkar (40230506) – Evaluation Specialist**

### Dataset Information:
1. Dataset-1: [Kaggle - Affectnet Training Data](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data)
2. Dataset-2: [Kaggle - Affectnet Sid](https://www.kaggle.com/datasets/sidd3k/affectnet-sid)
3. Dataset-3: [IITH - DAiSEE](https://people.iith.ac.in/vineethnb/resources/daisee/index.html)
4. Processed Dataset Link: [Google Drive](https://drive.google.com/file/d/1y-snmEcvpFf4qYkxElitwFuLluL5xo1H/view?usp=sharing)

---

## Source Code:

### 1. `distributor.py`

This script is designed to read images and their labels from given CSV files, and then copy and organize these images into a new directory structure based on their labels.

**Functionality:**
- Reads image paths and labels from a CSV file.
- Creates directories for each label if not already present.
- Copies images to the new directory structure.
- Generates a new CSV file containing the paths and labels of the moved images.

---

### 2. `imageLabeler.py`

This script is meant to generate CSV files containing the image paths and labels for a given dataset split (train, test, or val).

**Functionality:**
- Iterates through a dataset directory structure.
- Collects image names and labels.
- Generates a CSV file for each dataset split.

---

### 3. `stratifiedSplit.py`

This script organizes images into train, test, and validation sets using a stratified split.

**Functionality:**
- Performs a stratified split on the dataset.
- Moves images into train, test, and validation directories.

---

### 4. `datasetCleaning-1.py`

This script identifies and removes corrupted, blank, and inconsistent images from a dataset directory.

**Functionality:**
- Checks for corrupted images.
- Checks for images with inconsistent channels.
- Identifies and removes blank or nearly blank images.

---

### 5. `datasetCleaning-2.py`

This script uses face detection and feature extraction from a ResNet model to identify and process facial images in a dataset.

**Functionality:**
- Detects faces in images using MTCNN.
- Extracts features using a ResNet model.
- Determines image quality based on feature variance.
- Resizes and saves processed images.

---

### 6. `frameExtractor.py`

This script is designed to extract frames from AVI videos at specified time marks.

**Functionality:**
- Extracts frames from .avi videos at the 2-second, 5-second, and 7-second marks.
- Saves extracted frames in the same directory as the original video with specific suffixes.
- Overwrites existing images with the same name.
- Requires `[ffmpeg](https://www.ffmpeg.org/)` for successful execution.

---

### 7. `categorizeImages.py`

This script is developed to categorize images based on labels from a CSV and move them to their respective directories.

**Functionality:**
- Reads image names and labels from a specified CSV file.
- Organizes images into directories based on their labels ("Engagement" or "Boredom").
- Creates label-specific directories if they don't exist.

---

### 8. `utilityFunction1.py`

This script provides utilities for renaming images and generating a new CSV based on renamed images and an original CSV.

**Functionality:**
- Renames images based on specific conditions for Dataset-3.
- Creates a new CSV file with the renamed images and uses data from an original CSV.

---

### 9. `dataVisualization.py`

This script is built to visualize different aspects of an image dataset, including class distribution, sample images, and pixel intensity distributions.

**Functionality:**
- Loads image data and corresponding class labels from a directory structure.
- Visualizes the distribution of classes in the dataset.
- Displays a grid of random sample images from the dataset.
- Plots pixel intensity distribution for a sample of images, either RGB or grayscale.

---

## Setup and Dependencies:

### Prerequisites:

Ensure you have Python version `3.10.12` installed. You can check your Python version using the following command:

```bash
python --version
```

If you do not have the required version, please download and install it from the [official Python website](https://www.python.org/downloads/).

### Installing Dependencies:

The required dependencies for this project are listed in the `requirements.txt` file. To install these dependencies, navigate to the project's root directory in your terminal or command prompt and run the following command:

```bash
pip install -r requirements.txt
```

### Dependencies List:

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

## Usage

To execute any of the scripts, make sure to run the appropriate Python file in an environment that satisfies the required dependencies:

```bash
python <script_name>.py
```

### Dataset Preparation:

1. **Download the Datasets:** 
    - **Dataset-1:** Download from [Kaggle - Affectnet Training Data](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data).
    - **Dataset-2:** Download from [Kaggle - Affectnet Sid](https://www.kaggle.com/datasets/sidd3k/affectnet-sid). Note: The Dataset-2 is divided into **Dataset-2A (train)** and **Dataset-2B (val)** by the author. We use these names for clarity.

2. **Dataset Distribution:**
    - Execute `distributor.py`. Ensure the correct paths for all datasets are set in the script.

3. **Stratified Splitting:**
    - Run `stratifiedSplit.py`. Adjust the portions for train, test, and val datasets as needed. This script divides the entire dataset into three segments.

4. **Image Labeling:**
    - Execute `imageLabeler.py`. It will create custom labels for train, test, and val datasets. Ensure the path to the dataset is correct.

5. **Dataset Cleaning:**
    - Run `datasetCleaning-1.py` followed by `datasetCleaning-2.py`. Make sure to verify the paths before execution.

6. **Dataset-3 Preparation:**
    - **Download:** Get Dataset-3 from [IITH - DAiSEE](https://people.iith.ac.in/vineethnb/resources/daisee/index.html).
    - **Frame Extraction:** Use `frameExtractor.py` to extract frames from all video files in the dataset. Remember to set the correct path.
    - **Categorization:** Execute `categorizeImages.py` to sort images into two new folders. Ensure paths are set appropriately.
    - **Stratified Splitting:** Run `stratifiedSplit.py` to segment the data into train, test, and val portions.

7. **Dataset Cleaning:**
    - Run `datasetCleaning-1.py` followed by `datasetCleaning-2.py` for Dataset-3. Make sure to verify the paths before execution.
    
8. **Utility Functions:**
    - Execute `utilityFunction1.py` to rename files as needed and then generate custom labels for Dataset-3.

9. **Merging Datasets:**
    - Combine the folders of Dataset-1, Dataset-2, and our new Dataset-3 folder. All datasets have the same hierarchy. Optionally, execute `imageLabeler.py` once more to regenerate labels for the combined dataset.

10. **Data Visualization:**
    - Lastly, run `dataVisualization.py` for insights into the dataset.

**After following the above steps meticulously, you will successfully curate a dataset that aligns precisely with the structure and format provided in our reference link.**

---

⚠️ **Note:** It's essential to provide the proper paths in each script to ensure accurate execution. These scripts are tailored for specific use-cases. To apply them to different datasets or directory structures, adjustments may be necessary.

---

