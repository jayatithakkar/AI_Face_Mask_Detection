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

### 10. `train.py`

This script defines a convolutional neural network (CNN) for image classification, including the CustomCNN class, dataset loading functions, model training and validation, and performance evaluation on a test dataset.

**Functionality:**
- Defines the CustomCNN class for CNN models.
- Loads image datasets for training, validation, and testing.
- Trains and validates the CNN model.

---

### 11. `evaluate.py`

This script includes functions for evaluating a trained CNN model on a test dataset, providing detailed performance metrics.

**Functionality:**
- Loads a trained CNN model and test dataset.
- Evaluates the model on the test dataset.
- Calculates and displays performance metrics like confusion matrix, precision, recall, and F1 score.
- Plots confusion matrix and classification reports.

---

### 12. `test.py`

This script is designed for making predictions on individual images using a trained CNN model.

**Functionality:**
- Loads a trained CNN model.
- Predicts the class of individual images.
- Displays the images with their predicted classes.


### 13. `ageBiasTest.py`

This script implements a bias test for age in image classification using a custom convolutional neural network (CNN). It imports libraries like PyTorch for neural network implementation and sklearn for accuracy and performance metrics. The script includes a CustomCNN class with convolutional layers and functions for data loading and model evaluation.

**Functionality:**
- Implements a custom CNN with PyTorch for age bias testing.
- Provides functions for data loading and preprocessing.
- Evaluates model performance with accuracy, precision, recall, and F1 score.

---

### 14. `ageGenderPredictionPipeline.py`

This script sets up a pipeline for predicting age and gender from images. It uses OpenCV for deep neural network operations, pandas for data manipulation, and predefined constants for model and dataset management. Functions include model loading, CUDA optimization, and prediction execution.

**Functionality:**
- Utilizes OpenCV for neural network operations in age and gender prediction.
- Manages dataset and model parameters with predefined constants.
- Executes age and gender predictions on datasets.

---

### 15. `genderBiasTest.py`

Similar to `ageBiasTest.py`, this script focuses on testing gender bias in image classification models. It includes a CustomCNN class designed for gender bias evaluation, using PyTorch for neural network implementation and sklearn for performance metrics.

**Functionality:**
- Features a custom CNN model for gender bias testing in images.
- Includes data loading and preprocessing functions.
- Assesses model performance using standard metrics like accuracy and F1 score.

---

### 16. `kFoldCrossValidation.py`

This script is dedicated to performing k-fold cross-validation on image classification models. It includes a CustomCNN class, utilizes PyTorch for model definition, and sklearn for k-fold implementation and performance evaluation.

**Functionality:**
- Implements k-fold cross-validation using a custom CNN model.
- Utilizes PyTorch for neural network operations and model training.
- Employs sklearn for dataset splitting and performance evaluation.

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

## Training, Testing, and Evaluation:

After preparing the dataset, follow these steps to train, test, and evaluate the models using the updated scripts in the repository.

### Training the Models:

1. **Execute `mainmodel-train.py`, `variant1-train.py`, and `variant2-train.py`:**
   - These scripts train the respective CNN models using the processed dataset.
   - Run each script by executing:
     ```bash
     python mainmodel-train.py
     ```
     ```bash
     python variant1-train.py
     ```
     ```bash
     python variant2-train.py
     ```
   - The trained models will be saved as `mainmodel-latest.pth`, `variant1-latest.pth`, and `variant2-latest.pth` respectively.

### Testing the Models:

2. **Execute `mainmodel-test.py`, `variant1-test.py`, and `variant2-test.py`:**
   - These scripts are used to make predictions on individual images for each model.
   - Before running, place the image you want to test in the root directory and name it `image.jpg` (or modify the script to point to the correct image path).
   - Run each script by executing:
     ```bash
     python mainmodel-test.py
     ```
     ```bash
     python variant1-test.py
     ```
     ```bash
     python variant2-test.py
     ```
   - Each script will display the image along with its predicted class for the respective model.

### Evaluating the Models:

3. **Execute `mainmodel-evaluate.py`, `variant1-evaluate.py`, and `variant2-evaluate.py`:**
   - These scripts evaluate each trained model's performance on the test dataset.
   - They will provide detailed metrics such as accuracy, confusion matrix, precision, recall, and F1 score for each model.
   - Run each script by executing:
     ```bash
     python mainmodel-evaluate.py
     ```
     ```bash
     python variant1-evaluate.py
     ```
     ```bash
     python variant2-evaluate.py
     ```
   - The results, including a confusion matrix and classification report for each model, will be displayed.

---

## Bias Analysis and Model Validation:

To perform a comprehensive bias analysis and validate your model, follow these steps using the newly added scripts in the repository.

### Labeling the Dataset:

1. **Execute `ageGenderPredictionPipeline.py`:**
   - This script labels the dataset with age and gender predictions, essential for subsequent bias analysis.
   - Run the script by executing:
     ```bash
     python ageGenderPredictionPipeline.py
     ```
   - Ensure your dataset is in the specified format and directory as required by the script.

### Model Validation:

2. **Execute `kFoldCrossValidation.py`:**
   - This script performs k-fold cross-validation to validate your model's performance and robustness.
   - Run the script by executing:
     ```bash
     python kFoldCrossValidation.py
     ```
   - The script will output the performance of each fold, helping you identify the best model configuration.

### Bias Testing:

3. **Execute `ageBiasTest.py` and `genderBiasTest.py`:**
   - After model validation, use these scripts to test for any age and gender biases.
   - First, execute `ageBiasTest.py` by:
     ```bash
     python ageBiasTest.py
     ```
   - Then, execute `genderBiasTest.py` by:
     ```bash
     python genderBiasTest.py
     ```
   - Both scripts will evaluate the biases in your model, providing insights into any disparities in age and gender predictions.

---

⚠️ **Note:** It's essential to provide the proper paths in each script to ensure accurate execution. These scripts are tailored for specific use-cases. To apply them to different datasets or directory structures, adjustments may be necessary.

---

