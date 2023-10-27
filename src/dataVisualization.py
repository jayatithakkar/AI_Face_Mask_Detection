import os
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
from PIL import Image


class DataLoader:
    """
    Class responsible for loading image data.

    Attributes:
        root_dir (str): Root directory of the dataset.
        classes (list): List of classes in the dataset.
        dataset (list): List of tuples containing image paths and their corresponding classes.

    Author:
        Dhruv Nareshkumar Panchal
    """

    def __init__(self, root_dir, classes):
        """
        Initializes the DataLoader with the given root directory and classes.

        Args:
            root_dir (str): Root directory of the dataset.
            classes (list): List of classes in the dataset.
        """
        self.root_dir = root_dir
        self.classes = classes
        self.dataset = self._load_data()

    def _load_data(self):
        """Private method to load data from the specified directory."""
        data = []
        for cls in self.classes:
            class_dir = os.path.join(self.root_dir, cls)
            data.extend(
                [
                    (os.path.join(class_dir, img_name), cls)
                    for img_name in os.listdir(class_dir)
                ]
            )
        return data

    def get_class_distribution(self):
        """Returns the distribution of classes in the dataset."""
        return Counter([label for (_, label) in self.dataset])

    def get_sample_images(self, num_samples):
        """Returns a random sample of images from the dataset."""
        return random.sample(self.dataset, num_samples)


class DataVisualizer:
    """
    Class responsible for visualizing the image data.

    Author:
        Dhruv Nareshkumar Panchal
    """

    @staticmethod
    def plot_class_distribution(class_counts):
        """Plots the distribution of classes."""
        plt.bar(class_counts.keys(), class_counts.values())
        plt.xlabel("Class")
        plt.ylabel("Number of images")
        plt.title("Class Distribution")
        plt.show()

    @staticmethod
    def plot_sample_images(sample_images):
        """Plots a grid of sample images."""
        grid_size = int(np.sqrt(len(sample_images)))

        fig, axes = plt.subplots(grid_size, grid_size, figsize=(8.5, 11))
        for (img_path, label), ax in zip(sample_images, axes.ravel()):
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(label)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_intensity_distribution(sample_image_paths):
        """Plots the pixel intensity distribution for a sample of images."""
        intensities = {"R": [], "G": [], "B": []}

        for img_path in sample_image_paths:
            img = Image.open(img_path)
            arr = np.array(img)
            # Check if the image is grayscale or RGB
            if arr.ndim == 2:  # grayscale image
                intensities["R"].extend(arr.ravel())
                intensities["G"].extend(arr.ravel())
                intensities["B"].extend(arr.ravel())
            else:  # RGB image
                for channel, color in enumerate(["R", "G", "B"]):
                    intensities[color].extend(arr[:, :, channel].ravel())

        # Plot the intensity distribution for each channel
        plt.figure(figsize=(10, 6))
        for color in ["R", "G", "B"]:
            plt.hist(
                intensities[color],
                bins=256,
                color=color.lower(),
                alpha=0.5,
                label=color,
            )
        plt.title("Pixel Intensity Distribution")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.legend(loc="upper right")
        plt.show()


if __name__ == "__main__":
    root_dir = "./FINAL-DATASET-5/train"
    classes = ["anger", "Boredom", "Engagement", "neutral"]

    # Load the data
    loader = DataLoader(root_dir, classes)
    visualizer = DataVisualizer()

    # Visualize the class distribution, sample images, and pixel intensity distribution
    visualizer.plot_class_distribution(loader.get_class_distribution())
    sample_images = loader.get_sample_images(25)
    visualizer.plot_sample_images(sample_images)
    sample_image_paths = [img[0] for img in sample_images]
    visualizer.plot_intensity_distribution(sample_image_paths)
