import os
import pandas as pd

# Define the destination root directory
destination_root = "FINAL-DATASET-2"  # Where you saved train, val, test folders


def generate_csv_for_split(split_name):
    """
    Generate a CSV file for the provided dataset split (train, test, or val).

    Parameters:
    - split_name: The name of the split ('train', 'test', or 'val').
    """
    split_dir = os.path.join(destination_root, split_name)

    # Lists to store data for the CSV
    indices = []
    image_names = []
    labels = []

    # Assign an index for each image
    idx = 1

    # Navigate through each class subdirectory in the split
    for label in os.listdir(split_dir):
        label_dir = os.path.join(split_dir, label)

        # Ensure the current label is a directory
        if os.path.isdir(label_dir):
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)

                # Ensure the current item is a file (an image)
                if os.path.isfile(image_path):
                    # Append data to our lists
                    indices.append(idx)
                    image_names.append(image_name)
                    labels.append(label)
                    idx += 1

    # Convert the lists into a DataFrame
    df = pd.DataFrame({"Index": indices, "ImageName": image_names, "Label": labels})

    # Save the DataFrame to a CSV file
    csv_path = os.path.join(split_dir, f"{split_name}_labels.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV generated for {split_name} at {csv_path}")


# Generate CSVs for train, test, and val splits
generate_csv_for_split("train")
generate_csv_for_split("test")
generate_csv_for_split("val")
