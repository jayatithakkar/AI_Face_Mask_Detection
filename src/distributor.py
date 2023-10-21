import pandas as pd
import os
import shutil

# Define the root path for the new images
new_root_path = os.path.join(os.getcwd(), "FINAL-DATASET-1")

# List to store new paths and labels
new_data = []
counter = 1

# Check if new_root_path exists, if not create it
if not os.path.exists(new_root_path):
    os.makedirs(new_root_path)

# Dataset-1

# Read the CSV
df = pd.read_csv("./affectnet-training-data/labels.csv")


# Loop through each row
for index, row in df.iterrows():
    old_path = os.path.join(
        os.getcwd(), "affectnet-training-data", row[1].replace("/", "\\")
    )
    label = row[2]
    # print(old_path, label)

    # Create a new directory for the label if it doesn't exist
    new_dir = os.path.join(new_root_path, label)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    # print(new_dir)

    # Copy the image to the new directory
    # new_path = os.path.join(new_dir, os.path.basename(old_path))
    # Create new filename based on counter
    new_filename = f"image{counter:05d}.jpg"  # assuming jpg format
    new_path = os.path.join(new_dir, new_filename)
    shutil.copy(old_path, new_path)

    # Store new path and label
    new_data.append([counter, new_filename, label])

    # Increment counter
    counter += 1

# Dataset-2A

# Read the CSV
df = pd.read_csv("./affectnet-sid/AffectNet/train.csv")

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

# Loop through each row
for index, row in df.iterrows():
    old_path = os.path.join(
        os.getcwd(), "affectnet-sid", "AffectNet", "train_images", row[3]
    )
    numeric_label = row[2]

    # Convert the numeric label to its string equivalent
    label = label_mapping.get(numeric_label, "unknown")
    # print(old_path, label)

    # Create a new directory for the label if it doesn't exist
    new_dir = os.path.join(new_root_path, label)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    # print(new_dir)

    # Copy the image to the new directory
    # new_path = os.path.join(new_dir, os.path.basename(old_path))
    # Create new filename based on counter
    new_filename = f"image{counter:05d}.jpg"  # assuming jpg format
    new_path = os.path.join(new_dir, new_filename)
    shutil.copy(old_path, new_path)

    # Store new path and label
    new_data.append([counter, new_filename, label])

    # Increment counter
    counter += 1

# Dataset-2B

# Read the CSV
df = pd.read_csv("./affectnet-sid/AffectNet/valid.csv")

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

# Loop through each row
for index, row in df.iterrows():
    old_path = os.path.join(
        os.getcwd(), "affectnet-sid", "AffectNet", "val_images", row[3]
    )
    numeric_label = row[2]

    # Convert the numeric label to its string equivalent
    label = label_mapping.get(numeric_label, "unknown")
    # print(old_path, label)

    # Create a new directory for the label if it doesn't exist
    new_dir = os.path.join(new_root_path, label)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    # print(new_dir)

    # Copy the image to the new directory
    # new_path = os.path.join(new_dir, os.path.basename(old_path))
    # Create new filename based on counter
    new_filename = f"image{counter:05d}.jpg"  # assuming jpg format
    new_path = os.path.join(new_dir, new_filename)
    shutil.copy(old_path, new_path)

    # Store new path and label
    new_data.append([counter, new_filename, label])

    # Increment counter
    counter += 1

# Create a new DataFrame and save to CSV
new_df = pd.DataFrame(new_data, columns=["Index", "ImageName", "Label"])
new_df.to_csv(os.path.join(new_root_path, "new_labels.csv"), index=False)

print("Done!")
