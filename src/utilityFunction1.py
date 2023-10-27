"""
Image Renaming and CSV Handling Utility

This script provides functions to rename images based on specific conditions
and create a new CSV based on the renamed images and an original CSV.

Author:
    Dhruvil Patel
"""

import os
import pandas as pd


def rename_images(folder_path: str) -> tuple:
    """
    Rename images in the folder based on the given conditions.

    Args:
        folder_path (str): The directory path containing the images to be renamed.

    Returns:
        tuple: A tuple containing a list of all the original image names (without extension)
               and a dictionary mapping original image names to their new names.
    """
    renamed_images = {}
    all_images = []

    for image_name in os.listdir(folder_path):
        base_name = image_name.split(".")[0]
        all_images.append(base_name)

        conditions = {"_2s": "_1", "_5s": "_2", "_7s": "_3"}

        for k, v in conditions.items():
            if k in image_name:
                new_name = image_name.replace(k, v)
                os.rename(
                    os.path.join(folder_path, image_name),
                    os.path.join(folder_path, new_name),
                )
                renamed_images[image_name] = new_name
                break

    return all_images, renamed_images


def create_new_csv(
    all_images: list,
    renamed_images: dict,
    csv_path: str,
    output_csv_path: str,
    folder_path: str,
):
    """
    Create a new CSV based on the renamed images and the original CSV.

    Args:
        all_images (list): List of all the original image names without extension.
        renamed_images (dict): A dictionary mapping original image names to their new names.
        csv_path (str): Path to the original CSV file.
        output_csv_path (str): Path to save the new CSV file.
        folder_path (str): Directory path containing the images.

    Returns:
        None
    """
    df = pd.read_csv(csv_path)
    new_data = []

    for image_name in os.listdir(folder_path):
        image_name_no_ext = image_name.split(".")[0]
        vid_name = image_name_no_ext.split("_")[0]

        matching_rows = df[df["ClipID"].str.contains(vid_name)]

        if not matching_rows.empty:
            row = matching_rows.iloc[0]

            if row["Engagement"] > max(
                row["Boredom"], row["Confusion"], row["Frustration "]
            ):
                label = "Engagement"
            else:
                label = "Boredom"

            new_data.append([row.name, image_name_no_ext, label])

    new_df = pd.DataFrame(new_data, columns=["Index", "ImageName", "Label"])
    new_df.to_csv(output_csv_path, index=False)


if __name__ == "_main_":
    all_images, renamed_images = rename_images("./FINAL-DATASET-4")
    create_new_csv(
        all_images,
        renamed_images,
        "./AllLabels.csv",
        "./NewLabels.csv",
        "./FINAL-DATASET-4",
    )
