"""
Frame Extractor from AVI Videos

This script extracts frames at the 2-second, 5-second and 7-second marks from every .avi video
in the provided directory and its subdirectories. The frames are saved as images in the
same directory as the video with suffixes "_2s.jpg", "_5s.jpg" and "_7s.jpg". If an image with the 
same name already exists, it will be overwritten.

Requirements:
    - Ensure ffmpeg (https://www.ffmpeg.org/) is installed for successful execution.

Author:
    Dhruv Nareshkumar Panchal
"""

import os
import subprocess


def extract_frames(video_path: str):
    """
    Extract frames from a video at 2 seconds, 5 seconds and 7 seconds.

    Args:
        video_path (str): The full path to the video file.

    Returns:
        None
    """
    base_name = os.path.splitext(video_path)[0]
    _extract_frame(video_path, base_name, 2)
    _extract_frame(video_path, base_name, 5)
    _extract_frame(video_path, base_name, 7)


def _extract_frame(video_path: str, base_name: str, seconds: int):
    """
    Helper function to extract a frame from a video at a specified time.

    Args:
        video_path (str): The full path to the video file.
        base_name (str): The base name of the video file (without extension).
        seconds (int): The time in seconds at which to extract the frame.

    Returns:
        None
    """
    output_image_file = f"{base_name}_{seconds}s.jpg"
    subprocess.check_output(
        f'ffmpeg -y -ss 00:00:{seconds:02} -i "{video_path}" -frames:v 1 "{output_image_file}" -hide_banner',
        shell=True,
    )


def main():
    """
    Main function to process all AVI videos in the given directory and its subdirectories.

    Returns:
        None
    """
    root_dir = os.path.join(os.getcwd(), "DataSet")
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".avi"):
                full_path = os.path.join(dirpath, file)
                print(f"Processing: {full_path}")
                extract_frames(full_path)

    print(
        "================================================================================\n"
        "Frame Extraction Successful at 2-second, 5-second and 7-second marks"
    )


if __name__ == "__main__":
    main()
