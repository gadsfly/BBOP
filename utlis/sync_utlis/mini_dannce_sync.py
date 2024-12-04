# video_sync.py

import os
import sys
import cv2
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath('../..'))
from utlis.sync_utlis.sync_df_utlis import (
    find_calib_file,
    calculate_frame_brightness,
    find_brightness_drop,


)


def find_camera_with_frame_start(rec_path):
    """
    Finds the camera index where 'data_frame' starts with 1 in the calibration data.

    Parameters:
    - rec_path (str): The base path where the recording data is located.

    Returns:
    - int: The camera index with 'data_frame' starting at 1.
    """
    calib_file = find_calib_file(rec_path)
    calib_data = scipy.io.loadmat(calib_file)
    sync_data = calib_data['sync']
    for cam_idx in range(sync_data.shape[0]):
        cam_sync = sync_data[cam_idx][0]
        data_frame = cam_sync['data_frame'][0][0][0]  # Directly access the known key
        if data_frame[0] == 1:
            print(f"Camera index with data_frame starting at 1: {cam_idx}")
            return cam_idx
    print("No camera with data_frame starting at 1 found.")
    return None

def calculate_frame_brightness(video_path, max_frames, min_frames=0):
    """
    Calculates the average brightness of frames in a video.

    Parameters:
    - video_path (str): Path to the video file.
    - max_frames (int): Maximum number of frames to process.
    - min_frames (int): Starting frame index for processing.

    Returns:
    - list: A list of average brightness values for each frame.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_brightness = []
    frame_number = 0

    while frame_number < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process frames within the specified range
        if frame_number >= min_frames:
            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate the average brightness
            avg_brightness = np.mean(gray_frame)
            frame_brightness.append(avg_brightness)

        frame_number += 1

    cap.release()
    return frame_brightness

def find_brightness_drop(brightness_values, threshold):
    """
    Finds the indices where a significant brightness drop occurs.

    Parameters:
    - brightness_values (list): List of brightness values.
    - threshold (float): The minimum difference to consider as a drop.

    Returns:
    - list: Indices where brightness drops occur.
    """
    drops = []
    for i in range(1, len(brightness_values)):
        if brightness_values[i - 1] - brightness_values[i] > threshold:
            drops.append(i)
    return drops

def plot_brightness_values(brightness_values, drop_indices, title='Brightness Values', label='Brightness'):
    """
    Plots the brightness values and marks the intensity drops.

    Parameters:
    - brightness_values (list): List of brightness values.
    - drop_indices (list): Indices where brightness drops occur.
    - title (str): Title of the plot.
    - label (str): Label for the brightness line in the plot.

    Returns:
    - None
    """
    plt.figure(figsize=(12, 6))
    plt.plot(brightness_values, label=label)
    for idx in drop_indices:
        plt.axvline(idx, color='red', linestyle='--', label='Intensity Drop' if idx == drop_indices[0] else "")
    plt.title(title)
    plt.xlabel('Frame Number')
    plt.ylabel('Brightness')
    plt.legend()
    plt.show()


def sync_videos(
    rec_path,
    mini_path,
    start_frame=0,
    end_frame=300,
    threshold_mini=3,
    threshold_sixcam=3
):
    """
    Synchronize brightness drops between Mini Cam and Six Cam videos.

    Parameters:
        rec_path (str): Path to the main recording folder.
        mini_path (str): Path to the Mini Cam folder.
        start_frame (int): Start frame for brightness analysis.
        end_frame (int): End frame for brightness analysis.
        threshold_mini (int): Threshold for detecting intensity drops in Mini Cam.
        threshold_sixcam (int): Threshold for detecting intensity drops in Six Cam.

    Returns:
        dict: Dictionary containing synchronization frames and drop indices.
    """
    # Define paths
    mini_cam_path = os.path.join(mini_path, 'My_First_WebCam')
    mini_cam_vid = os.path.join(mini_cam_path, '0.avi')

    miniscope_path = os.path.join(mini_path, 'My_V4_Miniscope')
    miniscope_timestamps = os.path.join(miniscope_path, 'timeStamps.csv')

    # Find the camera index with data_frame starting at 1
    camera_index = find_camera_with_frame_start(rec_path)
    if camera_index is None:
        raise ValueError("No valid camera index found.")

    # Construct the video path for the selected camera
    camera_number = camera_index + 1  # If camera indices start from 0
    sixcam_video_path = f'{rec_path}/videos/Camera{camera_number}/0.mp4'

    # Check if the sixcam video exists
    if not os.path.exists(sixcam_video_path):
        raise FileNotFoundError(f"6Cam video not found at {sixcam_video_path}")

    # Calculate frame brightness for mini cam video
    brightness_values_mini = calculate_frame_brightness(mini_cam_vid, end_frame, start_frame)

    # Calculate frame brightness for sixcam video
    brightness_values_sixcam = calculate_frame_brightness(sixcam_video_path, end_frame, start_frame)

    # Find intensity drops
    drop_indices_mini = find_brightness_drop(brightness_values_mini, threshold_mini)
    drop_indices_sixcam = find_brightness_drop(brightness_values_sixcam, threshold_sixcam)

    # Synchronization frames
    sync_frame_mini = drop_indices_mini[0] if drop_indices_mini else None
    sync_frame_sixcam = drop_indices_sixcam[0] if drop_indices_sixcam else None

    # Plotting for verification
    plot_brightness_values(
        brightness_values_mini,
        drop_indices_mini,
        title='Brightness Values - Mini Cam',
        label='Mini Cam Brightness',
    )

    plot_brightness_values(
        brightness_values_sixcam,
        drop_indices_sixcam,
        title='Brightness Values - 6Cam',
        label='6Cam Brightness',
    )

    return {
        "sync_frame_mini": sync_frame_mini,
        "sync_frame_sixcam": sync_frame_sixcam,
        "drop_indices_mini": drop_indices_mini,
        "drop_indices_sixcam": drop_indices_sixcam,
    }
