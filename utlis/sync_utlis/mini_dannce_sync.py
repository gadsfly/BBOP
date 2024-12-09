# video_sync.py

import os
import sys
import cv2
import csv
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
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
    calib_data = sio.loadmat(calib_file)
    sync_data = calib_data['sync']
    for cam_idx in range(sync_data.shape[0]):
        cam_sync = sync_data[cam_idx][0]
        data_frame = cam_sync['data_frame'][0][0][0]  # Directly access the known key
        if data_frame[0] == 1:
            print(f"Camera index with data_frame starting at 1: {cam_idx}")
            return cam_idx
    print("No camera with data_frame starting at 1 found.")
    return None

# def calculate_frame_brightness(video_path, max_frames, min_frames=0):
#     """
#     Calculates the average brightness of frames in a video.

#     Parameters:
#     - video_path (str): Path to the video file.
#     - max_frames (int): Maximum number of frames to process.
#     - min_frames (int): Starting frame index for processing.

#     Returns:
#     - list: A list of average brightness values for each frame.
#     """
#     # Open the video file
#     cap = cv2.VideoCapture(video_path)
#     frame_brightness = []
#     frame_number = 0

#     while frame_number < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Only process frames within the specified range
#         if frame_number >= min_frames:
#             # Convert frame to grayscale
#             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             # Calculate the average brightness
#             avg_brightness = np.mean(gray_frame)
#             frame_brightness.append(avg_brightness)

#         frame_number += 1

#     cap.release()
#     return frame_brightness

# def find_brightness_drop(brightness_values, threshold):
#     """
#     Finds the indices where a significant brightness drop occurs.

#     Parameters:
#     - brightness_values (list): List of brightness values.
#     - threshold (float): The minimum difference to consider as a drop.

#     Returns:
#     - list: Indices where brightness drops occur.
#     """
#     drops = []
#     for i in range(1, len(brightness_values)):
#         if brightness_values[i - 1] - brightness_values[i] > threshold:
#             drops.append(i)
#     return drops

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
        title=f'Brightness Values - Cam{camera_number}',
        label=f'Cam{camera_number} Brightness',
    )

    return {
        "sync_frame_mini": sync_frame_mini,
        "sync_frame_sixcam": sync_frame_sixcam,
        "drop_indices_mini": drop_indices_mini,
        "drop_indices_sixcam": drop_indices_sixcam,
        "cam_numb": camera_number
    }


def load_mini_timestamps(csv_path):
    timestamps = []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            frame_number, timestamp_ms, _ = row
            # timestamps.append(int(timestamp_ms))
            timestamps.append(int(timestamp_ms))  # deleted Convert to seconds / 1000.0
    return timestamps

def load_sixcam_timestamps(rec_path, camera_number):
    frametimes_path = f'{rec_path}/videos/Camera{camera_number}/frametimes.npy'
    if os.path.exists(frametimes_path):
        return np.load(frametimes_path) * 1000 # s to ms
    else:
        raise FileNotFoundError(f"Frametimes file not found at {frametimes_path}")


def align_miniscope_to_sixcam(resultsss, mini_path, rec_path):
    # Extract required data
    sync_frame_mini = resultsss.get("sync_frame_mini")
    sync_frame_sixcam = resultsss.get("sync_frame_sixcam")
    drop_indices_mini = resultsss.get("drop_indices_mini")
    drop_indices_sixcam = resultsss.get("drop_indices_sixcam")
    camera_number = resultsss.get("cam_numb")
    
    # Miniscope webcam timestamps
    mini_cam_path = os.path.join(mini_path, 'My_First_WebCam')
    mini_cam_timestamps_p = os.path.join(mini_cam_path, 'timeStamps.csv')
    mini_cam_vid = os.path.join(mini_cam_path, '0.avi')
    
    miniscope_path = os.path.join(mini_path, 'My_V4_Miniscope')
    miniscope_timestamps_p = os.path.join(miniscope_path, 'timeStamps.csv')
    
    # Load timestamps
    mini_cam_timestamps = load_mini_timestamps(mini_cam_timestamps_p)
    mini_timstampsss = load_mini_timestamps(miniscope_timestamps_p)
    sync_time_mini_cam = mini_cam_timestamps[sync_frame_mini]
    
    # Load 6Cam timestamps
    sixcam_timestamps = load_sixcam_timestamps(rec_path, camera_number)
    sync_time_sixcam = sixcam_timestamps[1][sync_frame_sixcam]
    
    # Calculate time offset
    time_offset = sync_time_mini_cam - sync_time_sixcam
    print("offset: ", time_offset)
    adjusted_sixcam_timestamps = sixcam_timestamps[1] + time_offset
    
    # Create interpolation function
    interp_func = interp1d(adjusted_sixcam_timestamps, np.arange(len(adjusted_sixcam_timestamps)), kind='nearest', fill_value='extrapolate')
    mapped_sixcam_frame_indices = interp_func(mini_timstampsss).astype(int)
    
    if len(mapped_sixcam_frame_indices) != len(mini_timstampsss):
        raise ValueError(
            f"Length mismatch: {len(mapped_sixcam_frame_indices)} (mapped indices) "
            f"!= {len(mini_timstampsss)} (miniscope timestamps)."
        )
    
    # Paths for output and data
    pred_folder = 'DANNCE/predict00'
    pred_path = os.path.join(rec_path, pred_folder, 'save_data_AVG.mat')
    com_file = os.path.join(rec_path, pred_folder, 'com3d_used.mat')
    save_path = os.path.join(rec_path, "MIR_Aligned_preds")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    hdf5_output_path = os.path.join(save_path, 'aligned_predictions.h5')
    
    # Load COM data and 3D predictions
    com_data = sio.loadmat(com_file)['com']
    pred_3d = sio.loadmat(pred_path)['pred']
    pred_3d = np.squeeze(pred_3d, axis=1)
    
    # Validate and filter data
    valid_mask = (mapped_sixcam_frame_indices >= 0) & (mapped_sixcam_frame_indices < pred_3d.shape[0])
    mini_cam_timestamps_s = np.array(mini_timstampsss)[valid_mask]
    mapped_sixcam_frame_indices = mapped_sixcam_frame_indices[valid_mask]
    aligned_com = com_data[mapped_sixcam_frame_indices, :]
    aligned_pred_3d = pred_3d[mapped_sixcam_frame_indices, :, :]
    N_miniscope_frames = len(mini_cam_timestamps_s)
    
    # Flatten 3D predictions
    aligned_pred_3d_flat = aligned_pred_3d.reshape(N_miniscope_frames, 22 * 3)
    
    # Create DataFrame
    com_cols = ['com_x', 'com_y', 'com_z']
    kp_cols = [f'kp{kp_idx}_{coord}' for kp_idx in range(1, 23) for coord in ['x', 'y', 'z']]
    all_columns = com_cols + kp_cols
    combined_data = np.hstack([aligned_com, aligned_pred_3d_flat])
    df = pd.DataFrame(data=combined_data, index=mini_cam_timestamps_s, columns=all_columns)
    df.index.name = 'timestamp_ms_mini'
    
    # Save to HDF5
    df.to_hdf(hdf5_output_path, key='df', mode='w')
    print(f"Aligned data saved to {hdf5_output_path}")
    
    # Print shapes
    print("aligned_pred_3d_flat.shape:", aligned_pred_3d_flat.shape)
    print("aligned_pred_3d.shape:", aligned_pred_3d.shape)
    print("aligned_com.shape:", aligned_com.shape)


#since prob i will need to adjust some threshold and start and end frame and stuff, cannot automate yet. so now will have to use sync_videos functiona and align function, from above. cannot just write a function to call both or something...
# def align_and_save_recmini():
