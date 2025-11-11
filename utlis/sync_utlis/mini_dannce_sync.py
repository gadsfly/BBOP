# video_sync.py
import glob
import os
import sys
# import cv2
import xarray as xr
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
import json


#adding this to handle soscial situations:
def load_and_reshape_com(com_mat):

    raw = sio.loadmat(com_mat)['com']
    # already (frames,3,n)?
    if raw.ndim == 3 and raw.shape[1] == 3:
        return raw
    # flattened (frames,3*n)?
    if raw.ndim == 2 and raw.shape[1] % 3 == 0:
        n = raw.shape[1] // 3
        return raw.reshape(-1, 3, n)
    # single-mouse (frames,3)
    if raw.ndim == 2 and raw.shape[1] == 3:
        return raw[:, :, np.newaxis]
    raise ValueError(f"Unexpected COM shape {raw.shape}")



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


#below version works, just not saving the plots. now i would like to save them fro any future references
# def plot_brightness_values(brightness_values, drop_indices, title='Brightness Values', label='Brightness'):
#     """
#     Plots the brightness values and marks the intensity drops.

#     Parameters:
#     - brightness_values (list): List of brightness values.
#     - drop_indices (list): Indices where brightness drops occur.
#     - title (str): Title of the plot.
#     - label (str): Label for the brightness line in the plot.

#     Returns:
#     - None
#     """
#     plt.figure(figsize=(12, 6))
#     plt.plot(brightness_values, label=label)
#     for idx in drop_indices:
#         plt.axvline(idx, color='red', linestyle='--', label='Intensity Drop' if idx == drop_indices[0] else "")
#     plt.title(title)
#     plt.xlabel('Frame Number')
#     plt.ylabel('Brightness')
#     plt.legend()
#     plt.show()

def plot_brightness_values(
    brightness_values,
    drop_indices,
    title='Brightness Values',
    label='Brightness',
    save_paths=None     # ← now accepts str or list[str]
):
    """
    Plots the brightness values and marks the intensity drops.
    If `save_paths` is provided, saves the figure to each path.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(brightness_values, label=label)
    for idx in drop_indices:
        plt.axvline(
            idx,
            color='red',
            linestyle='--',
            label='Intensity Drop' if idx == drop_indices[0] else ""
        )
    plt.title(title)
    plt.xlabel('Frame Number')
    plt.ylabel('Brightness')
    plt.legend()

    if save_paths:
        # normalize to list
        paths = save_paths if isinstance(save_paths, (list, tuple)) else [save_paths]
        for p in paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            plt.savefig(p)
            print(f"Saved plot to {p}")

    plt.show()
    plt.close()

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
    if not os.path.exists(mini_cam_path):
        mini_cam_path = os.path.join(mini_path, 'My_WebCam')
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

    # # Plotting for verification
    # plot_brightness_values(
    #     brightness_values_mini,
    #     drop_indices_mini,
    #     title='Brightness Values - Mini Cam',
    #     label='Mini Cam Brightness',
    # )

    # plot_brightness_values(
    #     brightness_values_sixcam,
    #     drop_indices_sixcam,
    #     title=f'Brightness Values - Cam{camera_number}',
    #     label=f'Cam{camera_number} Brightness',
    # )

    # return {
    #     "sync_frame_mini": sync_frame_mini,
    #     "sync_frame_sixcam": sync_frame_sixcam,
    #     "drop_indices_mini": drop_indices_mini,
    #     "drop_indices_sixcam": drop_indices_sixcam,
    #     "cam_numb": camera_number
    # }

        # ----------------
    # Mini Cam plot → save in both places
    mini_save_m = os.path.join(mini_cam_path,         'sync_mini.png')
    mini_save_r = os.path.join(rec_path,              'sync_mini.png')
    plot_brightness_values(
        brightness_values_mini,
        drop_indices_mini,
        title='Brightness Values - Mini Cam',
        label='Mini Cam Brightness',
        save_paths=[mini_save_m, mini_save_r]
    )

    # ----------------
    # Six Cam plot → save in both places
    six_fname    = f'sync_cam{camera_number}.png'
    six_save_m   = os.path.join(mini_cam_path,         six_fname)
    six_save_r   = os.path.join(rec_path,              six_fname)
    plot_brightness_values(
        brightness_values_sixcam,
        drop_indices_sixcam,
        title=f'Brightness Values - Cam{camera_number}',
        label=f'Cam{camera_number} Brightness',
        save_paths=[six_save_m, six_save_r]
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

# works well below, just trying to add the social part, did not test though. also trying to add the com only option. anywyas below should sort of work well
# no below did not work. apprently below would not take into consideration of shape or whatever. the xyz, xyc is somehow loaded as xx,yy,zz,which is wrong
# def align_miniscope_to_sixcam(resultsss, mini_path, rec_path):
#     # Extract required data
#     sync_frame_mini = resultsss.get("sync_frame_mini")
#     sync_frame_sixcam = resultsss.get("sync_frame_sixcam")
#     drop_indices_mini = resultsss.get("drop_indices_mini")
#     drop_indices_sixcam = resultsss.get("drop_indices_sixcam")
#     camera_number = resultsss.get("cam_numb")
    
#     # Miniscope webcam timestamps
#     mini_cam_path = os.path.join(mini_path, 'My_First_WebCam')
#     if not os.path.exists(mini_cam_path):
#         mini_cam_path = os.path.join(mini_path, 'My_WebCam')
#     mini_cam_timestamps_p = os.path.join(mini_cam_path, 'timeStamps.csv')
#     mini_cam_vid = os.path.join(mini_cam_path, '0.avi')
    
#     miniscope_path = os.path.join(mini_path, 'My_V4_Miniscope')
#     miniscope_timestamps_p = os.path.join(miniscope_path, 'timeStamps.csv')
    
#     # Load timestamps
#     mini_cam_timestamps = load_mini_timestamps(mini_cam_timestamps_p)
#     mini_timstampsss = load_mini_timestamps(miniscope_timestamps_p)
#     sync_time_mini_cam = mini_cam_timestamps[sync_frame_mini]
    
#     # Load 6Cam timestamps
#     sixcam_timestamps = load_sixcam_timestamps(rec_path, camera_number)
#     sync_time_sixcam = sixcam_timestamps[1][sync_frame_sixcam]
    
#     # Calculate time offset
#     time_offset = sync_time_mini_cam - sync_time_sixcam
#     print("offset: ", time_offset)
#     adjusted_sixcam_timestamps = sixcam_timestamps[1] + time_offset
    
#     # Create interpolation function
#     interp_func = interp1d(adjusted_sixcam_timestamps, np.arange(len(adjusted_sixcam_timestamps)), kind='nearest', fill_value='extrapolate')
#     mapped_sixcam_frame_indices = interp_func(mini_timstampsss).astype(int)
    
#     if len(mapped_sixcam_frame_indices) != len(mini_timstampsss):
#         raise ValueError(
#             f"Length mismatch: {len(mapped_sixcam_frame_indices)} (mapped indices) "
#             f"!= {len(mini_timstampsss)} (miniscope timestamps)."
#         )
    
#     # Paths for output and data
#     pred_folder = 'DANNCE/predict00'
#     pred_path = os.path.join(rec_path, pred_folder, 'save_data_AVG.mat')
#     com_file = os.path.join(rec_path, pred_folder, 'com3d_used.mat')
#     save_path = os.path.join(rec_path, "MIR_Aligned")
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     hdf5_output_path = os.path.join(save_path, 'aligned_predictions.h5')
    
#     # Load COM data and 3D predictions
#     # com_data = sio.loadmat(com_file)['com']
#     com_data = load_and_reshape_com(rec_path, pred_folder)

#     pred_3d = sio.loadmat(pred_path)['pred']
#     pred_3d = np.squeeze(pred_3d, axis=1)

#     if pred_3d.shape[1] == 3 and pred_3d.shape[2] == 22:
#         print(f'pred current shape{pred_3d.shape}')
#         pred_3d = pred_3d.transpose((0, 2, 1))  # Now it's (N_frames, 22, 3)
#         print(f'preds transposed to {pred_3d.shape}')

    
#     # Validate and filter data
#     valid_mask = (mapped_sixcam_frame_indices >= 0) & (mapped_sixcam_frame_indices < pred_3d.shape[0])
#     mini_cam_timestamps_s = np.array(mini_timstampsss)[valid_mask]
#     mapped_sixcam_frame_indices = mapped_sixcam_frame_indices[valid_mask]
#     # aligned_com = com_data[mapped_sixcam_frame_indices, :]
#     aligned_com = com_data[mapped_sixcam_frame_indices, :, :]  # shape (F,3,n_animals)
#     aligned_pred_3d = pred_3d[mapped_sixcam_frame_indices, :, :]
#     N_miniscope_frames = len(mini_cam_timestamps_s)
    
#     # Flatten 3D predictions
#     aligned_pred_3d_flat = aligned_pred_3d.reshape(N_miniscope_frames, 22 * 3)
    
#     # Create DataFrame
#     # com_cols = ['com_x', 'com_y', 'com_z']
#     n_animals = aligned_com.shape[2]
#     # e.g. ['com1_x','com1_y','com1_z','com2_x',…]
#     com_cols = [
#         f'com{a+1}_{axis}'
#         for a in range(n_animals)
#         for axis in ('x','y','z')
#     ]
#     aligned_com_flat = aligned_com.reshape(F, 3*n_animals)
#     kp_cols = [f'kp{kp_idx}_{coord}' for kp_idx in range(1, 23) for coord in ['x', 'y', 'z']]
#     # kp_cols = [f'kp{kp_idx}_{coord}' for coord in ['x', 'y', 'z'] for kp_idx in range(1, 23)]

#     all_columns = com_cols + kp_cols
#     combined_data = np.hstack([aligned_com_flat, aligned_pred_3d_flat])
#     df = pd.DataFrame(data=combined_data, index=mini_cam_timestamps_s, columns=all_columns)
#     df.index.name = 'timestamp_ms_mini'
    

#     df['camera_frame_sixcam'] = mapped_sixcam_frame_indices

#     # Save to HDF5
#     df.to_hdf(hdf5_output_path, key='df', mode='w')
#     print(f"Aligned data saved to {hdf5_output_path}")
    

#     # ----------------------------
#     # We'll store both in a JSON file for simplicity.
#     map_data = {
#         "time_offset": float(time_offset),  # ensure it's JSON-serializable
#         "mapped_sixcam_frame_indices": mapped_sixcam_frame_indices.tolist(),
#         "mini_cam_timestamps": mini_cam_timestamps_s.tolist()
#     }
#     json_mapping_file = os.path.join(save_path, "frame_mapping.json")
#     with open(json_mapping_file, "w") as f:
#         json.dump(map_data, f, indent=2)
#     print(f"Frame mapping data saved to {json_mapping_file}")
#     # ----------------------------

#     # Print shapes
#     print("aligned_pred_3d_flat.shape:", aligned_pred_3d_flat.shape)
#     print("aligned_pred_3d.shape:", aligned_pred_3d.shape)
#     print("aligned_com.shape:", aligned_com.shape)
#     return df



def align_miniscope_to_sixcam(resultsss, mini_path, rec_path, only_com=False):
    # Extract required data
    sync_frame_mini = resultsss.get("sync_frame_mini")
    sync_frame_sixcam = resultsss.get("sync_frame_sixcam")
    camera_number = resultsss.get("cam_numb")

    # Paths for timestamps
    mini_cam_path = os.path.join(mini_path, 'My_First_WebCam')
    if not os.path.exists(mini_cam_path):
        mini_cam_path = os.path.join(mini_path, 'My_WebCam')
    mini_cam_timestamps_p = os.path.join(mini_cam_path, 'timeStamps.csv')
    miniscope_path = os.path.join(mini_path, 'My_V4_Miniscope')
    miniscope_timestamps_p = os.path.join(miniscope_path, 'timeStamps.csv')

    # Load timestamps
    mini_cam_timestamps = load_mini_timestamps(mini_cam_timestamps_p)
    mini_timstamps = load_mini_timestamps(miniscope_timestamps_p)
    sync_time_mini_cam = mini_cam_timestamps[sync_frame_mini]

    # Load and adjust 6Cam timestamps
    sixcam_timestamps = load_sixcam_timestamps(rec_path, camera_number)
    sync_time_sixcam = sixcam_timestamps[1][sync_frame_sixcam]
    time_offset = sync_time_mini_cam - sync_time_sixcam
    adjusted_sixcam = sixcam_timestamps[1] + time_offset

    # Map frames
    interp_func = interp1d(adjusted_sixcam, np.arange(len(adjusted_sixcam)),
                            kind='nearest', fill_value='extrapolate')
    mapped_idx = interp_func(mini_timstamps).astype(int)
    if len(mapped_idx) != len(mini_timstamps):
        raise ValueError(
            f"Length mismatch: {len(mapped_idx)} != {len(mini_timstamps)}"
        )

    # Prepare output paths
    pred_folder = 'DANNCE/predict00'
    pred_path = os.path.join(rec_path, pred_folder, 'save_data_AVG.mat')
    # com_file = os.path.join(rec_path, pred_folder, 'com3d_used.mat')
    # choose prediction folder and COM file based on only_com flag
    pred_folder = 'COM/predict00' if only_com else 'DANNCE/predict00'
    com_filename = 'com3d0.mat'    if only_com else 'com3d_used.mat'
    com_file      = os.path.join(rec_path, pred_folder, com_filename)

    save_path = os.path.join(rec_path, 'MIR_Aligned')
    os.makedirs(save_path, exist_ok=True)
    # h5_path = os.path.join(save_path, 'aligned_predictions.h5')
    # Prefix h5 filename with only_com if requested
    filename = f"{'only_com_' if only_com else ''}aligned_predictions.h5"
    h5_path = os.path.join(save_path, filename)

    # Load COM data (shape: frames × 3 × n_animals)
    com_data = load_and_reshape_com(com_file)

    # Optionally load predictions
    if not only_com:
        pred_3d = sio.loadmat(pred_path)['pred']
        pred_3d = np.squeeze(pred_3d, axis=1)
        if pred_3d.shape[1] == 3 and pred_3d.shape[2] == 22:
            pred_3d = pred_3d.transpose((0, 2, 1))  # (F,22,3)

    # Filter valid indices
    mini_cam_timestamps_s = np.array(mini_timstamps)[
        (mapped_idx >= 0) & (mapped_idx < (pred_3d.shape[0] if not only_com else com_data.shape[0]))
    ]
    valid_mask = (mapped_idx >= 0) & (mapped_idx < (pred_3d.shape[0] if not only_com else com_data.shape[0]))
    mapped_idx = mapped_idx[valid_mask]

    # Align data
    aligned_com = com_data[mapped_idx, :, :]  # (F,3,n_animals)
    N = len(mini_cam_timestamps_s)

    # Flatten COM
    n_animals = aligned_com.shape[2]
    com_flat = aligned_com.reshape(N, 3 * n_animals)
    # com_cols = [f'com{a+1}_{axis}' for a in range(n_animals) for axis in ('x','y','z')]
    com_cols = [f'com{a+1}_{axis}' for axis in ('x','y','z') for a in range(n_animals)]

    # Flatten predictions if requested
    if not only_com:
        aligned_pred = pred_3d[mapped_idx, :, :]  # (F,22,3)
        pred_flat = aligned_pred.reshape(N, 22 * 3)
        # need to revise for multi animals later lol
        kp_cols = [f'kp{idx}_{coord}' for idx in range(1, 23) for coord in ('x','y','z')]
        data_matrix = np.hstack([com_flat, pred_flat])
        columns = com_cols + kp_cols
    else:
        data_matrix = com_flat
        columns = com_cols

    # Build DataFrame
    df = pd.DataFrame(data=data_matrix,
                      index=mini_cam_timestamps_s,
                      columns=columns)
    df.index.name = 'timestamp_ms_mini'
    df['camera_frame_sixcam'] = mapped_idx

    # Save results
    df.to_hdf(h5_path, key='df', mode='w')
    with open(os.path.join(save_path, 'frame_mapping.json'), 'w') as f:
        json.dump({
            'time_offset': float(time_offset),
            'mapped_sixcam_frame_indices': mapped_idx.tolist(),
            'mini_cam_timestamps': mini_cam_timestamps_s.tolist()
        }, f, indent=2)

    print(f"Aligned data saved to {h5_path}")
    return df


#since prob i will need to adjust some threshold and start and end frame and stuff, cannot automate yet. so now will have to use sync_videos functiona and align function, from above. cannot just write a function to call both or something...
# def align_and_save_recmini():


def load_aligneddannce_and_process_ca_data(rec_path, mini_path):
    """
    Process calcium imaging data, interpolate to match miniscope timestamps, calculate \u0394F/F, 
    and save the merged data into an HDF5 file.

    Parameters:
        rec_path (str): Path to the recording folder containing aligned_predictions.h5.
        mini_path (str): Path to the miniscope data folder containing timeStamps.csv and .nc file.
    """
    # Load miniscope data
    hdf5_input_path = os.path.join(rec_path, 'MIR_Aligned/aligned_predictions.h5')
    df = pd.read_hdf(hdf5_input_path, key='df')
    mini_cam_timestamps = df.index.values
    print("Data loaded successfully!")

    # Miniscope data paths
    miniscope_path = os.path.join(mini_path, 'My_V4_Miniscope')
    miniscope_timestamps_path = os.path.join(miniscope_path, 'timeStamps.csv')

    # Load .nc file containing calcium imaging data
    nc_files = glob.glob(os.path.join(miniscope_path, '*.nc'))
    if not nc_files:
        raise FileNotFoundError("No .nc files found in the specified miniscope path.")
    ca_file_path = nc_files[0]

    # Load calcium data
    data = xr.open_dataset(ca_file_path)

    # Load timestamps
    timestamps = pd.read_csv(miniscope_timestamps_path)
    ts = timestamps['Time Stamp (ms)'].values  # shape (num_ca_frames,)

    C = data['C'].values  # Calcium signals
    num_neuron, num_frame = C.shape
    print("Calcium data shape:", C.shape)
    print("Timestamps shape:", ts.shape)

    # Interpolate calcium signals to miniscope timestamps
    C_interpolated = []
    for i_neuron in range(num_neuron):
        func = interp1d(ts, C[i_neuron, :], kind='linear', bounds_error=False, fill_value='extrapolate')
        neuron_interp = func(mini_cam_timestamps)
        C_interpolated.append(neuron_interp)

    C_interpolated = np.array(C_interpolated)  # shape (num_neuron, N_miniscope_frames)
    print("Interpolated Ca data shape:", C_interpolated.shape)

    # Calculate \u0394F/F using interpolated calcium data
    F0_interpolated = np.percentile(C_interpolated, 20, axis=1, keepdims=True)  # shape (num_neuron, 1)
    dF_F_interpolated = (C_interpolated - F0_interpolated) / F0_interpolated  # shape (num_neuron, N_miniscope_frames)
    print("\u0394F/F interpolated shape:", dF_F_interpolated.shape)

    # Add calcium and \u0394F/F columns to DataFrame
    calcium_cols = [f'calcium_roi{i}' for i in range(num_neuron)]
    dF_F_cols = [f'dF_F_roi{i}' for i in range(num_neuron)]

    # Transpose to match DataFrame format
    C_interpolated_transposed = C_interpolated.T  # shape (N_miniscope_frames, num_neuron)
    dF_F_interpolated_transposed = dF_F_interpolated.T  # shape (N_miniscope_frames, num_neuron)

    # Create DataFrames for calcium and \u0394F/F data
    df_ca = pd.DataFrame(data=C_interpolated_transposed, index=mini_cam_timestamps, columns=calcium_cols)
    df_dF_F = pd.DataFrame(data=dF_F_interpolated_transposed, index=mini_cam_timestamps, columns=dF_F_cols)

    # Set index name
    df_ca.index.name = 'timestamp_ms'
    df_dF_F.index.name = 'timestamp_ms'

    # Merge with the existing DataFrame
    df_merged = df.join(df_ca, how='inner')
    df_merged_with_dF_F = df_merged.join(df_dF_F, how='inner')

    print("DataFrame with Ca and \u0394F/F signals merged:")
    # print(df_merged_with_dF_F.head())

    # Save updated DataFrame
    updated_hdf5_path = os.path.join(rec_path, 'MIR_Aligned', 'aligned_predictions_with_ca_and_dF_F.h5')
    df_merged_with_dF_F.to_hdf(updated_hdf5_path, key='df', mode='w')
    print(f"Updated DataFrame with Ca and \u0394F/F data saved to {updated_hdf5_path}")

    # also save as CSV
    csv_output_path = os.path.join(rec_path, 'MIR_Aligned', 'aligned_predictions_with_ca_and_dF_F.csv')
    df_merged_with_dF_F.to_csv(csv_output_path, index=True)
    print(f"CSV saved to {csv_output_path}")


    return df_merged_with_dF_F


def load_aligneddannce_and_process_ca_data_custom(rec_path, mini_path, nc_file=None):
    """
    Process calcium imaging data, interpolate to match miniscope timestamps, calculate ΔF/F, 
    and save the merged data into an HDF5 file.
    
    Parameters:
        rec_path (str): Path to the recording folder containing aligned_predictions.h5.
        mini_path (str): Path to the miniscope data folder containing timeStamps.csv and .nc file.
        nc_file (str, optional): If provided, it can be either the full path to the .nc file or 
                                 a key (e.g., 'wnd1500_stp700_max15_diff3.5_pnrauto') that is used 
                                 to build the file name (as "minian_dataset_<key>.nc").
    """

    # Load miniscope data
    hdf5_input_path = os.path.join(rec_path, 'MIR_Aligned', 'aligned_predictions.h5')
    df = pd.read_hdf(hdf5_input_path, key='df')
    mini_cam_timestamps = df.index.values
    print("Data loaded successfully!")
    
    # Define paths for miniscope data
    miniscope_path = os.path.join(mini_path, 'My_V4_Miniscope')
    miniscope_timestamps_path = os.path.join(miniscope_path, 'timeStamps.csv')
    
    # Determine which .nc file to use:
    if nc_file is not None:
        # If the provided string is an existing file, use it as-is.
        if os.path.exists(nc_file):
            ca_file_path = nc_file
        else:
            # Otherwise, assume it's a key and build the filename accordingly.
            ca_file_path = os.path.join(miniscope_path, f"minian_dataset_{nc_file}.nc")
            if not os.path.exists(ca_file_path):
                raise FileNotFoundError(f"No .nc file found using the key '{nc_file}' at {ca_file_path}.")
    else:
        nc_files = glob.glob(os.path.join(miniscope_path, '*.nc'))
        if not nc_files:
            raise FileNotFoundError("No .nc files found in the specified miniscope path.")
        ca_file_path = nc_files[0]
    
    # Load calcium data
    data = xr.open_dataset(ca_file_path)
    
    # Load timestamps for the miniscope data
    timestamps = pd.read_csv(miniscope_timestamps_path)
    ts = timestamps['Time Stamp (ms)'].values  # shape (num_ca_frames,)
    
    C = data['C'].values  # Calcium signals
    num_neuron, num_frame = C.shape
    print("Calcium data shape:", C.shape)
    print("Timestamps shape:", ts.shape)
    
    # Interpolate calcium signals to match miniscope timestamps
    C_interpolated = []
    for i_neuron in range(num_neuron):
        func = interp1d(ts, C[i_neuron, :], kind='linear', bounds_error=False, fill_value='extrapolate')
        neuron_interp = func(mini_cam_timestamps)
        C_interpolated.append(neuron_interp)
    C_interpolated = np.array(C_interpolated)  # shape (num_neuron, N_miniscope_frames)
    print("Interpolated Ca data shape:", C_interpolated.shape)
    
    # Calculate ΔF/F using the interpolated calcium data
    F0_interpolated = np.percentile(C_interpolated, 20, axis=1, keepdims=True)
    dF_F_interpolated = (C_interpolated - F0_interpolated) / F0_interpolated
    print("ΔF/F interpolated shape:", dF_F_interpolated.shape)
    
    # Create column names for calcium and ΔF/F data
    calcium_cols = [f'calcium_roi{i}' for i in range(num_neuron)]
    dF_F_cols = [f'dF_F_roi{i}' for i in range(num_neuron)]
    
    # Transpose arrays to match the DataFrame format (frames x neurons)
    C_interpolated_transposed = C_interpolated.T
    dF_F_interpolated_transposed = dF_F_interpolated.T
    
    # Create DataFrames for the calcium and ΔF/F signals
    df_ca = pd.DataFrame(data=C_interpolated_transposed, index=mini_cam_timestamps, columns=calcium_cols)
    df_dF_F = pd.DataFrame(data=dF_F_interpolated_transposed, index=mini_cam_timestamps, columns=dF_F_cols)
    
    df_ca.index.name = 'timestamp_ms'
    df_dF_F.index.name = 'timestamp_ms'
    
    # Merge the new data with the existing DataFrame
    df_merged = df.join(df_ca, how='inner')
    df_merged_with_dF_F = df_merged.join(df_dF_F, how='inner')
    print("DataFrame with Ca and ΔF/F signals merged:")
    # print(df_merged_with_dF_F.head())
    
    # Save the updated DataFrame to a new HDF5 file
    updated_hdf5_path = os.path.join(rec_path, 'MIR_Aligned', 'aligned_predictions_with_ca_and_dF_F.h5')
    df_merged_with_dF_F.to_hdf(updated_hdf5_path, key='df', mode='w')
    print(f"Updated DataFrame with Ca and ΔF/F data saved to {updated_hdf5_path}")

        # also save as CSV
    csv_output_path = os.path.join(rec_path, 'MIR_Aligned', 'aligned_predictions_with_ca_and_dF_F.csv')
    df_merged_with_dF_F.to_csv(csv_output_path, index=True)
    print(f"CSV saved to {csv_output_path}")

    
    return df_merged_with_dF_F

# #this below version works well but, does not take into considerations when there is no key. usually when there is no key the default one is good enough
# def load_aligneddannce_and_process_ca_data_custom_key(rec_path, mini_path, nc_key):
#     """
#     Process calcium imaging data, interpolate to match miniscope timestamps, calculate ΔF/F, 
#     and save the merged data into an HDF5 file. The output HDF5 filenames are customized using
#     the provided output_key.
    
#     Parameters:
#         rec_path (str): Path to the recording folder containing the aligned predictions.
#         mini_path (str): Path to the miniscope data folder containing timeStamps.csv and .nc file.
#         output_key (str): A key string to customize the output HDF5 file names.
#         nc_file (str, optional): If provided, either the full path to the .nc file or 
#                                  a key used to build the file name (as "minian_dataset_<key>.nc").
#     """

#     # Determine input and output HDF5 filenames using the output_key
#     hdf5_input_filename = "aligned_predictions.h5"
#     updated_hdf5_filename = f"aligned_predictions_with_ca_and_dF_F_{nc_key}.h5"
    
#     hdf5_input_path = os.path.join(rec_path, 'MIR_Aligned', hdf5_input_filename)
#     df = pd.read_hdf(hdf5_input_path, key='df')
#     mini_cam_timestamps = df.index.values
#     print("Data loaded successfully!")
    
#     # Define paths for miniscope data
#     miniscope_path = os.path.join(mini_path, 'My_V4_Miniscope')
#     miniscope_timestamps_path = os.path.join(miniscope_path, 'timeStamps.csv')
    
    
    
#     ca_file_path = os.path.join(miniscope_path, f"minian_dataset_{nc_key}.nc")
#     if not os.path.exists(ca_file_path):
#         raise FileNotFoundError(f"No .nc file found using the key '{nc_key}' at {ca_file_path}.")
#     # else:
#     #     nc_files = glob.glob(os.path.join(miniscope_path, '*.nc'))
#     #     if not nc_files:
#     #         raise FileNotFoundError("No .nc files found in the specified miniscope path.")
            
    
#     # Load calcium data
#     data = xr.open_dataset(ca_file_path)
    
#     # Load timestamps for the miniscope data
#     timestamps = pd.read_csv(miniscope_timestamps_path)
#     ts = timestamps['Time Stamp (ms)'].values  # shape (num_ca_frames,)
    
#     C = data['C'].values  # Calcium signals
#     num_neuron, num_frame = C.shape
#     print("Calcium data shape:", C.shape)
#     print("Timestamps shape:", ts.shape)
    
#     # Interpolate calcium signals to match miniscope timestamps
#     C_interpolated = []
#     for i_neuron in range(num_neuron):
#         func = interp1d(ts, C[i_neuron, :], kind='linear', bounds_error=False, fill_value='extrapolate')
#         neuron_interp = func(mini_cam_timestamps)
#         C_interpolated.append(neuron_interp)
#     C_interpolated = np.array(C_interpolated)  # shape (num_neuron, N_miniscope_frames)
#     print("Interpolated Ca data shape:", C_interpolated.shape)
    
#     # Calculate ΔF/F using the interpolated calcium data
#     F0_interpolated = np.percentile(C_interpolated, 20, axis=1, keepdims=True)
#     dF_F_interpolated = (C_interpolated - F0_interpolated) / F0_interpolated
#     print("ΔF/F interpolated shape:", dF_F_interpolated.shape)
    
#     # Create column names for calcium and ΔF/F data
#     calcium_cols = [f'calcium_roi{i}' for i in range(num_neuron)]
#     dF_F_cols = [f'dF_F_roi{i}' for i in range(num_neuron)]
    
#     # Transpose arrays to match the DataFrame format (frames x neurons)
#     C_interpolated_transposed = C_interpolated.T
#     dF_F_interpolated_transposed = dF_F_interpolated.T
    
#     # Create DataFrames for the calcium and ΔF/F signals
#     df_ca = pd.DataFrame(data=C_interpolated_transposed, index=mini_cam_timestamps, columns=calcium_cols)
#     df_dF_F = pd.DataFrame(data=dF_F_interpolated_transposed, index=mini_cam_timestamps, columns=dF_F_cols)
    
#     df_ca.index.name = 'timestamp_ms'
#     df_dF_F.index.name = 'timestamp_ms'
    
#     # Merge the new data with the existing DataFrame
#     df_merged = df.join(df_ca, how='inner')
#     df_merged_with_dF_F = df_merged.join(df_dF_F, how='inner')
#     print("DataFrame with Ca and ΔF/F signals merged:")
    
#     # Save the updated DataFrame to a new HDF5 file
#     updated_hdf5_path = os.path.join(rec_path, 'MIR_Aligned', updated_hdf5_filename)
#     df_merged_with_dF_F.to_hdf(updated_hdf5_path, key='df', mode='w')
#     print(f"Updated DataFrame with Ca and ΔF/F data saved to {updated_hdf5_path}")

    
#     csv_output_path = os.path.join(
#         rec_path, 'MIR_Aligned',
#         f'aligned_predictions_with_ca_and_dF_F_{nc_key}.csv'
#     )
#     df_merged_with_dF_F.to_csv(csv_output_path, index=True)
#     print(f"CSV saved to {csv_output_path}")

    
#     return df_merged_with_dF_F

#below works well, though, not with only_com
# def load_aligneddannce_and_process_ca_data_custom_key(rec_path, mini_path, nc_key):
#     """
#     Loads MIR_Aligned/aligned_predictions.h5, finds the correct .nc file
#     (falling back to 'minian_dataset.nc' if needed), interpolates Ca,
#     computes ΔF/F, and writes out HDF5 and CSV without embedding 'None'.
#     """
#     # 1) load existing aligned preds
#     hdf5_in = os.path.join(rec_path, 'MIR_Aligned', "aligned_predictions.h5")
#     df = pd.read_hdf(hdf5_in, key='df')
#     timestamps_ms = df.index.values

#     # 2) locate the .nc file
#     miniscope_path = os.path.join(mini_path, 'My_V4_Miniscope')
#     # choose filename
#     if nc_key:
#         candidate = f"minian_dataset_{nc_key}.nc"
#     else:
#         candidate = "minian_dataset.nc"
#     ca_file_path = os.path.join(miniscope_path, candidate)

#     # fallback if candidate missing
#     if not os.path.exists(ca_file_path):
#         fb = os.path.join(miniscope_path, "minian_dataset.nc")
#         if os.path.exists(fb):
#             ca_file_path = fb
#         else:
#             raise FileNotFoundError(
#                 f"Tried '{candidate}' and fallback 'minian_dataset.nc', found neither at {miniscope_path}"
#             )
#     print(f"Using CA file: {ca_file_path}")

#     # 3) open and interpolate
#     data = xr.open_dataset(ca_file_path)
#     ts_ca = pd.read_csv(os.path.join(miniscope_path, 'timeStamps.csv'))['Time Stamp (ms)'].values
#     C = data['C'].values  # shape (n_neurons, n_ca_frames)
#     n_neurons = C.shape[0]

#     C_interp = np.stack([
#         interp1d(ts_ca, C[i], kind='linear', bounds_error=False, fill_value='extrapolate')(timestamps_ms)
#         for i in range(n_neurons)
#     ], axis=0)  # (n_neurons, n_miniframe)

#     F0 = np.percentile(C_interp, 20, axis=1, keepdims=True)
#     dF_F = (C_interp - F0) / F0

#     # 4) build DataFrames
#     calcium_cols = [f'calcium_roi{i}' for i in range(n_neurons)]
#     dF_F_cols     = [f'dF_F_roi{i}'    for i in range(n_neurons)]

#     df_ca  = pd.DataFrame(C_interp.T, index=timestamps_ms, columns=calcium_cols)
#     df_dff = pd.DataFrame(dF_F.T,  index=timestamps_ms, columns=dF_F_cols)

#     merged = df.join(df_ca, how='inner').join(df_dff, how='inner')

#     # 5) save
#     suffix = f"_{nc_key}" if nc_key else ""
#     out_h5 = os.path.join(
#         rec_path, 'MIR_Aligned',
#         f"aligned_predictions_with_ca_and_dF_F{suffix}.h5"
#     )
#     merged.to_hdf(out_h5, key='df', mode='w')
#     print(f"Saved HDF5: {out_h5}")

#     out_csv = os.path.join(
#         rec_path, 'MIR_Aligned',
#         f"aligned_predictions_with_ca_and_dF_F{suffix}.csv"
#     )
#     merged.to_csv(out_csv, index=True)
#     print(f"Saved CSV: {out_csv}")

#     return merged
def load_aligneddannce_and_process_ca_data_custom_key(rec_path, mini_path, nc_key=None, only_com=False):
    """
    ... same docstring ...
    """
    # 1) load existing aligned preds
    prefix = 'only_com_' if only_com else ''
    input_file = f"{prefix}aligned_predictions.h5"
    hdf5_in = os.path.join(rec_path, 'MIR_Aligned', input_file)
    df = pd.read_hdf(hdf5_in, key='df')
    timestamps_ms = df.index.values

    # 2) locate the .nc file
    miniscope_path = os.path.join(mini_path, 'My_V4_Miniscope')
    used_key = nc_key  # track whether we actually end up using this key

    if nc_key:
        candidate = f"minian_dataset_{nc_key}.nc"
    else:
        candidate = "minian_dataset.nc"
    ca_file_path = os.path.join(miniscope_path, candidate)
    # print(f"using CA file {ca_file_path}")

    # fallback if candidate missing
    if not os.path.exists(ca_file_path):
        fb = os.path.join(miniscope_path, "minian_dataset.nc")
        if os.path.exists(fb):
            ca_file_path = fb
            used_key = None  # clear, since we fell back to the default
        else:
            raise FileNotFoundError(
                f"Tried '{candidate}' and fallback 'minian_dataset.nc', found neither at {miniscope_path}"
            )
    print(f"Using CA file: {ca_file_path}")

    # 3) open and interpolate
    data = xr.open_dataset(ca_file_path)
    ts_ca = pd.read_csv(os.path.join(miniscope_path, 'timeStamps.csv'))['Time Stamp (ms)'].values
    C = data['C'].values
    n_neurons = C.shape[0]

    C_interp = np.stack([
        interp1d(ts_ca, C[i], kind='linear', bounds_error=False, fill_value='extrapolate')(timestamps_ms)
        for i in range(n_neurons)
    ], axis=0)

    F0 = np.percentile(C_interp, 20, axis=1, keepdims=True)
    dF_F = (C_interp - F0) / F0

    # 4) build DataFrames
    calcium_cols = [f'calcium_roi{i}' for i in range(n_neurons)]
    dF_F_cols   = [f'dF_F_roi{i}'    for i in range(n_neurons)]

    df_ca  = pd.DataFrame(C_interp.T, index=timestamps_ms, columns=calcium_cols)
    df_dff = pd.DataFrame(dF_F.T,      index=timestamps_ms, columns=dF_F_cols)

    merged = df.join(df_ca, how='inner').join(df_dff, how='inner')

    # 5) save outputs with proper prefixes and suffixes
    suffix = f"_{used_key}" if used_key else ""
    out_base = f"{prefix}aligned_predictions_with_ca_and_dF_F{suffix}"

    out_h5 = os.path.join(rec_path, 'MIR_Aligned', out_base + '.h5')
    merged.to_hdf(out_h5, key='df', mode='w')
    print(f"Saved HDF5: {out_h5}")

    out_csv = os.path.join(rec_path, 'MIR_Aligned', out_base + '.csv')
    merged.to_csv(out_csv, index=True)
    print(f"Saved CSV: {out_csv}")

    return merged


def run_mini_dannce_sync(
    rec_path,
    mini_path,
    nc_key,
    only_com=False,
    start_frame=0,
    end_frame=500,
    threshold_mini=15,
    threshold_sixcam=3,
    
):
    try:
        resultsss = sync_videos(
            rec_path,
            mini_path,
            start_frame,
            end_frame,
            threshold_mini,
            threshold_sixcam
        )
    except Exception as e:
        print(f"Error while syncing videos: {e}")
        return

    try:
        align_miniscope_to_sixcam(resultsss, mini_path, rec_path, only_com)
    except Exception as e:
        print(f"Error while aligning miniscope to sixcam: {e}")
        return

    try:
        load_aligneddannce_and_process_ca_data_custom_key(rec_path, mini_path, nc_key,only_com)
    except Exception as e:
        print(f"Error while loading aligned data and processing CA data: {e}")
        return


# if just producing the json.mapping.... 
def generate_frame_mapping(
    rec_path,
    mini_path,
    nc_key,
    only_com = False,   
    start_frame=0,
    end_frame=500,
    threshold_mini=15,
    threshold_sixcam=3,
    
):
    """
    Only run the parts up to align_miniscope_to_sixcam so that
    frame_mapping.json is created in MIR_Aligned/.
    """
    # 1) sync videos and get the sync result dict
    try:
        result = sync_videos(
            rec_path,
            mini_path,
            start_frame,
            end_frame,
            threshold_mini,
            threshold_sixcam
        )
    except Exception as e:
        print(f"[generate_frame_mapping] sync_videos error: {e}")
        return

    # 2) align miniscope to sixcam, which writes frame_mapping.json
    try:
        align_miniscope_to_sixcam(result, mini_path, rec_path, only_com)
        print("produced final h5 files")
    except Exception as e:
        print(f"[generate_frame_mapping] align_miniscope_to_sixcam error: {e}")
        return
    
    try:
        load_aligneddannce_and_process_ca_data_custom_key(rec_path, mini_path, nc_key,only_com)
    except Exception as e:
        print(f"Error while loading aligned data and processing CA data: {e}")
        return