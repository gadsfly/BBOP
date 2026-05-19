import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import scipy.io
from utlis.com_traga_utlis import find_calib_file

#updated below function so that it will only take in the first 3 min for calculations...
def calculate_frame_brightness(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    frame_brightness = []
    frame_number = 0
    max_frames = 100
    
    while frame_number < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate the average brightness
        avg_brightness = np.mean(gray_frame)
        frame_brightness.append(avg_brightness)
        
        frame_number += 1
    
    cap.release()
    return frame_brightness

def find_brightness_drop(brightness_values, threshold):
    drops = []
    for i in range(1, len(brightness_values)): 
        if brightness_values[i-1] - brightness_values[i] > threshold:
            drops.append(i)
    return drops

def process_videos(base_path, cameras, threshold):
    drop_frames = {}
    for camera in cameras:
        video_path = os.path.join(base_path, camera, '0.mp4')
        brightness_values = calculate_frame_brightness(video_path)
        
        drop_frame = find_brightness_drop(brightness_values, threshold)
        if drop_frame is not None:
            drop_frames[camera] = drop_frame
        else:
            print(f"No significant drop found in first 3 min in {video_path}")

        plt.plot(brightness_values, label=camera)
    
    plt.title('Frame Brightness Over Time')
    plt.xlabel('Frame Number, first 3 min')
    plt.ylabel('Average Brightness')
    plt.legend()
    plt.show()

    return drop_frames


def find_min_frame(dtf):
    # if 
    min_frame = min([frame[0] for frame in dtf.values()])
    return min_frame

def align_frames(calib_file, light_change_frames, save_path):
    """
    Aligns camera frames based on the given light change frames.
    
    Parameters:
    - light_change_frames: A dictionary where keys are camera names and values are the frames where light change is detected.
    - camera_data_frames: A dictionary where keys are camera names and values are the corresponding data frames.
    
    Returns:
    - A dictionary with the adjusted data frames.
    """
    calib_data = scipy.io.loadmat(calib_file)
    sync = calib_data['sync']
    


    camera_keys = list(light_change_frames.keys())
    min_frame = find_min_frame(light_change_frames)
    # print(min_frame)

    adjusted_data_frames = {}
    for cam_key in camera_keys:
        cam_idx = camera_keys.index(cam_key)
        keyyyyy = 'data_frame'  # Assuming this is the key for data frames in your structure
        data_frame = sync[cam_idx][0][keyyyyy][0][0][0]
        # print(data_frame)
        # print(sync[cam_idx][0][keyyyyy])
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data_frame)
        
        # Adjust frames
        frames_to_remove = light_change_frames[cam_key][0] - min_frame
        if frames_to_remove > 0:
            df = df.iloc[frames_to_remove:].reset_index(drop=True)
        
        # print(df.to_numpy().flatten())
        # adjusted_data_frames[cam_key] = df
        # sync[cam_idx][0][keyyyyy][0][0][0] = None
        sync[cam_idx][0][keyyyyy][0][0] = [df.to_numpy().flatten()]
        # print(sync[cam_idx][0][keyyyyy])
    # return sync[cam_idx][0][keyyyyy][0][0]
    calib_data['sync'] = sync
    scipy.io.savemat(save_path, calib_data)
    print('alined data saved to:', save_path)    
    
    # return adjusted_data_frames

def df_sync(base_path, threshold = 3):
    vi_path = os.path.join(base_path, 'videos')
    calib_file = find_calib_file(base_path)
    save_name = os.path.basename(calib_file)
    cameras = [f'Camera{i}' for i in range(1, 7)]
    save_path = os.path.join(base_path, save_name)
    
    drop_frames = process_videos(vi_path, cameras, threshold)
    
    # adjusted_data_frames = 
    align_frames(calib_file, drop_frames, save_path)