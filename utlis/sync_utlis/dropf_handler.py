import numpy as np
import sys
import os
sys.path.append(os.path.abspath('../..'))
from utlis.sync_utlis.sync_df_utlis import find_calib_file
import scipy.io as sio
import shutil

def load_frametimes(base_path, num_cameras=6):
    """
    Load frametimes.npy files for a specified number of cameras.

    Parameters:
    - base_path (str): The base path where the main folder is located.
    - num_cameras (int): The number of cameras to load frametimes for. Default is 6.

    Returns:
    - dict: A dictionary with camera names as keys and frametimes data as values.
    """
    frametimes = {}
    for i in range(1, num_cameras + 1):
        camera_path = f'{base_path}/videos/Camera{i}/frametimes.npy'
        try:
            frametimes[f'Camera{i}'] = np.load(camera_path)
            print(f"Loaded frametimes for Camera{i}")
        except FileNotFoundError:
            print(f"File not found: {camera_path}")
    return frametimes

def check_max_shapes_consistency(frametimes_data):
    max_shapes = {camera: data.shape[1] for camera, data in frametimes_data.items()}
    print("Maximum shapes for each camera:", max_shapes)
    
    # Get the set of unique shape values
    unique_shapes = set(max_shapes.values())
    
    # Check if all cameras have the same shape and find the maximum shape value
    consistency = len(unique_shapes) == 1
    max_shape_value = max(max_shapes.values())
    
    return consistency, max_shape_value


def create_standard_timeline(frametimes_data, max_shape_value, fps=30):
    # Extract the end times and frame numbers for each camera
    end_times = {camera: frametimes_data[camera][1][-1] for camera in frametimes_data.keys()}
    # max_frame_numbers_real = {camera: frametimes_data[camera][0][-1] for camera in frametimes_data.keys()}
    
    # Determine the maximum end time and frame number across all cameras
    # max_end_time = int(round(max(end_times.values())))
    max_end_time = (max(end_times.values()))
    # max_frame_number_real = max(max_frame_numbers.values())
    # if max_frame_number > max_shape_value:
    #     max_frame_number = max_shape_value
    max_frame_number = max_end_time * fps
    print(f"max frame number {max_frame_number}")
    
    # Calculate the interval between frames
    frame_interval = 1 / fps
    
    # Generate standard timeline and frame numbers
    standard_timeline = np.arange(0, max_end_time, frame_interval)

    standard_frame_numbers = np.arange(1, max_frame_number+1)
    
    return standard_frame_numbers, standard_timeline, max_frame_number

def align_to_standard_timeline(frametimes_data, standard_frame_numbers, standard_timeline, max_frame_numbers):
    aligned_data = {}
    for camera, data in frametimes_data.items():
        frame_numbers, timestamps = np.array(data[0]), np.array(data[1])

        # Use searchsorted to find indices where standard_timeline would be inserted in timestamps
        indices = np.searchsorted(timestamps, standard_timeline)

        # Handle edge cases for indices at the boundaries
        idx_left = np.clip(indices - 1, 0, len(timestamps) - 1)
        idx_right = np.clip(indices, 0, len(timestamps) - 1)

        # Get the timestamps and frame numbers at idx_left and idx_right
        time_left = timestamps[idx_left]
        time_right = timestamps[idx_right]
        frame_left = frame_numbers[idx_left]
        frame_right = frame_numbers[idx_right]

        # Compute absolute differences between the standard timeline and the left/right timestamps
        diff_left = np.abs(standard_timeline - time_left)
        diff_right = np.abs(time_right - standard_timeline)

        # Determine which timestamps are closer to the standard timeline
        use_left = diff_left <= diff_right
        closest_indices = np.where(use_left, idx_left, idx_right)

        # Align frames and timestamps based on the closest indices
        aligned_frames = frame_numbers[closest_indices]
        aligned_timestamps = timestamps[closest_indices]

        aligned_data[camera] = {
            "frames": aligned_frames,
            "timestamps": aligned_timestamps
        }
        print(f"Aligned data for {camera} with {len(standard_timeline)} entries.")

    return aligned_data


def update_data_frame(base_path, aligned_frametimes):
    calib_path = find_calib_file(base_path)
    calib_data = sio.loadmat(calib_path)
    sync = calib_data['sync']
    cameras = [f'Camera{i}' for i in range(1, 7)]
    keyyyyy = 'data_frame'
    updated_frames = {}
    

    for cam_idx, camera in enumerate(cameras):
        # data_frame = sync[cam_idx][0][keyyyyy][0][0][0]
        # print(data_frame)
        if camera in aligned_frametimes:
            # import pdb
            # pdb.set_trace()

            data_frame = sync[cam_idx][0][keyyyyy][0][0][0]
            # print(data_frame.shape)
            print(data_frame)
            aligned_frames = aligned_frametimes[camera]['frames']

            # Compute the offset based on the starting value of data_frame
            frame_start = data_frame[0]
            offset = frame_start - 1

            # Map aligned frames to the adjusted data_frame
            mapped_frames = aligned_frames + offset
            # print(mapped_frames.shape, mapped_frames.shape, data_frame.shape)
            # Handle NaN values: Replace NaNs with original data_frame values
            # updated_data_frame = np.where(~np.isnan(mapped_frames), mapped_frames, data_frame)

            # Store the updated data_frame
            updated_frames[camera] = mapped_frames
            sync[cam_idx][0][keyyyyy][0][0] = mapped_frames
        else:
            print(f"{camera} does not exist in aligned_frametimes")
        

    
    try:  

        calib_data['sync'] = sync
        calib_name = os.path.basename(calib_path)

        save_path = os.path.join(base_path, f'df_dh_{calib_name}')
        sio.savemat(save_path, calib_data)
        print('dropped_handled data saved to:', save_path)

        prev_calib_folder = os.path.join(base_path, 'prev_df_calib')
        os.makedirs(prev_calib_folder, exist_ok=True)
        shutil.move(calib_path, prev_calib_folder)
        print(f"Moved prior calibration file to {prev_calib_folder}")
        return True
        # return
        # time.sleep(1)

    except Exception as e:
        print(f"Error during alignment: {e}")
        return
        return False

    return updated_frames

# Main function to process camera data
def process_camera_data(base_path, frametimes_data, fps=30):
    calib_path = find_calib_file(base_path)
    if calib_path == '' or calib_path is None:
        print('no calib path found. generate calib path first, then sync.')
        return
    
    if not os.path.basename(calib_path).startswith("df"):
        print("sync first! because the person wrote the scripts kinda wacky so you have to follow this wacky steps that is not yet dynamic. very sorry.")
        return
    # Step 1: Check for consistency in frame shapes
    consistency, max_shape_value = check_max_shapes_consistency(frametimes_data)
    if consistency:
        # raise ValueError("Cameras have inconsistent shapes.")
        print("Cameras have consistent shapes. Skipping alignment.")
        return
    
    # Step 2: Create the standard timeline and frame numbers
    standard_frame_numbers, standard_timeline, max = create_standard_timeline(frametimes_data, max_shape_value)
    
    # Step 3: Align camera data to the standard timeline
    aligned_frametimes = align_to_standard_timeline(frametimes_data, standard_frame_numbers, standard_timeline, max_shape_value)
    
    # step 4, update & save
    updated_frames = update_data_frame(base_path, aligned_frametimes)
    print("All cameras aligned successfully.")
    print(updated_frames)
    # return aligned_frametimes, standard_frame_numbers, standard_timeline

def drop_frame_handler(base_path):
    frametimes_data = load_frametimes(base_path)
    process_camera_data(base_path, frametimes_data, fps=30)