import scipy.io as sio
import numpy as np
import os
import cv2
import glob
import re

def mir_generate_param_z(base_path, calib_path, vid_path, output_file):
    """Processes video frame count, calibration data, and saves to a .mat file in DANNCE format."""

    # Prepare the dictionary to hold all the data
    label3d_dannce = {'camnames': [], 'params': [], 'sync': []}

    # List all .mat files in the calibration directory
    input_files = glob.glob(os.path.join(calib_path, '*.mat'))
    input_files.sort(key=lambda x: int(re.search(r'cam(\d+)', x).group(1)) if re.search(r'cam(\d+)', x) else float('inf'))
    print(f"Found {len(input_files)} calibration files.")
    
    if len(input_files) > 6:
        print("Found more than 6 camera parameter files. Move the extra files elsewhere.")
        return False
    if len(input_files) == 0:
        print(f"no calib file found. skipping. calib path of {calib_path}")
        return False
    
    

    # Get video frame count for sync data
    video_path = os.path.join(base_path, vid_path, "videos/", "Camera2/", "0.mp4")
    cap = cv2.VideoCapture(video_path)

    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}. Skipping.")
        return False
    
    if not cap.isOpened():
        print("Error opening video file.")
        return False
    else:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Frame count:", frame_count)
        cap.release()

        # Generate sync data
        sync_data = {
            "data_2d": np.zeros((frame_count, 44)),
            "data_3d": np.zeros((frame_count, 66)),
            "data_frame": np.arange(1, frame_count + 1, dtype=np.float64),
            "data_sampleID": np.arange(1, frame_count + 1, dtype=np.float64)
        }




        # Process each input calibration file
        for file_path in input_files:
            file_name = os.path.basename(file_path)
            data = sio.loadmat(file_path)

            # Prepare camera parameters
            camera_params = {
                'K': data['K'], 
                'RDistort': data['RDistort'],
                'TDistort': data['TDistort'],
                'r': data['r'],
                't': data['t']
            }

            # Extract camera name
            cam_name = f"Camera{file_name.split('_')[1].lstrip('cam')}"

            # Append data to the dictionary
            label3d_dannce['camnames'].append(cam_name)
            label3d_dannce['params'].append(camera_params)
            label3d_dannce['sync'].append(sync_data)
            print(f'Processed {file_path}')

        # Ensure the correct format for camnames, params, and sync
        label3d_dannce['camnames'] = np.array([label3d_dannce['camnames']], dtype='O')
        label3d_dannce['params'] = np.array(label3d_dannce['params'], dtype='O').reshape(-1, 1)
        label3d_dannce['sync'] = np.array(label3d_dannce['sync'], dtype='O').reshape(-1, 1)
        # import pdb
        # pdb.set_trace()
        # Save the data to a .mat file
        output_full_path = os.path.join(base_path, vid_path, output_file)
        sio.savemat(output_full_path, label3d_dannce)
        print(f"Data saved to {output_full_path}")
        return True

