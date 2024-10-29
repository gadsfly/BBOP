import scipy.io as sio
import numpy as np
import os
import cv2
import glob
import re
#you can just `python G:\newsdannce_general_mir\src\calibration\new\mir_generate_param.py`
#you only need to change the first three paths and names
# now it is saved for individual video since the video i may stop recording early...
# Define the directory where the files are stored

base_path = '/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_09_18' # where calib is saved to, and where your actual recorded mice video locates.  2024_09_16_test_chris_new
calib_path = os.path.join(base_path, 'calib_before_13_17') #from chris, with 
vid_path = '20240819-V1-r2'




calibname = os.path.basename(calib_path)
output_file = f'ssh_try_{calibname}_label3d_dannce.mat'


# calib_path = 'G:\\Videos\\6cam\\akg61\\2024_06_28\\calib_before_8'
# base_path = 'G:\\Videos\\6cam\\akg61\\2024_06_28\\1686941_left_right_2'
# output_file = '2024_06_28_1686941_left_label3d_dannce.mat'


# Prepare a dictionary to hold all the data
label3d_dannce = {'camnames': [], 'params': [], 'sync': []}

# List all .mat files in the directory
input_files = glob.glob(os.path.join(calib_path, '*.mat')) #, 'out/' 'pyxy3d_noT/'
# Sort the files based on camera number
# This assumes filenames contain "cam" followed by the camera number, like "cam1", "cam2", etc.
input_files.sort(key=lambda x: int(re.search(r'cam(\d+)', x).group(1)) if re.search(r'cam(\d+)', x) else float('inf'))
print(input_files)
print(os.path.join(calib_path, '*.mat'))
print(f"Found {len(input_files)} files.")
if len(input_files) > 6:
    print("Found not only 6 camera parameter files, but more. Go and move the extraneous files elsewhere.")

# video_path = 'G:/Videos/6cam/jn187/2024_04_29_chris_validationfor27_mir/videos/Camera1/0.mp4'
# Prepare the sync dictionary by accessing a video file
video_path = os.path.join(base_path, vid_path, "videos/",  "Camera2/", "0.mp4") #extrinsic
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video file.")
else:
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Frame count:", frame_count)
    cap.release()

    sync = {
        "data_2d": np.zeros((frame_count, 44)),
        "data_3d": np.zeros((frame_count, 66)),
        "data_frame": np.arange(1, frame_count + 1, dtype=np.float64),
        "data_sampleID": np.arange(1, frame_count + 1, dtype=np.float64)
    }

    # Process each input file
    for file_path in input_files:
        file_name = os.path.basename(file_path)
        data = sio.loadmat(file_path)
        print(f'processing {file_name}')
        # Prepare the parameters dictionary
        camera_params = {
            'K': data['K'],
            'RDistort': data['RDistort'],
            'TDistort': data['TDistort'],
            'r': data['r'],
            't': data['t']
        }

        # Extract camera name from file name
        cam_name = f"Camera{file_name.split('_')[1].lstrip('cam')}"
        print(f'processing cam {cam_name}')

        # Append the camera name and parameters to the lists
        label3d_dannce['camnames'].append(cam_name)
        label3d_dannce['params'].append(camera_params)
        label3d_dannce['sync'].append(sync)
        print(f'generated from {file_path}')

    # Ensure the correct format for camnames, params, and sync
    label3d_dannce['camnames'] = np.array([label3d_dannce['camnames']], dtype='O')
    label3d_dannce['params'] = np.array(label3d_dannce['params'], dtype='O').reshape(-1, 1)
    label3d_dannce['sync'] = np.array(label3d_dannce['sync'], dtype='O').reshape(-1, 1)

    # Save the converted data to a new .mat file
    output_full_path = os.path.join(base_path, vid_path, output_file) #'out/', pyxy3d_noT
    # output_full_path = os.path.join('G:/Videos/6cam/jn187/2024_04_29_chris_validationfor27_mir', output_file)
    sio.savemat(output_full_path, label3d_dannce)
    print(f"Data saved to {output_full_path}")


