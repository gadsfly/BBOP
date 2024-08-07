import numpy as np
import scipy.io as sio
import os
import copy
import shutil

from utlis.com_traga_utlis import find_calib_file

def count_leading_nans(arr):
    count = 0
    for val in arr:
        if np.isnan(val):
            count += 1
        else:
            break
    return count

def convert_sync(old_calib_path, new_calib_name):
    old_calib_data = sio.loadmat(old_calib_path)
    sync = old_calib_data['sync']
    ew_sync = copy.deepcopy(sync)
    new_sync = copy.deepcopy(sync)

    dff = 'data_frame'
    sid = 'data_sampleID'

    min_length = len(sync[0][0][sid][0][0][0])

    for camera in range(6):
        old_data_sampleID = sync[camera][0][sid][0][0][0]

        # # change all sampleID to actually one, note that this starts with 1, instead of 0. the old one started with 0
        lenn = len(old_data_sampleID)
        
        new_data_sampleID = np.arange(1, lenn + 1, dtype=np.float64) # list(range(1,lenn+1))
        ew_sync[camera][0][sid][0][0] = new_data_sampleID

        # now, if sampleID starts with nan, then we need to adjust the new data_frame for it. if it is not nan, then it stays the same
        if np.isnan(old_data_sampleID[0]): #old_data_sampleID[0]==0:
            # print(f'camera {camera+1} does not have nan, remains unchanged')
            nan_counts = count_leading_nans(old_data_sampleID)
            # print(f'camera {camera+1} has nan count of {nan_counts}, now starts with {nan_counts+1}')
            new_data_frame = np.arange(nan_counts + 1, lenn + 1, dtype=np.float64) # list(range(nan_counts+1,lenn+1))
            ew_sync[camera][0][dff][0][0] = new_data_frame # np.array(new_data_frame, dtype=np.float64)
            min_length = min(min_length, len(new_data_frame))
        else:
            ew_sync[camera][0][dff][0][0] = sync[camera][0][dff][0][0][0]
            min_length = min(min_length, lenn)
        
        # print(min_length)
    
    # print("final", min_length)


    # Cut the end of other data frames and sample IDs to match the minimum length
    for camera in range(6):
        # print("0", len(ew_sync[camera][0][dff][0][0][:min_length]))
        new_sync[camera][0][dff][0][0] = ew_sync[camera][0][dff][0][0][:min_length]
        # print("1", len(ew_sync[camera][0][dff][0][0][:min_length]))
        new_sync[camera][0][sid][0][0] = ew_sync[camera][0][sid][0][0][:min_length]
        # print("2", len(ew_sync[camera][0][sid][0][0][:min_length]))
    
    base_path = os.path.dirname(old_calib_path)
    save_path = os.path.join(base_path, new_calib_name)

    old_calib_data['sync'] = new_sync
    sio.savemat(save_path, old_calib_data)
    print('converted sync data saved to:', save_path) 



def replace_elements(A, B, C):
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    C = C.astype(np.float64)
    #replace element of array in C, from A to B. but this method would not handle any nan values in the label....
    # Flatten the input array C to make indexing straightforward
    C_flat = C.flatten()
    
    # Create a dictionary to map values in A to their indices
    value_to_index = {value: idx for idx, value in enumerate(A)}
    
    # Iterate over the flattened array and replace values
    for i in range(len(C_flat)):
        if C_flat[i] in value_to_index:
            index_in_A = value_to_index[C_flat[i]]
            C_flat[i] = B[index_in_A]
    
    # Reshape the flattened array back to the original shape of C
    C = C_flat.reshape(C.shape)
    
    return C


def convert_label(old_calib_path, new_calib_path, label_path, new_label_name):
    old_calib_data = sio.loadmat(old_calib_path)
    new_calib_data = sio.loadmat(new_calib_path)
    label_data = sio.loadmat(label_path)

    dff = 'data_frame'
    sid = 'data_sampleID'

    labels = label_data['labelData']
    new_sync = new_calib_data['sync']
    old_sync = old_calib_data['sync']

    for camera_idx in range(6):
        old_sync_sid = old_sync[camera_idx][0][sid][0][0][0]

        if old_sync_sid[0]==0:
            good_label_sid = labels[camera_idx][0][sid][0][0]
            valid_label_sid = np.array([x + 1 for x in good_label_sid], dtype=np.float64)

    for camera_idx in range(6):
        old_sync_sid = old_sync[camera_idx][0][sid][0][0][0]
        new_label_sid = valid_label_sid
        
        if old_sync_sid[0]==0:
            labels[camera_idx][0][sid][0][0] = new_label_sid
            continue
        # then convert all incidences in the data 
        label_dff = labels[camera_idx][0][dff][0][0]
        # label_sid = labels[camera_idx][0][sid][0][0]

        old_sync_dff = old_sync[camera_idx][0][dff][0][0][0]
        

        new_sync_dff = new_sync[camera_idx][0][dff][0][0][0]
        # new_sync_sid = new_sync[camera_idx][0][sid][0][0][0]

        new_label_dff = replace_elements(old_sync_dff, new_sync_dff, label_dff)
        # replace_elements(old_sync_sid, new_sync_sid, label_sid)

        labels[camera_idx][0][dff][0][0] = new_label_dff
        labels[camera_idx][0][sid][0][0] = new_label_sid




    # step one: replace sync:
    label_data['sync'] = new_calib_data['sync']
    label_data['labelData'] = labels


    base_path = os.path.dirname(label_path)
    save_path = os.path.join(base_path, new_label_name)

    sio.savemat(save_path, label_data)
    print('converted label data saved to:', save_path) 


def process_calib_and_labels(data_sets):
    """
    Processes multiple sets of data by converting sync and labels.

    Parameters:
    data_sets (list of dict): Each dict should contain the paths for 'old_calib_path', 'new_calib_name', 'new_calib_path', 'label_path', and 'new_label_name'.
    """
    for data in data_sets:
        base_path = data['base_path']
        old_calib_path = find_calib_file(base_path)
        prev_calib_folder = os.path.join(base_path, 'prev_calib')
        if not os.path.exists(prev_calib_folder):
                os.makedirs(prev_calib_folder)
        if old_calib_path is None:
            raise FileNotFoundError(f'Calibration file not found in the {base_path}.')

        # Check if calib starts with '0_' then find pos_ in prev_calib
        if os.path.basename(old_calib_path).startswith('0_'):
            raise FileNotFoundError(f'Calibration file not found in the {base_path}.')
        if os.path.basename(old_calib_path).startswith('df_'):
            raise FileNotFoundError(f'Calibration file not found in the {base_path}.')
        #     old_calib_path = find_calib_file(prev_calib_folder)
        #     if old_calib_path is None:
        #         raise FileNotFoundError(f'Calibration file not found in the {base_path}/prev_calib.')   

        old_calib_name = os.path.basename(old_calib_path)
        new_calib_name = 'df_converted_' + old_calib_name
        new_calib_path = os.path.join(base_path, new_calib_name)

        label_path = data['label_path']

        old_label_name = os.path.basename(label_path)
        new_label_name = 'df_converted_' + old_label_name

        # Convert sync data
        convert_sync(old_calib_path, new_calib_name)
        
        # Convert label data
        convert_label(old_calib_path, new_calib_path, label_path, new_label_name)

        # # Move the previous calibration file to 'prev_calib' folder
        shutil.move(old_calib_path, os.path.join(prev_calib_folder, old_calib_name))
        print(f'removed prior calib files to {prev_calib_folder}')

        
def process_calibs(data_sets):
    """
    Processes multiple sets of data by converting sync and labels.

    Parameters:
    data_sets (list of dict): Each dict should contain the paths for 'old_calib_path', 'new_calib_name', 'new_calib_path', 'label_path', and 'new_label_name'.
    """
    for data in data_sets:
        base_path = data['base_path']
        old_calib_path = find_calib_file(base_path)
        prev_calib_folder = os.path.join(base_path, 'prev_calib')
        if not os.path.exists(prev_calib_folder):
                os.makedirs(prev_calib_folder)
        if old_calib_path is None:
            raise FileNotFoundError(f'Calibration file not found in the {base_path}.')

        # Check if calib starts with '0_' then find pos_ in prev_calib
        if os.path.basename(old_calib_path).startswith('0_'):
            raise FileNotFoundError(f'Calibration file not found in the {base_path}.')
        if os.path.basename(old_calib_path).startswith('df_'):
            raise FileNotFoundError(f'Calibration file not found in the {base_path}.')
        #     old_calib_path = find_calib_file(prev_calib_folder)
        #     if old_calib_path is None:
        #         raise FileNotFoundError(f'Calibration file not found in the {base_path}/prev_calib.')   

        old_calib_name = os.path.basename(old_calib_path)
        new_calib_name = 'df_converted_' + old_calib_name
        new_calib_path = os.path.join(base_path, new_calib_name)

        # label_path = data['label_path']

        # old_label_name = os.path.basename(label_path)
        # new_label_name = 'df_converted_' + old_label_name

        # Convert sync data
        convert_sync(old_calib_path, new_calib_name)
        
        # Convert label data
        # convert_label(old_calib_path, new_calib_path, label_path, new_label_name)

        # # Move the previous calibration file to 'prev_calib' folder
        shutil.move(old_calib_path, os.path.join(prev_calib_folder, old_calib_name))
        print(f'removed prior calib files to {prev_calib_folder}')
