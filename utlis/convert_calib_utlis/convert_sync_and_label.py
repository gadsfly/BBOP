import numpy as np
import scipy.io as sio
import os, re
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
        # new_calib_path = os.path.join(base_path, new_calib_name)

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


def categorize_files(directory):
    df_files = []
    offset_files = []

    for f in os.listdir(directory):
        full_path = os.path.join(directory, f)
        if f.startswith('df_'):
            df_files.append(full_path)
        elif 'offset' in f and f.startswith('pos_'):
            offset_files.append(full_path)

    return df_files, offset_files

def process_calibs_auto(summ_path):
    """
    Processes multiple sets of data by converting sync and labels.

    Parameters:
    data_sets (list of dict): Each dict should contain the paths for 'old_calib_path', 'new_calib_name', 'new_calib_path', 'label_path', and 'new_label_name'.
    """
    matching_folders = [f for f in os.listdir(summ_path) if os.path.isdir(os.path.join(summ_path, f)) and re.match(r'^\d{4}_\d{2}_\d{2}$', f)]
    if matching_folders == []:
        print('No qualified folders found')
        return
    # print(1)
    print(matching_folders)
    processed_folders_path = os.path.join(summ_path, 'processed_folders.txt')
    reprocess_folders_path = os.path.join(summ_path, 'reprocess_folders.txt')

    processed_folders = None
    if os.path.exists(processed_folders_path):
        with open(processed_folders_path, 'r') as f:
            processed_folders = set(f.read().splitlines())

    for date in matching_folders:
        # print(2)
        date_path = os.path.join(summ_path, date)
        # print(3)
        if not os.path.exists(date_path):
            print(f"Date folder {date_path} does not exist. Skipping.")
            continue
        for folder_name in os.listdir(date_path):
            folder_path = os.path.join(date_path, folder_name)

            if processed_folders is not None:
                if folder_path in processed_folders:
                    print(f"Skipping already processed folder: {folder_path}")
                    continue



            if os.path.isdir(folder_path) and folder_name[0].isdigit():
                base_path = folder_path

                old_calib_path = find_calib_file(base_path)
                prev_calib_folder = os.path.join(base_path, 'prev_calib')
                if not os.path.exists(prev_calib_folder):
                        os.makedirs(prev_calib_folder)
                if old_calib_path is None:
                    print(f'Calibration file not found in the {base_path}. Skipping conversion.')
                    continue
                    # raise FileNotFoundError(f'Calibration file not found in the {base_path}.')
                    
                if os.path.basename(old_calib_path).startswith('df_'):
                    # raise FileNotFoundError(f'Calibration file not found in the {base_path}.')
                    print(f'Calibration already converted {old_calib_path}. Skipping conversion.')
                    continue
                #     old_calib_path = find_calib_file(prev_calib_folder)
                #     if old_calib_path is None:
                #         raise FileNotFoundError(f'Calibration file not found in the {base_path}/prev_calib.')   
                
                old_calib_name = os.path.basename(old_calib_path)

                # Check if calib starts with '0_' then find pos_ in prev_calib
                if os.path.basename(old_calib_path).startswith('0_'):
                    if len(os.listdir(prev_calib_folder)) != 1:
                        df_files2, offset_files = categorize_files(prev_calib_folder)
                        if df_files2 != []:
                        # if os.path.basename(prev_old_calib_path).startswith('df_'):
                            df_path = df_files2[0]
                            prev_old_calib_name = os.path.basename(df_path) # os.path.basename(prev_old_calib_path)
                            shutil.move(df_path, os.path.join(base_path, prev_old_calib_name))
                            print(f'removed df calib file {df_path} to {base_path}')
                            shutil.move(old_calib_path, os.path.join(prev_calib_folder, old_calib_name))
                            print(f'removed prior calib files {old_calib_path} to {prev_calib_folder}')
                        elif offset_files != []:
                            offset_files_path = offset_files[0]
                            prev_old_calib_name = os.path.basename(offset_files_path) # os.path.basename(prev_old_calib_path)
                            shutil.move(offset_files_path, os.path.join(base_path, prev_old_calib_name))
                            print(f'moved pos file to {base_path}')
                            shutil.move(old_calib_path, os.path.join(prev_calib_folder, old_calib_name))
                            print(f'removed prior calib files {old_calib_path} to {prev_calib_folder}')
                            # print("I'm too lazy to code so you go and move the pos_ for another processing")

                        else:
                        # print(f"Warning: Expected exactly one calibration file in {prev_calib_folder}, but found {len(os.listdir(prev_calib_folder))}. Skipping conversion.")
                            print(f"not able to find only one pos, or df. skipping {prev_calib_folder}")
                            continue
                        # raise FileNotFoundError(f'Calibration file not found in the {base_path}.')
                    else: 
                        prev_old_calib_path = find_calib_file(prev_calib_folder)
                        if prev_old_calib_path is None:
                            print(f"No calibration file found in {prev_calib_folder}. Skipping conversion.")
                            continue

                        if os.path.basename(prev_old_calib_path).startswith('pos_'):
                            prev_old_calib_name = os.path.basename(prev_old_calib_path)
                            new_calib_name = 'df_converted_' + prev_old_calib_name

                            convert_sync(prev_old_calib_path, new_calib_name)
                            print("you will have to mannuly move 0_ and df_ file now hh")
                            # not that this operation did not move df files, which will cause issues
                            # # Move the previous calibration file to 'prev_calib' folder
                            shutil.move(old_calib_path, os.path.join(prev_calib_folder, prev_old_calib_name))
                            print(f'removed prior calib files {old_calib_path} to {prev_calib_folder}')
                        else:
                            print(f'no pos calib find in pre_calib folder in {prev_calib_folder}, skipping')
                            continue
                        
                if os.path.basename(old_calib_path).startswith('pos_'):
                    old_calib_name = os.path.basename(old_calib_path)
                
                    new_calib_name = 'df_converted_' + old_calib_name

                    convert_sync(old_calib_path, new_calib_name)
                    # # Move the previous calibration file to 'prev_calib' folder
                    shutil.move(old_calib_path, os.path.join(prev_calib_folder, old_calib_name))
                    print(f'removed prior calib files {old_calib_path} to {prev_calib_folder}')
                
                df_files = [f for f in os.listdir(base_path) if f.startswith('df_')]
                if len(df_files) == 0 or len(df_files) > 1:
                    print(f'need to reprocess, {base_path}')
                    with open(reprocess_folders_path, 'a') as f:
                        f.write(f"{base_path}\n")
                        print(f'Saved {base_path} to {reprocess_folders_path}')
                # if :
                #     print(f'special attention needed! potentially mannualy process and removal and process again, {base_path}')
                if len(df_files) == 1:
                    with open(processed_folders_path, 'a') as f:
                        f.write(f"{base_path}\n")
                        print(f'Saved {base_path} to {processed_folders_path}')


def check_processed_results(summ_path):

    matching_folders = [f for f in os.listdir(summ_path) if os.path.isdir(os.path.join(summ_path, f)) and re.match(r'^\d{4}_\d{2}_\d{2}$', f)]
    if matching_folders == []:
        print('No qualified folders found')
        return
    # print(1)
    print(matching_folders)
    processed_folders_path = os.path.join(summ_path, 'processed_folders.txt')
    reprocess_folders_path = os.path.join(summ_path, 'reprocess_folders.txt')

    processed_folders = None
    if os.path.exists(processed_folders_path):
        with open(processed_folders_path, 'r') as f:
            processed_folders = set(f.read().splitlines())

    for date in matching_folders:
        # print(2)
        date_path = os.path.join(summ_path, date)
        # print(3)
        if not os.path.exists(date_path):
            # print(f"Date folder {date_path} does not exist. Skipping.")
            continue
        for folder_name in os.listdir(date_path):
            folder_path = os.path.join(date_path, folder_name)

            if processed_folders is not None:
                if folder_path in processed_folders:
                    print(f"Skipping already processed folder: {folder_path}")
                    continue
            else:
                print(f'need to be processed: {folder_path}')



            # if os.path.isdir(folder_path) and folder_name[0].isdigit():

            #     df_files = [f for f in os.listdir(base_path) if f.startswith('df_')]
            #     if len(df_files) == 0 or len(df_files) > 1:
            #         print(f'need to reprocess, {base_path}')
            #         with open(reprocess_folders_path, 'a') as f:
            #             f.write(f"{base_path}\n")
            #             print(f'Saved {base_path} to {reprocess_folders_path}')
            #     # if :
            #     #     print(f'special attention needed! potentially mannualy process and removal and process again, {base_path}')
            #     if len(df_files) == 1:
            #         with open(processed_folders_path, 'a') as f:
            #             f.write(f"{base_path}\n")
            #             print(f'Saved {base_path} to {processed_folders_path}')