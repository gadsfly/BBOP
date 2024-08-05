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
    new_sync = copy.deepcopy(sync)

    dff = 'data_frame'
    sid = 'data_sampleID'

    for camera in range(6):
        old_data_sampleID = sync[camera][0][sid][0][0][0]

        # # change all sampleID to actually one, note that this starts with 1, instead of 0. the old one started with 0
        lenn = len(old_data_sampleID)
        new_data_sampleID = list(range(1,lenn+1))
        new_sync[camera][0][sid][0][0][0] = new_data_sampleID

        # now, if sampleID starts with nan, then we need to adjust the new data_frame for it. if it is not nan, then it stays the same
        if old_data_sampleID[0]==0:
            # print(f'camera {camera+1} does not have nan, remains unchanged')
            continue
        else:
            nan_counts = count_leading_nans(old_data_sampleID)
            # print(f'camera {camera+1} has nan count of {nan_counts}, now starts with {nan_counts+1}')
            new_data_frame = list(range(nan_counts+1,lenn+1))
            new_sync[camera][0][dff][0][0] = np.array(new_data_frame)
    
    base_path = os.path.dirname(old_calib_path)
    save_path = os.path.join(base_path, new_calib_name)

    old_calib_data['sync'] = new_sync
    sio.savemat(save_path, old_calib_data)
    print('converted sync data saved to:', save_path) 



def replace_elements(A, B, C):
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
            valid_label_sid = [x + 1 for x in good_label_sid]

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
