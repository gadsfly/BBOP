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
