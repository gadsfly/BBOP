import os
import glob
import shutil
from utlis.sync_utlis.sync_df_utlis import process_sync


def rerun_with_prev_calib(base_folder, threshold=2, max_frames=600, min_frame=300):
    # 1. delete the current calib file
    curr_pattern = os.path.join(base_folder, "df*label3d_dannce.mat")
    curr_files = glob.glob(curr_pattern)
    if curr_files:
        os.remove(curr_files[0])
        print(f"removed{curr_files[0]}")

    # 2. move the backup from prev_calib
    prev_dir = os.path.join(base_folder, "prev_calib")
    prev_pattern = os.path.join(prev_dir, "*label3d_dannce.mat")
    prev_files = glob.glob(prev_pattern)
    if prev_files:
        shutil.move(prev_files[0], base_folder)

    # # 3. remove prev_calib if now empty
    # if os.path.isdir(prev_dir) and not os.listdir(prev_dir):
    #     os.rmdir(prev_dir)

    # 4. rerun process_sync
    process_sync(base_folder, threshold=threshold, max_frames=max_frames, min_frame=min_frame)