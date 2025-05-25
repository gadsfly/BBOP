import os
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import datetime
import sys
sys.path.append(os.path.abspath('../..'))
from utlis.exe_engine_utlis.mir_generate_param_modu import mir_generate_param_z
# from utlis.sync_utlis.sync_df_utlis import process_sync
from utlis.exe_engine_utlis.exe_single_utlis import rerun_with_prev_calib
from concurrent.futures import ThreadPoolExecutor

# Function to process each "unit" (rec_file) and update its status in the corresponding Parquet file
def process_unit_and_update_status_mirgenparam(rec_file_data, base_folder):
    date_folder = rec_file_data['date_folder']
    rec_file = rec_file_data['rec_file']
    
    # Generate the paths needed for processing
    combined_path = os.path.join(base_folder, date_folder)
    calib_path = rec_file_data['calib_path'] if 'calib_path' in rec_file_data else os.path.join(combined_path, 'calib_before') #calib_before*
    calib_path = os.path.join(base_folder, date_folder, calib_path)
    if not calib_path:  # Check for empty or None calib_path
        print(f'No calib folder found. Aborting. {combined_path}/{rec_file}')
        return
    
    output_file = f'{os.path.basename(date_folder)}_{rec_file}_{os.path.basename(calib_path)}_label3d_dannce.mat'

    # Call your processing function
    ssssta = mir_generate_param_z(combined_path, calib_path, rec_file, output_file)
    print("mir_generate_param ran successfully.")

    # After processing, update the status in the specific Parquet file
    parquet_file_path = os.path.join(base_folder, date_folder, rec_file, "folder_log.parquet")

    # Load the existing Parquet file
    try:
        table = pq.read_table(parquet_file_path)
        df = table.to_pandas()  # Convert to pandas for easier manipulation
    except FileNotFoundError:
        print(f"Parquet file not found at {parquet_file_path}")
        return

    if ssssta is True:
        # Update the status field (assuming 'sync' is the column)
        df['mir_generate_param'] = '1'  # Set status to '1' for processed

        # Add scan_time (or other updates)
        df['scan_time'] = datetime.datetime.now().isoformat()

        # Write the updated DataFrame back to the Parquet file
        updated_table = pa.Table.from_pandas(df)
        pq.write_table(updated_table, parquet_file_path)

        print(f"Updated Parquet file at {parquet_file_path} with new status.")
    else:
        print(f'processed failed. please check {combined_path}/{rec_file}')

# Function to handle sequential processing and status updates
def sequential_process_and_update_mirgenparam(filtered_table, base_folder):
    # Convert PyArrow table to pandas DataFrame
    filtered_df = filtered_table.to_pandas()

    # Process each row sequentially
    for row in filtered_df.itertuples(index=False):
        try:
            process_unit_and_update_status_mirgenparam(row._asdict(), base_folder)
        except Exception as e:
            print(f"Error in processing: {e}")













# Function to process each "unit" (rec_file) and update its status in the corresponding Parquet file sequentially
def process_unit_and_update_status_sync(rec_file_data, base_folder, threshold=2, max_frames=300, min_frame=0):
    date_folder = rec_file_data['date_folder']
    rec_file = rec_file_data['rec_file']
    
    # Generate the paths needed for processing
    combined_path = os.path.join(base_folder, date_folder, rec_file)
    calib_path = rec_file_data.get('calib_path', os.path.join(base_folder, 'calib_before'))
    
    if not calib_path:
        print(f'No calib folder found. Aborting. {combined_path}')
        return
    
    output_file = f'{os.path.basename(date_folder)}_{rec_file}_{os.path.basename(calib_path)}_label3d_dannce.mat'
    print(f"Processing: {combined_path}")
    
    # Call the sync processing function
    sync_status = rerun_with_prev_calib(combined_path, threshold, max_frames, min_frame)
    # sync_status = process_sync(combined_path, threshold, max_frames) #base_folder, threshold=3, max_frames=100, min_frame=0):
    if sync_status is True:
        print("Sync ran successfully.")
    else:
        print(f"please mannually process {combined_path}")

    # Update the status in the Parquet file
    parquet_file_path = os.path.join(base_folder, date_folder, rec_file, "folder_log.parquet")

    try:
        table = pq.read_table(parquet_file_path)
        df = table.to_pandas()
    except FileNotFoundError:
        print(f"Parquet file not found at {parquet_file_path}")
        return
    
    # status = '1' if sync_status else '3'
    status = '3' if not sync_status else '1'
    if status == '3':
        print(f"Failed processing: {combined_path}, status set to 3.")
    
    # Update status and scan_time
    df['sync'] = status
    df['scan_time'] = datetime.datetime.now().isoformat()

    # Write back to the Parquet file
    updated_table = pa.Table.from_pandas(df)
    pq.write_table(updated_table, parquet_file_path)
    print(f"Updated Parquet file at {parquet_file_path} with new status.")
   
# Sequentially process and update the status for each rec_file
def sequential_process_and_update_sync(filtered_table, base_folder, threshold=2, max_frames=300, min_frame=0):
    filtered_df = filtered_table.to_pandas()
    
    for _, row in filtered_df.iterrows():
        process_unit_and_update_status_sync(row.to_dict(), base_folder,threshold, max_frames, min_frame)












def dispatch_slurm_jobs(
    base_path,
    table,
    slurm_launch_file,
    predict_flag,
    conda_env="sdannce",
    partition="scavenger-gpu",
    skip_txt=None,
    max_workers=None,
    dry_run=True,
):
    """
    Generic launcher for slurm_predict scripts.

    Args:
      base_path (str): root path under which date_folder/rec_file live
      table (pyarrow.Table): must have 'date_folder' and 'rec_file' columns
      slurm_launch_file (str): path to your slurm_launch_predict.py
      predict_flag (str): e.g. "--predict_com" or "--predict_dannce"
      conda_env (str): name of the conda env to use
      partition (str): slurm partition
      skip_txt (str|None): path to a newline-separated list of rel-paths to skip
      max_workers (int|None): how many threads to run in parallel
      dry_run (bool): if True, just prints commands instead of running them
    """
    # load skip list if provided
    skip_set = set()
    if skip_txt:
        with open(skip_txt) as f:
            skip_set = {line.strip() for line in f if line.strip()}

    def job_runner(date_folder, rec_file):
        rel = os.path.join(date_folder, rec_file)
        if rel in skip_set:
            print(f"Skipping (skip list): {rel}")
            return

        expdir = os.path.join(base_path, rel)
        if not os.path.isdir(expdir):
            print(f"Skipping (not found): {expdir}")
            return

        cmd = (
            f"conda run -n {conda_env} python {slurm_launch_file} "
            f"--expdir {expdir} {predict_flag} "
            f"--partition {partition}"
        )
        if dry_run:
            print(f"[DRY-RUN] {cmd}")
        else:
            print(f"Executing: {cmd}")
            os.system(cmd)

    # build list of records
    records = [
        (df.as_py(), rf.as_py())
        for df, rf in zip(table["date_folder"], table["rec_file"])
    ]

    workers = max_workers or len(records)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for df, rf in records:
            pool.submit(job_runner, df, rf)




def process_dannce_jobs(
    base_path: str,
    for_com_vis,              # PyArrow table with 'date_folder' and 'rec_file' columns
    slurm_launch_file: str,
    txt_file: str,
    partition: str = 'scavenger-gpu',
    max_concurrent_jobs: int = 4,
    dry_run: bool = True,
):
    """
    Read skip-list from txt_file, then in parallel run or dry-run DANNCE prediction
    on each (date_folder, rec_file) in `for_com_vis`.
    """

    def check_expdir(path):
        if not os.path.exists(path):
            print(f"Skipping: {path} does not exist")
            return False
        return True

    def run_command(date_folder, rec_file):
        rel_path = os.path.join(date_folder, rec_file)
        expdir = os.path.join(base_path, rel_path)
        if rel_path in skip_set or not check_expdir(expdir):
            print(f"Skipping: {rel_path}")
            return

        cmd = (
            f"conda run -n sdannce python {slurm_launch_file}"
            f" --expdir {expdir}"
            f" --predict_dannce"
            f" --partition {partition}"
        )
        if dry_run:
            print(f"[DRY-RUN] {cmd}")
        else:
            print(f"Executing: {cmd}")
            os.system(cmd)

    # load skip list
    skip_set = set()
    with open(txt_file, 'r') as f:
        for line in f:
            p = line.strip()
            if p:
                skip_set.add(p)

    # extract records
    records = [
        (df.as_py(), rf.as_py())
        for df, rf in zip(for_com_vis['date_folder'], for_com_vis['rec_file'])
    ]

    # launch in parallel
    with ThreadPoolExecutor(max_workers=max_concurrent_jobs) as exe:
        for date_folder, rec_file in records:
            exe.submit(run_command, date_folder, rec_file)
