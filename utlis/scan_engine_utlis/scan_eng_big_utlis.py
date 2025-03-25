import threading
import datetime
import concurrent.futures
import sys
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Import functions from utils.py
from utlis.scan_engine_utlis.scan_log_utlis import (
    load_scan_log,
    save_scan_log,
    clean_scan_log,
    update_scan_log,
    get_folders_to_scan
)

from utlis.sync_utlis.sync_df_utlis import find_calib_file
from utlis.scan_engine_utlis.scan_engine_utlis import (
    read_failed_paths,
    match_date_pattern,
    assign_status_codes,
)

def scan_folder(folder_name, base_folder, failed_paths, config, rec_files_to_scan):
    folder_path = os.path.join(base_folder, folder_name)
    rec_files_data = []  # To store rec files and their status
    calib_files = []  # To store calibration files

    # Check for calibration files starting with 'calib'
    for file_name in os.listdir(folder_path):
        if file_name.startswith("calib"):
            calib_files.append(file_name)

    # Traverse subfolders within this folder
    for subfolder_name in rec_files_to_scan:
        subfolder_path = os.path.join(folder_path, subfolder_name)

        # Check for subfolders starting with a digit (rec folders)
        if os.path.isdir(subfolder_path) and subfolder_name[0].isdigit():
            # Find calibration file for each subfolder
            calib_file = find_calib_file(subfolder_path)

            # Assign status codes dynamically based on the config
            rec_file_data = assign_status_codes(
                folder_name, subfolder_path, calib_file, failed_paths, config
            )

            rec_file_data['rec_file'] = subfolder_name  # Add rec_file to the data
            # Add date-time for update and some future
            rec_file_data['scan_time'] = datetime.datetime.now().isoformat()

            rec_files_data.append(rec_file_data)

    return {
        'date_folder': folder_name,
        'calib_files': calib_files,  # Store the calibration files under date_folder level
        'rec_files_data': rec_files_data  # Each rec file with its status fields
    }

def log_folder_to_parquet_sep(base_folder, failed_paths_file, config, force_rescan_rec_files=None, rescan_threshold_days=7):
    """Log folders and save Parquet in subfolders with partial scan support."""

    # Paths for scan log
    scan_log_path = os.path.join(base_folder, 'paret', 'scan_log.csv')

    # Load or initialize the scan log
    scan_log_df = load_scan_log(scan_log_path)

    # Read manually inputted failed paths
    failed_paths = read_failed_paths(failed_paths_file) if failed_paths_file else set()

    # Forced rescans
    # force_rescan_rec_files = [
    #     # ('2023-10-01', '001'),
    #     # ('2023-10-02', '002'),
    #     # Add more as needed
    # ]
    # force_rescan_rec_files_set = set(force_rescan_rec_files)
    
    if force_rescan_rec_files is None:
        force_rescan_rec_files = []
    force_rescan_rec_files_set = set(force_rescan_rec_files)



    # Rescan threshold
    # rescan_threshold_days = 7

    # Determine folders to scan
    folders_to_scan = get_folders_to_scan(base_folder, scan_log_df, rescan_threshold_days, force_rescan_rec_files_set)

    if not folders_to_scan:
        print("No new or modified folders to scan.")
        return

    # Use ThreadPoolExecutor for parallel folder scanning
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for date_folder, rec_files_to_scan in folders_to_scan.items():
            futures.append(
                executor.submit(
                    scan_folder,
                    date_folder,
                    base_folder,
                    failed_paths,
                    config,
                    rec_files_to_scan
                )
            )

        for future in concurrent.futures.as_completed(futures):
            folder_log = future.result()
            date_folder = folder_log['date_folder']
            calib_files = folder_log.get('calib_files', [])

            # Ensure 'calib_files' is always a list of strings
            calib_files = [str(f) for f in calib_files] if calib_files else []

            # Process and save each experiment's log separately
            for rec_file_data in folder_log['rec_files_data']:
                rec_file = rec_file_data['rec_file']
                subfolder_save_path = os.path.join(base_folder, date_folder, rec_file, "folder_log.parquet")

                # Ensure the experiment/rec_file folder exists
                os.makedirs(os.path.dirname(subfolder_save_path), exist_ok=True)

                # Add 'date_folder' and 'calib_files' to rec_file_data
                rec_file_data['date_folder'] = date_folder
                rec_file_data['calib_files'] = calib_files

                # Dynamically ensure all relevant columns are strings based on config
                status_columns = list(config.keys())
                df = pd.DataFrame([rec_file_data])
                df[status_columns] = df[status_columns].astype(str)

                # Convert the data into a DataFrame and save the Parquet file
                table = pa.Table.from_pandas(df)
                pq.write_table(table, subfolder_save_path)

                print(f"Log for {rec_file} saved at {subfolder_save_path}")

                # Update the scan log
                scan_log_df = update_scan_log(scan_log_df, date_folder, rec_file)

    # Clean up the scan log
    scan_log_df = clean_scan_log(scan_log_df, base_folder)

    # Save the updated scan log
    save_scan_log(scan_log_df, scan_log_path)


#######################################################################################33
#below are for miniscope data!

def remove_all_parquet_files(root_path):
    """
    Recursively remove all .parquet files under the specified root path.
    for miniscope folders use only.
    """
    for dirpath, dirnames, filenames in os.walk(root_path):
        for fname in filenames:
            if fname.endswith(".parquet"):
                full_path = os.path.join(dirpath, fname)
                os.remove(full_path)
                print(f"Deleted: {full_path}")

def process_experiment_dir(animal_id, custom_label, date_str, time_str, root_dir, failed_entries, config):
    """
    Process a single experiment directory following:
    root_dir / animal_id / custom_label / date_str / time_str

    Only logs data for subfolders named 'My_V4_Miniscope'.
    """
    exp_path = os.path.join(root_dir, animal_id, custom_label, date_str, time_str)
    rec_entries = []
    calib_files = []

    # Identify calibration files within the experiment folder
    for file_name in os.listdir(exp_path):
        if file_name.startswith("calib"):
            calib_files.append(file_name)

    # Scan only for the rec folder named 'My_V4_Miniscope'
    for rec_folder in os.listdir(exp_path):
        if rec_folder == "My_V4_Miniscope":  # <-- Only process this folder
            rec_path = os.path.join(exp_path, rec_folder)
            if os.path.isdir(rec_path):
                # Detect associated calibration file (if any)
                calib_file = find_calib_file(rec_path)
                rec_entry = assign_status_codes(
                    animal_id, rec_path, calib_file, failed_entries, config
                )
                rec_entry["rec_name"] = rec_folder
                rec_entry["timestamp"] = datetime.datetime.now().isoformat()
                rec_entry["animal_id"] = animal_id
                rec_entry["custom_label"] = custom_label
                rec_entry["date"] = date_str
                rec_entry["time"] = time_str
                rec_entries.append(rec_entry)

    return {
        'animal_id': animal_id,
        'custom_label': custom_label,
        'date': date_str,
        'time': time_str,
        'calib_files': calib_files,
        'rec_entries': rec_entries
    }

#below works well, the only issues is it has redundent things.... such as calib files..
# def log_experiment_data(root_dir, failed_paths_file, config, force_reprocess=None, rescan_days=7):
#     """
#     Scan experiments and log data, saving Parquet files **only** for folders named 'My_V4_Miniscope'.

#     Assumes folder structure:
#       root_dir/
#           animal_id (e.g., "20240910-V1-R")/
#               custom_label (e.g., "customEntValHere")/
#                   date (e.g., "2024_11_13")/
#                       time (e.g., "16_18_24")/
#                           My_V4_Miniscope  <-- only these are logged

#     Generates a `folder_log.parquet` inside each rec folder that matches.
#     The scan log is updated using "animal_id/custom_label/date/time".
#     """
#     scan_log_path = os.path.join(root_dir, '#paret', 'scan_log.csv')
#     scan_log_df = load_scan_log(scan_log_path)

#     # Read manually recorded failed paths
#     failed_entries = read_failed_paths(failed_paths_file) if failed_paths_file else set()

#     if force_reprocess is None:
#         force_reprocess = []
#     force_reprocess_set = set(force_reprocess)

#     # Identify experiments needing a scan
#     experiments_to_process = []
#     for animal_id in os.listdir(root_dir):
#         animal_path = os.path.join(root_dir, animal_id)
#         if not os.path.isdir(animal_path):
#             continue
#         for custom_label in os.listdir(animal_path):
#             custom_path = os.path.join(animal_path, custom_label)
#             if not os.path.isdir(custom_path):
#                 continue
#             for date_str in os.listdir(custom_path):
#                 date_path = os.path.join(custom_path, date_str)
#                 if not os.path.isdir(date_path) or not match_date_pattern(date_str):
#                     continue
#                 for time_str in os.listdir(date_path):
#                     time_path = os.path.join(date_path, time_str)
#                     if not os.path.isdir(time_path):
#                         continue
#                     exp_key = (animal_id, custom_label, date_str, time_str)
#                     experiments_to_process.append(exp_key)

#     if not experiments_to_process:
#         print("No new or modified experiments to process.")
#         return

#     # Parallel processing of experiment directories
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = []
#         for animal_id, custom_label, date_str, time_str in experiments_to_process:
#             futures.append(
#                 executor.submit(
#                     process_experiment_dir,
#                     animal_id, custom_label, date_str, time_str,
#                     root_dir, failed_entries, config
#                 )
#             )

#         for future in concurrent.futures.as_completed(futures):
#             exp_data = future.result()
#             animal_id = exp_data['animal_id']
#             custom_label = exp_data['custom_label']
#             date_str = exp_data['date']
#             time_str = exp_data['time']
#             calib_files = exp_data.get('calib_files', [])
#             calib_files = [str(f) for f in calib_files] if calib_files else []

#             # Process and save logs **only** for the matching rec entries
#             for rec_entry in exp_data['rec_entries']:
#                 rec_name = rec_entry['rec_name']
#                 save_path = os.path.join(
#                     root_dir,
#                     animal_id,
#                     custom_label,
#                     date_str,
#                     time_str,
#                     rec_name,
#                     "folder_log.parquet"
#                 )
#                 os.makedirs(os.path.dirname(save_path), exist_ok=True)

#                 # Attach calibration info
#                 rec_entry['calib_files'] = calib_files

#                 # Convert status fields to string based on config
#                 status_columns = list(config.keys())
#                 df = pd.DataFrame([rec_entry])
#                 df[status_columns] = df[status_columns].astype(str)

#                 table = pa.Table.from_pandas(df)
#                 pq.write_table(table, save_path)
#                 print(f"Log for {rec_name} saved at {save_path}")

#                 # Update scan log with a key like "animal_id/custom_label/date/time"
#                 scan_key = f"{animal_id}/{custom_label}/{date_str}/{time_str}"
#                 scan_log_df = update_scan_log(scan_log_df, scan_key, rec_name)

#     # Finalize scan log
#     scan_log_df = clean_scan_log(scan_log_df, root_dir)
#     save_scan_log(scan_log_df, scan_log_path)

def load_manual_log(csv_path):
    import csv
    manual_log = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rec_path = row['rec_path']
            condition = row['condition']
            manual_log[rec_path] = condition
    return manual_log


def process_experiment_dir_with_manual(animal_id, custom_label, date_str, time_str,
                                         root_dir, failed_entries, config, manual_log):
    # Call your existing function
    exp_data = process_experiment_dir(animal_id, custom_label, date_str, time_str,
                                      root_dir, failed_entries, config)
    
    # Compute the full experiment path (adjust if necessary)
    experiment_path = os.path.join(root_dir, animal_id, custom_label, date_str, time_str, "My_V4_Miniscope")
    
    # Determine mapping based on manual log
    if experiment_path in manual_log:
        print("Match found:", experiment_path)
        mapping_value = 1
        quality_value = manual_log[experiment_path]
    else:
        mapping_value = 0
        quality_value = 'unknown'
    
    # Update the top-level experiment data (if needed elsewhere)
    exp_data['mapped'] = mapping_value
    exp_data['quality'] = quality_value

    # Update each rec_entry with the computed values so that these fields are logged
    for rec_entry in exp_data['rec_entries']:
        rec_entry['mapped'] = mapping_value
        rec_entry['quality'] = quality_value
        # Optionally, also include experiment_path if your config conditions need it:
        rec_entry['experiment_path'] = experiment_path
    
    return exp_data



def log_experiment_data(root_dir, manual_log_path, failed_paths_file, config, force_reprocess=None, rescan_days=7):
    # Load the manual CSV log once
    manual_log = load_manual_log(manual_log_path)
    
    scan_log_path = os.path.join(root_dir, '#paret', 'scan_log.csv')
    scan_log_df = load_scan_log(scan_log_path)
    
    failed_entries = read_failed_paths(failed_paths_file) if failed_paths_file else set()
    
    if force_reprocess is None:
        force_reprocess = []
    force_reprocess_set = set(force_reprocess)
    
    experiments_to_process = []
    for animal_id in os.listdir(root_dir):
        animal_path = os.path.join(root_dir, animal_id)
        if not os.path.isdir(animal_path):
            continue
        for custom_label in os.listdir(animal_path):
            custom_path = os.path.join(animal_path, custom_label)
            if not os.path.isdir(custom_path):
                continue
            for date_str in os.listdir(custom_path):
                date_path = os.path.join(custom_path, date_str)
                if not os.path.isdir(date_path) or not match_date_pattern(date_str):
                    continue
                for time_str in os.listdir(date_path):
                    time_path = os.path.join(date_path, time_str)
                    if not os.path.isdir(time_path):
                        continue
                    exp_key = (animal_id, custom_label, date_str, time_str)
                    experiments_to_process.append(exp_key)
    
    if not experiments_to_process:
        print("No new or modified experiments to process.")
        return
    
    # Parallel processing using the wrapper that passes down the manual log
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for animal_id, custom_label, date_str, time_str in experiments_to_process:
            futures.append(
                executor.submit(
                    process_experiment_dir_with_manual,
                    animal_id, custom_label, date_str, time_str,
                    root_dir, failed_entries, config, manual_log
                )
            )
    
        for future in concurrent.futures.as_completed(futures):
            exp_data = future.result()
            animal_id = exp_data['animal_id']
            custom_label = exp_data['custom_label']
            date_str = exp_data['date']
            time_str = exp_data['time']
    
            for rec_entry in exp_data['rec_entries']:
                save_path = os.path.join(
                    root_dir,
                    animal_id,
                    custom_label,
                    date_str,
                    time_str,
                    "folder_log.parquet"
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                status_columns = list(config.keys())
                df = pd.DataFrame([rec_entry])
                df = df.drop(columns=['rec_name', 'calib_files'], errors='ignore')
                df[status_columns] = df[status_columns].astype(str)
                table = pa.Table.from_pandas(df)
                pq.write_table(table, save_path)
                scan_key = f"{animal_id}/{custom_label}/{date_str}/{time_str}"
                print(f"Log for {scan_key} saved at {save_path}")
                scan_log_df = update_scan_log(scan_log_df, date_str, scan_key)
    
    scan_log_df = clean_scan_log(scan_log_df, root_dir)
    save_scan_log(scan_log_df, scan_log_path)


# #below function is without manual red flags.
# def log_experiment_data(root_dir, failed_paths_file, config, force_reprocess=None, rescan_days=7):
#     """
#     Scan experiments and log data, saving Parquet files **only** for folders named 'My_V4_Miniscope'.

#     Assumes folder structure:
#       root_dir/
#           animal_id (e.g., "20240910-V1-R")/
#               custom_label (e.g., "customEntValHere")/
#                   date (e.g., "2024_11_13")/
#                       time (e.g., "16_18_24")/
#                           My_V4_Miniscope  <-- only these are logged

#     Generates a `folder_log.parquet` inside each rec folder that matches.
#     The scan log is updated using "animal_id/custom_label/date/time".
#     """
#     # manual_log = load_manual_log(manual_log_path)
#     scan_log_path = os.path.join(root_dir, '#paret', 'scan_log.csv')
#     scan_log_df = load_scan_log(scan_log_path)

#     # Read manually recorded failed paths
#     failed_entries = read_failed_paths(failed_paths_file) if failed_paths_file else set()

#     if force_reprocess is None:
#         force_reprocess = []
#     force_reprocess_set = set(force_reprocess)

#     # Identify experiments needing a scan
#     experiments_to_process = []
#     for animal_id in os.listdir(root_dir):
#         animal_path = os.path.join(root_dir, animal_id)
#         if not os.path.isdir(animal_path):
#             continue
#         for custom_label in os.listdir(animal_path):
#             custom_path = os.path.join(animal_path, custom_label)
#             if not os.path.isdir(custom_path):
#                 continue
#             for date_str in os.listdir(custom_path):
#                 date_path = os.path.join(custom_path, date_str)
#                 if not os.path.isdir(date_path) or not match_date_pattern(date_str):
#                     continue
#                 for time_str in os.listdir(date_path):
#                     time_path = os.path.join(date_path, time_str)
#                     if not os.path.isdir(time_path):
#                         continue
#                     exp_key = (animal_id, custom_label, date_str, time_str)
#                     experiments_to_process.append(exp_key)

#     if not experiments_to_process:
#         print("No new or modified experiments to process.")
#         return

#     # Parallel processing of experiment directories
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = []
#         # the commented part is without the red flag version.
#         for animal_id, custom_label, date_str, time_str in experiments_to_process:
#             futures.append(
#                 executor.submit(
#                     process_experiment_dir,
#                     animal_id, custom_label, date_str, time_str,
#                     root_dir, failed_entries, config
#                 )
#             )


#         # #trying the red flag thing
#         # for animal_id, custom_label, date_str, time_str in experiments_to_process:
#         #     experiment_path = os.path.join(root_dir, animal_id, custom_label, date_str, time_str, "My_V4_Miniscope")
#         #     futures.append(
#         #         executor.submit(
#         #             process_experiment_dir,
#         #             animal_id, custom_label, date_str, time_str,
#         #             root_dir, failed_entries, config,
#         #             experiment_path=experiment_path,
#         #             manual_log=manual_log
#         #         )
#         #     )

#         for future in concurrent.futures.as_completed(futures):
#             exp_data = future.result()
#             animal_id = exp_data['animal_id']
#             custom_label = exp_data['custom_label']
#             date_str = exp_data['date']
#             time_str = exp_data['time']

#             # Process and save logs **only** for the matching rec entries
#             for rec_entry in exp_data['rec_entries']:
#                 save_path = os.path.join(
#                     root_dir,
#                     animal_id,
#                     custom_label,
#                     date_str,
#                     time_str,
#                     "folder_log.parquet"
#                 )
#                 os.makedirs(os.path.dirname(save_path), exist_ok=True)

#                 # Convert status fields to string based on config
#                 status_columns = list(config.keys())
#                 df = pd.DataFrame([rec_entry])
#                 # Drop unwanted columns (e.g., rec_name and calib_files) if present
#                 df = df.drop(columns=['rec_name', 'calib_files'], errors='ignore')
#                 df[status_columns] = df[status_columns].astype(str)

#                 table = pa.Table.from_pandas(df)
#                 pq.write_table(table, save_path)

#                 # Update scan log with a key like "animal_id/custom_label/date/time"
#                 scan_key = f"{animal_id}/{custom_label}/{date_str}/{time_str}"
#                 print(f"Log for {scan_key} saved at {save_path}")
#                 scan_log_df = update_scan_log(scan_log_df, date_str, scan_key)

#     # Finalize scan log
#     scan_log_df = clean_scan_log(scan_log_df, root_dir)
#     save_scan_log(scan_log_df, scan_log_path)
