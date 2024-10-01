import os
import sys
sys.path.append(os.path.abspath('../..'))
import pandas as pd
import datetime
import threading
from utlis.scan_engine_utlis.scan_engine_utlis import match_date_pattern

# Thread lock for updating scan log
scan_log_lock = threading.Lock()

def load_scan_log(scan_log_path):
    if os.path.exists(scan_log_path):
        scan_log_df = pd.read_csv(scan_log_path)
    else:
        scan_log_df = pd.DataFrame(columns=['date_folder', 'rec_file', 'scan_time'])
    return scan_log_df

def save_scan_log(scan_log_df, scan_log_path):
    scan_log_df.to_csv(scan_log_path, index=False)

def clean_scan_log(scan_log_df, base_folder):
    existing_folders = set()
    date_folders = [
        f for f in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, f)) and match_date_pattern(f)
    ]
    for date_folder in date_folders:
        folder_path = os.path.join(base_folder, date_folder)
        rec_files = [
            f for f in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, f)) and f[0].isdigit()
        ]
        for rec_file in rec_files:
            existing_folders.add((date_folder, rec_file))

    # Remove entries not in existing_folders
    scan_log_df = scan_log_df[
        scan_log_df.apply(lambda row: (row['date_folder'], row['rec_file']) in existing_folders, axis=1)
    ]
    return scan_log_df

def update_scan_log(scan_log_df, date_folder, rec_file):
    with scan_log_lock:
        scan_time = datetime.datetime.now().isoformat()
        mask = (scan_log_df['date_folder'] == date_folder) & (scan_log_df['rec_file'] == rec_file)
        if scan_log_df[mask].empty:
            new_row = pd.DataFrame([{
                'date_folder': date_folder,
                'rec_file': rec_file,
                'scan_time': scan_time
            }])
            scan_log_df = pd.concat([scan_log_df, new_row], ignore_index=True)
        else:
            scan_log_df.loc[mask, 'scan_time'] = scan_time
    return scan_log_df


def get_folders_to_scan(base_folder, scan_log_df, rescan_threshold_days, force_rescan_rec_files_set):
    one_week_ago = datetime.datetime.now() - datetime.timedelta(days=rescan_threshold_days)
    folders_to_scan = {}

    date_folders = [
        f for f in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, f)) and match_date_pattern(f)
    ]

    for date_folder in date_folders:
        folder_path = os.path.join(base_folder, date_folder)
        rec_files = [
            f for f in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, f)) and f[0].isdigit()
        ]
        rec_files_to_scan = []
        for rec_file in rec_files:
            key = (date_folder, rec_file)

            # Get last scan time from scan_log_df
            last_scan_entry = scan_log_df[
                (scan_log_df['date_folder'] == date_folder) & (scan_log_df['rec_file'] == rec_file)
            ]
            last_scan_time = None
            if not last_scan_entry.empty:
                last_scan_time_str = last_scan_entry.iloc[0]['scan_time']
                last_scan_time = datetime.datetime.fromisoformat(last_scan_time_str)

            rec_file_path = os.path.join(folder_path, rec_file)
            if os.path.exists(rec_file_path):
                last_modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(rec_file_path))
            else:
                continue  # Skip if rec_file_path does not exist

            # Decide whether to scan
            needs_scan = False
            if not last_scan_time:
                needs_scan = True  # Never scanned before
            elif last_modified_time > last_scan_time:
                needs_scan = True  # Modified since last scan
            elif last_scan_time < one_week_ago:
                needs_scan = True  # Last scan was over threshold

            # Check for forced rescans
            if (date_folder, rec_file) in force_rescan_rec_files_set:
                needs_scan = True

            if needs_scan:
                rec_files_to_scan.append(rec_file)

        if rec_files_to_scan:
            folders_to_scan[date_folder] = rec_files_to_scan

    return folders_to_scan
