import os
import yaml
import re
import pandas as pd
import pyarrow.parquet as pq

# Function to load the universal status mapping from a YAML file
def load_status_mapping(file_path):
    """Load status mapping from the YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)['status_codes']

# Function to map a status code to a human-readable string
def get_status_description(status_code, status_mapping):
    """Map status code to human-readable description."""
    return status_mapping.get(status_code, "UNKNOWN")

# Read failed paths from a text file
def read_failed_paths(failed_paths_file):
    """Read failed paths from the provided file."""
    failed_paths = set()  # Use a set to avoid duplicates
    if os.path.exists(failed_paths_file):
        with open(failed_paths_file, 'r') as file:
            failed_paths = {line.strip() for line in file}
    return failed_paths

# Function for special dates handling
def is_special_date(folder_name):
    """Check if the folder's date is before the target date."""
    target_date_str = '2024_09_18'
    return folder_name < target_date_str  # Direct string comparison

# Function to find calibration files in subfolders
def find_calib_file(subfolder_path):
    """Check for calibration files starting with 'calib'."""
    for file_name in os.listdir(subfolder_path):
        if file_name.startswith("calib"):
            return file_name
    return None

# # Assign status codes based on the folder, subfolder, and calibration file information
def assign_status_codes(folder_name, subfolder_path, calib_file, failed_paths, status_mapping):
    """Assign status codes for various categories based on the folder and calibration file."""
    z_adjusted_code = 2 if not is_special_date(folder_name) else 0  # NO-NEED or NO
    sync_code = 0  # Default NO
    label3d_status_code = 0  # Default NO
    mir_generate_param_code = 0  # Default NO

    if calib_file is not None:
        mir_generate_param_code = 1  # YES
        if calib_file.startswith("df_") and calib_file.endswith("label3d_dannce.mat"):
            sync_code = 1  # YES
    
    if subfolder_path in failed_paths:
        sync_code = 3  # FAILED

    return {
        'mir_generate_param': get_status_description(mir_generate_param_code, status_mapping),
        'label3d_status': get_status_description(label3d_status_code, status_mapping),
        'sync': get_status_description(sync_code, status_mapping),
        'z_adjusted': get_status_description(z_adjusted_code, status_mapping)
    }

# Regex to match date format yyyy_mm_dd
def match_date_pattern(folder_name):
    """Match the date format yyyy_mm_dd for folder names."""
    date_pattern = re.compile(r"\d{4}_\d{2}_\d{2}")
    return date_pattern.match(folder_name)


def load_parquet(parquet_file):
    # Read the parquet file into a pandas DataFrame
    table = pq.read_table(parquet_file)
    df = table.to_pandas()
    return df