import os
import yaml
import re
import pandas as pd
import pyarrow.parquet as pq
# from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow.dataset as ds  # Import pyarrow.dataset at the top
import pyarrow as pa
# from datetime import datetime


def read_all_parquet_files(base_folder):
    """
    Efficiently read all Parquet files from the base folder using PyArrow's Dataset,
    returning a PyArrow table with all columns.
    
    Parameters:
    - base_folder (str): Path to the base folder containing Parquet files.
    
    Returns:
    - table (pa.Table): Combined PyArrow table with all columns.
    """
    # Create a dataset from the base folder
    dataset = ds.dataset(base_folder, format="parquet", exclude_invalid_files=True)
    
    # Read the entire dataset with all columns
    table = dataset.to_table()
    
    return table


def read_all_parquet_files_auto_exclude(base_folder, exclude_columns=None):
    """
    Efficiently read all Parquet files from the base folder using PyArrow's Dataset,
    dynamically excluding specified columns and returning a PyArrow table.
    
    Parameters:
    - base_folder (str): Path to the base folder containing Parquet files.
    - exclude_columns (list): List of column names to exclude from the read. Defaults to None.
    
    Returns:
    - table (pa.Table): Combined PyArrow table with specified columns excluded.
    """
    # Create a dataset from the base folder
    dataset = ds.dataset(base_folder, format="parquet", exclude_invalid_files=True)
    
    # Get the first fragment's schema to determine available columns
    first_fragment = next(dataset.get_fragments())
    schema = first_fragment.physical_schema
    
    # List of all available columns
    all_columns = schema.names
    
    # Exclude the columns specified by the user
    if exclude_columns:
        columns_to_read = [col for col in all_columns if col not in exclude_columns]
    else:
        columns_to_read = all_columns
    
    # Read the dataset with the specified columns
    table = dataset.to_table(columns=columns_to_read)
    
    return table


#everything to pandas is extremely stupid...
# def read_all_parquet_files_auto_exclude(base_folder, exclude_columns=None):
#     """
#     Efficiently read all Parquet files from the base folder using PyArrow's Dataset,
#     dynamically excluding specified columns.
    
#     Parameters:
#     - base_folder (str): Path to the base folder containing Parquet files.
#     - exclude_columns (list): List of column names to exclude from the read. Defaults to None.
    
#     Returns:
#     - df (pd.DataFrame): Combined DataFrame with specified columns excluded.
#     """
#     # Create a dataset from the base folder
#     dataset = ds.dataset(base_folder, format="parquet", exclude_invalid_files=True)
    
#     # Get the first fragment's schema to determine available columns
#     first_fragment = next(dataset.get_fragments())
#     schema = first_fragment.physical_schema
    
#     # List of all available columns
#     all_columns = schema.names
    
#     # Exclude the columns specified by the user
#     if exclude_columns:
#         columns_to_read = [col for col in all_columns if col not in exclude_columns]
#     else:
#         columns_to_read = all_columns
    
#     # Read the dataset with the specified columns
#     table = dataset.to_table(columns=columns_to_read)
    
#     # Convert the table to a Pandas DataFrame
#     df = table.to_pandas()
    
#     return df

# #this function may cause errors...
# def read_existing_parquet_files(base_folder):
#     """
#     Read all existing Parquet files under base_folder and return a DataFrame
#     with 'date_folder', 'rec_file', and 'scan_time'.
#     """
#     try:
#         # Create a dataset from the base folder; PyArrow will automatically recurse into subfolders
#         dataset = ds.dataset(base_folder, format="parquet", exclude_invalid_files=True)

#         # Convert the dataset to a PyArrow table and then to a Pandas DataFrame
#         table = dataset.to_table(columns=['date_folder', 'rec_file', 'scan_time'])
        
#         # Convert to Pandas DataFrame for further manipulation
#         df = table.to_pandas()

#         return df
#     except Exception as e:
#         print(f"Could not read existing Parquet files: {e}")
#         return pd.DataFrame(columns=['date_folder', 'rec_file', 'scan_time'])

# def read_all_parquet_files(base_folder):
#     """
#     Efficiently read all Parquet files from the date folder structure using PyArrow's Dataset,
#     and return a combined DataFrame.
#     """
#     # Create a dataset from the base folder; PyArrow will automatically recurse into subfolders
#     dataset = ds.dataset(base_folder, format="parquet", exclude_invalid_files=True)

#     # Convert the dataset to a PyArrow table and then to a Pandas DataFrame
#     table = dataset.to_table()

#     # Convert to Pandas DataFrame for further manipulation
#     df = table.to_pandas()

#     return df


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

# # Assign status codes based on the folder, subfolder, and calibration file information
# def assign_status_codes_cha(folder_name, subfolder_path, calib_file, failed_paths, status_mapping):
#     """Assign status codes for various categories based on the folder and calibration file."""
#     z_adjusted_code = 2 if not is_special_date(folder_name) else 0  # NO-NEED or NO
#     sync_code = 0  # Default NO
#     label3d_status_code = 0  # Default NO
#     mir_generate_param_code = 0  # Default NO

#     if calib_file is not None:
#         mir_generate_param_code = 1  # YES
#         if calib_file.startswith("df_") and calib_file.endswith("label3d_dannce.mat"):
#             sync_code = 1  # YES
    
#     if subfolder_path in failed_paths:
#         sync_code = 3  # FAILED

#     return {
#         'mir_generate_param': get_status_description(mir_generate_param_code, status_mapping),
#         'label3d_status': get_status_description(label3d_status_code, status_mapping),
#         'sync': get_status_description(sync_code, status_mapping),
#         'z_adjusted': get_status_description(z_adjusted_code, status_mapping)
#     }

# Assign numerical status codes instead of descriptions
# def assign_status_codes_numb(folder_name, subfolder_path, calib_file, failed_paths):
#     """Assign numerical status codes for various categories."""
#     z_adjusted_code = 2 if not is_special_date(folder_name) else 0  # NO-NEED or NO
#     sync_code = 0  # Default NO
#     mir_generate_param_code = 0  # Default NO

#     if calib_file is not None:
#         mir_generate_param_code = 1  # YES
#         if calib_file.startswith("df_") and calib_file.endswith("label3d_dannce.mat"):
#             sync_code = 1  # YES
    
#     for file_name in os.listdir(subfolder_path):
#         if file_name.endswith("label3d_dannce.mat.old"):
#             z_adjusted_code = 1
    
#     if subfolder_path in failed_paths:
#         sync_code = 3  # FAILED

#     return {
#         'mir_generate_param': mir_generate_param_code,  # Return numerical code
#         'sync': sync_code,  # Return numerical code
#         'z_adjusted': z_adjusted_code  # Return numerical code
#     }

#somehow, some assesment issues are here...
# def assign_status_codes(folder_name, subfolder_path, calib_file, failed_paths, config):
#     """Assign numerical status codes for various categories based on dynamic config."""
#     status_codes = {}

#     for status_field, rules in config.items():
#         # Start with the default value
#         status_code = rules['default']

#         # Apply each condition in the rules
#         for condition_rule in rules['conditions']:
#             condition = condition_rule['condition']
            
#             # Check if the condition is met, passing the necessary arguments
#             if 'folder_name' in condition.__code__.co_varnames:
#                 if condition(folder_name):
#                     status_code = condition_rule['value']
#             elif 'subfolder_path' in condition.__code__.co_varnames and 'failed_paths' in condition.__code__.co_varnames:
#                 if condition(subfolder_path, failed_paths):
#                     status_code = condition_rule['value']
#             elif 'calib_file' in condition.__code__.co_varnames:
#                 if condition(calib_file):
#                     status_code = condition_rule['value']

#         # Assign the final code to the status field
#         status_codes[status_field] = status_code

#     return status_codes

def assign_status_codes(folder_name, subfolder_path, calib_file, failed_paths, config):
    """Assign numerical status codes for various categories based on dynamic config."""
    status_codes = {}

    # Prepare the context dictionary
    context = {
        'folder_name': folder_name,
        'subfolder_path': subfolder_path,
        'calib_file': calib_file,
        'failed_paths': failed_paths,
    }

    for status_field, rules in config.items():
        # Start with the default value
        status_code = rules['default']

        # Apply each condition in the rules
        for condition_rule in rules['conditions']:
            condition = condition_rule['condition']
            try:
                if condition(**context):
                    # print(f"Evaluating condition for {status_field}: {condition.__name__ if hasattr(condition, '__name__') else condition} with context {context}. Result: {condition(**context)}")
                    status_code = condition_rule['value']
            except Exception as e:
                print(f"Error evaluating condition for {status_field}: {e}")
                pass

        # Assign the final code to the status field
        status_codes[status_field] = status_code

    return status_codes




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

def translate_status_code(status_code, status_mapping):
    """Translate numerical status code to human-readable string."""
    return status_mapping.get(status_code, "UNKNOWN")


# Function to create a timestamped backup of the main Parquet file
# def backup_parquet_file(parquet_file):
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # backup_file = parquet_file.replace(".parquet", f"_{timestamp}.parquet")
    # copyfile(parquet_file, backup_file)
    # print(f"Backup created: {backup_file}")
    # return backup_file