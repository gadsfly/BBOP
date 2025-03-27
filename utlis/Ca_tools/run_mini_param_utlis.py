import pyarrow as pa
import subprocess
import os

from utlis.Ca_tools.mini_param_vis_utlis import vis_param_opti


def run_param_search_from_table(filtered_table_mini, method):
    """
    Iterate through each row in `filtered_table_mini` (a PyArrow Table),
    construct the session_dir from relevant columns, and then call the appropriate
    param_search_driver script in a subprocess based on the specified method.
    
    Parameters:
        filtered_table_mini (pyarrow.Table): Table containing columns such as
            "animal_id", "custom_label", "date", "time".
        method (str): The type of param search to run. Options: "single", "simple", "complex".
    
    Raises:
        ValueError: If an unsupported method is provided.
    """
    # Convert the PyArrow Table to a list of dictionaries, each representing one row.
    rows = filtered_table_mini.to_pylist()
    
    # Map the method to the corresponding script file.
    base_script_dir = "/home/lq53/mir_repos/BBOP/random_tests/25mar_minibbop_integration"
    if method == "single":
        script_name = "param_search_driver_single.py"
    elif method == "simple":
        script_name = "param_search_driver_simple.py"
    elif method == "complex":
        script_name = "param_search_driver_complex.py"
    else:
        raise ValueError(f"Unsupported method: {method}. Choose from 'single', 'simple', or 'complex'.")
    
    script_path = os.path.join(base_script_dir, script_name)
    
    # Iterate through each row and run the appropriate script.
    for row in rows:
        animal_id    = row["animal_id"]       # e.g. "20241015PMCBE1"
        custom_label = row["custom_label"]    # e.g. "customEntValHere"
        date_str     = row["date"]            # e.g. "2025_03_11"
        time_str     = row["time"]            # e.g. "14_22_12"
        
        # Construct the session directory path.
        session_dir = os.path.join(
            "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted",
            animal_id,
            custom_label,
            date_str,
            time_str,
            "My_V4_Miniscope"
        )
        
        # (Optional) Verify that the directory exists.
        if not os.path.exists(session_dir):
            print(f"Warning: session_dir not found: {session_dir}")
            # Optionally create it:
            # os.makedirs(session_dir, exist_ok=True)
        
        print(f"\n=== Running {script_name} for session ===")
        print(f"Animal:       {animal_id}")
        print(f"Session dir:  {session_dir}")
        
        # Build the command to call the script using conda.
        cmd = [
            "conda", "run", "-n", "minian",
            "python", script_path,
            "--session_dir", session_dir,
            "--animal", animal_id
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stdout:
            print("[STDOUT]", result.stdout)
        if result.stderr:
            print("[STDERR]", result.stderr)


# def run_param_search_from_table(filtered_table_mini, method):
#     """
#     Iterate through each row in `filtered_table_mini` (a PyArrow Table),
#     construct the session_dir from relevant columns, and then call
#     param_search_driver.py in a subprocess with the desired arguments.
#     """
#     # Convert table to a list of dictionaries, each representing one row
#     rows = filtered_table_mini.to_pylist()

#     for row in rows:
#         # Extract needed columns (adjust the column names as necessary)
#         # e.g. row["animal_id"], row["custom_label"], row["date"], row["time"]
#         animal_id    = row["animal_id"]       # e.g. "20241015PMCBE1"
#         custom_label = row["custom_label"]    # e.g. "customEntValHere"
#         date_str     = row["date"]            # e.g. "2025_03_11"
#         time_str     = row["time"]            # e.g. "14_22_12"

#         # Build the session_dir path based on your folder structure.
#         # Example:
#         #   /data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/{animal_id}/{custom_label}/{date_str}/{time_str}/My_V4_Miniscope
#         session_dir = os.path.join(
#             "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted",
#             animal_id,
#             custom_label,
#             date_str,
#             time_str,
#             "My_V4_Miniscope"
#         )

#         # (Optional) Check that it exists or not, or create if needed:
#         if not os.path.exists(session_dir):
#             print(f"Warning: session_dir not found: {session_dir}")
#             # If you want to create it:
#             # os.makedirs(session_dir, exist_ok=True)

#         print(f"\n=== Running param_search_driver for session ===")
#         print(f"Animal:       {animal_id}")
#         print(f"Session dir:  {session_dir}")

#         # Now call param_search_driver.py, passing in the appropriate args.
#         # Adjust environment calls if needed (conda, absolute path, etc.)
#         cmd = [
#             "conda", "run", "-n", "minian",
#             "python",               # or "conda", "run", "-n", "my_env", "python" if you prefer
#             "param_search_driver.py",
#             "--session_dir", session_dir,
#             "--animal", animal_id
#         ]

#         result = subprocess.run(cmd, capture_output=True, text=True)
        
#         if result.stdout:
#             print("[STDOUT]", result.stdout)
#         if result.stderr:
#             print("[STDERR]", result.stderr)


# def run_single_minian_from_table(filtered_table_mini):
#     """
#     Iterate through each row in `filtered_table_mini` (a PyArrow Table),
#     construct the session_dir from relevant columns, and then call
#     param_search_driver.py in a subprocess with the desired arguments.
#     """
#     # Convert table to a list of dictionaries, each representing one row
#     rows = filtered_table_mini.to_pylist()

#     for row in rows:
#         # Extract needed columns (adjust the column names as necessary)
#         # e.g. row["animal_id"], row["custom_label"], row["date"], row["time"]
#         animal_id    = row["animal_id"]       # e.g. "20241015PMCBE1"
#         custom_label = row["custom_label"]    # e.g. "customEntValHere"
#         date_str     = row["date"]            # e.g. "2025_03_11"
#         time_str     = row["time"]            # e.g. "14_22_12"

#         # Build the session_dir path based on your folder structure.
#         # Example:
#         #   /data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/{animal_id}/{custom_label}/{date_str}/{time_str}/My_V4_Miniscope
#         session_dir = os.path.join(
#             "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted",
#             animal_id,
#             custom_label,
#             date_str,
#             time_str,
#             "My_V4_Miniscope"
#         )

#         # (Optional) Check that it exists or not, or create if needed:
#         if not os.path.exists(session_dir):
#             print(f"Warning: session_dir not found: {session_dir}")
#             # If you want to create it:
#             # os.makedirs(session_dir, exist_ok=True)

#         print(f"\n=== Running param_search_driver for session ===")
#         print(f"Animal:       {animal_id}")
#         print(f"Session dir:  {session_dir}")

#         # Now call param_search_driver.py, passing in the appropriate args.
#         # Adjust environment calls if needed (conda, absolute path, etc.)
#         cmd = [
#             "conda", "run", "-n", "minian",
#             "python",               # or "conda", "run", "-n", "my_env", "python" if you prefer
#             "param_search_driver_single.py",
#             "--session_dir", session_dir,
#             "--animal", animal_id
#         ]

#         result = subprocess.run(cmd, capture_output=True, text=True)
        
#         if result.stdout:
#             print("[STDOUT]", result.stdout)
#         if result.stderr:
#             print("[STDERR]", result.stderr)

def run_vis_param_opti_from_table(mini_param_for_vis):
    """
    Iterate through each row in `mini_param_for_vis` (a PyArrow Table),
    construct the session_dir from relevant columns, and then call
    vis_param_opti for visualization parameter optimization.
    """
    # Convert table to a list of dictionaries, each representing one row.
    rows = mini_param_for_vis.to_pylist()

    for row in rows:
        # Extract needed columns (adjust the column names as necessary)
        # e.g. row["animal_id"], row["custom_label"], row["date"], row["time"]
        animal_id    = row["animal_id"]       # e.g. "20241015PMCBE1"
        custom_label = row["custom_label"]    # e.g. "customEntValHere"
        date_str     = row["date"]            # e.g. "2025_03_11"
        time_str     = row["time"]            # e.g. "14_22_12"

        # Build the session_dir path based on your folder structure.
        session_dir = os.path.join(
            "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted",
            animal_id,
            custom_label,
            date_str,
            time_str,
            "My_V4_Miniscope"
        )

        # (Optional) Check that the directory exists; warn if not.
        if not os.path.exists(session_dir):
            print(f"Warning: session_dir not found: {session_dir}")
            # Optionally create the directory:
            # os.makedirs(session_dir, exist_ok=True)

        print(f"\n=== Running vis_param_opti for session ===")
        print(f"Animal:       {animal_id}")
        print(f"Session dir:  {session_dir}")

        try:
            vis_param_opti(session_dir)
        except Exception as e:
            print(f"Error processing session {session_dir}: {e}")

