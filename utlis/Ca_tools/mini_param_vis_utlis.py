import os
import glob
import shutil
import tempfile
import matplotlib.pyplot as plt
import json

import numpy as np
from scipy.ndimage import binary_erosion
import xarray as xr
import pandas as pd

from utlis.Ca_tools.roi_spike_vis_utlis import (
    load_minian_data_specific,
    # calculate_dff,
    overlay_all_roi_edges_no_show
)


def vis_param_opti(session_dir):
# Use your real session directory path.
    # session_dir = "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted/20241224PMCLE1/customEntValHere/2025_02_13/11_07_37/My_V4_Miniscope"
    nc_files = glob.glob(os.path.join(session_dir, "*.nc")) #minian_dataset

    if not nc_files:
        print("No output files found in the session directory.")
    else:
        print(f"Found {len(nc_files)} output files:")
        for f in nc_files:
            print(f)

    mini_timestamps = os.path.join(session_dir, 'timeStamps.csv')

    # Remove previous overlay PNG files (trash) before processing new ones.
    old_overlay_files = glob.glob(os.path.join(session_dir, "overlay_*.png"))
    for old_file in old_overlay_files:
        try:
            os.remove(old_file)
            print(f"Removed old overlay file: {old_file}")
        except Exception as e:
            print(f"Error removing {old_file}: {e}")

    # Set display_only to True to display the overlays in the notebook (won't save PNG files).
    # Set to False if you want to save the images instead.
    display_only = True

    for nc_file in nc_files:
        print("Processing:", nc_file)
        
        # Copy each file to a temporary file in the system's temporary directory.
        temp_nc = os.path.join(tempfile.gettempdir(), os.path.basename(nc_file))
        shutil.copy2(nc_file, temp_nc)
        
        # Load the data (fully loaded into memory) and timestamps.
        data, ts = load_minian_data_specific(temp_nc, mini_timestamps)
        
        # Extract the max projection.
        max_proj = data['max_proj'].values
        
        # Create the overlay figure (without automatically displaying it).
        fig, ax = overlay_all_roi_edges_no_show(data, max_proj)
        
        # Update the title with an ID derived from the filename.
        combination_id = os.path.basename(nc_file).replace("minian_dataset_", "").replace(".nc", "")
        # title = "_".join(session_dir.split("/")[-5:-2])
        title = "_".join([session_dir.split("/")[-5], session_dir.split("/")[-3], session_dir.split("/")[-2]])
        ax.set_title(title + combination_id) #"Overlay ROI Edges: "
        
        if display_only:
            # For VSCode Jupyter Notebook, display the figure inline.
            plt.show()
        else:
            # Save the figure if not in display-only mode.
            output_fig = os.path.join(session_dir, f"overlay_{combination_id}.png")
            fig.savefig(output_fig)
            print(f"Saved overlay to: {output_fig}")
        
        plt.close(fig)
        os.remove(temp_nc)


#below will skip defaults, not good
# def vis_param_all_with_mapping(csv_file_path, mapping_json_path, threshold=10, display_only=True):
#     """
#     Processes every session defined in the CSV file and generates overlay images.
    
#     For each session (with columns "rec_path" and "nc_file" in the CSV):
#       - It finds the .nc files matching the stored selection identifier.
#       - It generates an overlay plot using your helper functions.
#       - It either displays or saves the overlay in the session directory.
    
#     Then, if a valid mapping entry is found (i.e. time_diff < threshold and a rec_path exists)
#     in the mapping JSON file (which maps session directories to a recording path), it will 
#     also save a duplicate copy of the overlay image in a subfolder called "MIR_Aligned" 
#     within the mapped recording directory.
    
#     Parameters:
#        csv_file_path (str): Path to the CSV file containing 'rec_path' and 'nc_file'.
#        mapping_json_path (str): Path to the JSON file with the mapping information.
#        threshold (int, optional): Maximum allowed time_diff for saving the mapped copy. Defaults to 10.
#        display_only (bool, optional): If True, overlays are only shown inline; if False, they are saved.
#     """
#     # Load the CSV file.
#     try:
#         df = pd.read_csv(csv_file_path)
#     except Exception as e:
#         print(f"Error reading CSV file '{csv_file_path}': {e}")
#         return

#     # Load the JSON mapping.
#     try:
#         with open(mapping_json_path, 'r') as f:
#             mapping = json.load(f)
#     except Exception as e:
#         print(f"Error reading mapping JSON file '{mapping_json_path}': {e}")
#         mapping = {}
    
#     # Loop through each entry in the CSV.
#     for index, row in df.iterrows():
#         session_dir = row['rec_path']
#         selection = row['nc_file']
#         print(f"\nProcessing row {index}: session_dir = {session_dir}, selection = {selection}")
        
#         # Create a glob pattern to find .nc files in the session directory that contain the selection.
#         pattern = os.path.join(session_dir, f"*{selection}*.nc")
#         nc_files = glob.glob(pattern)
#         if not nc_files:
#             print(f"No .nc files matching pattern '{pattern}' found in session directory: {session_dir}")
#             continue
#         else:
#             print(f"Found {len(nc_files)} matching .nc file(s) for session: {session_dir}")
        
#         # Remove any old overlay PNG files in the session directory.
#         old_overlay_files = glob.glob(os.path.join(session_dir, "overlay_*.png"))
#         for old_file in old_overlay_files:
#             try:
#                 os.remove(old_file)
#                 print(f"Removed old overlay file: {old_file}")
#             except Exception as e:
#                 print(f"Error removing {old_file}: {e}")
        
#         # Process each matching .nc file.
#         for nc_file in nc_files:
#             print("Processing file:", nc_file)
#             temp_nc = os.path.join(tempfile.gettempdir(), os.path.basename(nc_file))
#             shutil.copy2(nc_file, temp_nc)
            
#             mini_timestamps = os.path.join(session_dir, 'timeStamps.csv')
#             try:
#                 data, ts = load_minian_data_specific(temp_nc, mini_timestamps)
#             except Exception as e:
#                 print(f"Error loading data from {temp_nc}: {e}")
#                 continue

#             max_proj = data['max_proj'].values
#             fig, ax = overlay_all_roi_edges_no_show(data, max_proj)
            
#             # Derive a combination id and title from the session_dir.
#             combination_id = os.path.basename(nc_file).replace("minian_dataset_", "").replace(".nc", "")
#             try:
#                 title = "_".join([session_dir.split("/")[-5], session_dir.split("/")[-3], session_dir.split("/")[-2]])
#             except IndexError:
#                 title = "session"
#             ax.set_title(f"{title}_{combination_id}")
            
#             # Determine the overlay image output path within the session directory.
#             local_output_path = os.path.join(session_dir, f"overlay_{selection}.png")
#             if display_only:
#                 plt.show()
#             else:
#                 fig.savefig(local_output_path)
#                 print(f"Saved overlay locally to: {local_output_path}")
            
#             # Check if there's a mapping for the session directory, and if valid, save a duplicate.
#             mapping_entry = mapping.get(session_dir)
#             if mapping_entry:
#                 time_diff = mapping_entry.get("time_diff")
#                 mapped_rec = mapping_entry.get("rec_path")
#                 if time_diff is not None and time_diff < threshold and mapped_rec:
#                     # Save the image inside a subfolder "MIR_Aligned" in the directory containing the mapped rec file.
#                     dest_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(mapped_rec))), "MIR_Aligned")
#                     os.makedirs(dest_dir, exist_ok=True)
#                     mapped_output_path = os.path.join(dest_dir, f"overlay_{selection}.png")
#                     fig.savefig(mapped_output_path)
#                     print(f"Saved overlay into mapped directory: {mapped_output_path}")
#                 else:
#                     print("Mapping found but time_diff is too high or rec_path missing; no mapped copy saved.")
#             else:
#                 print("No mapping found for session; skipping mapped save.")
            
#             # Clean up: close figure and remove temporary file.
#             plt.close(fig)
#             os.remove(temp_nc)


# def move_existing_overlays(mini_path):
#     """
#     Moves any existing overlay_*.png files in `mini_path` into a subfolder named `prev_overlays`.
#     If `prev_overlays` does not exist, it is created.
#     """
#     prev_dir = os.path.join(mini_path, "prev_overlays")
#     os.makedirs(prev_dir, exist_ok=True)

#     pattern = os.path.join(mini_path, "overlay_*.png")
#     for overlay_file in glob.glob(pattern):
#         dest = os.path.join(prev_dir, os.path.basename(overlay_file))
#         try:
#             shutil.move(overlay_file, dest)
#         except Exception as e:
#             print(f"Error moving {overlay_file} to {dest}: {e}")


#below will produce multiple, not good.
# def vis_param_missing_from_df_with_mapping(df, mapping_json_path, threshold=10, display_only=True):
#     """
#     Processes sessions from a DataFrame (with a 'path' column) that might be missing in the CSV.
#     For each session directory in df['path']:
    
#       1. Constructs a candidate rec_path by appending "DANNCE/predict00/save_data_AVG.mat" to the given path.
#       2. Uses an inverted mapping JSON (mini_path -> mapping info) to determine the corresponding mini_path.
#          If no mapping entry is found, the given path is used as the mini_path.
#       3. Looks for .nc files in the mini_path that match "minian_dataset*.nc".  
#          - If exactly one file is found (typically "minian_dataset.nc"), that file is used.
#          - If multiple files are found, if one of the filenames contains an extra identifier (e.g. "denoised"),
#            that file is selected. Otherwise, the first file is chosen.
#       4. Any previous overlay files in the mini_path (files beginning with "overlay_") are removed.
#       5. A temporary copy of the selected netCDF file is created and loaded (using an associated 'timeStamps.csv'
#          from the mini_path).  
#       6. An overlay is generated and its title is set using a combination of folder name components and the
#          extracted combination id from the .nc filename.
#       7. Depending on the `display_only` flag:
#          - If True, the function prints out the intended local and mapped file paths before showing the plot.
#          - If False, the overlay is saved to the mini_path and, if applicable, a duplicate is saved in the 
#            "MIR_Aligned" folder.
    
#     Parameters:
#        df (pandas.DataFrame): DataFrame containing a 'path' column with session directories.
#        mapping_json_path (str): Path to the mapping JSON file (keys are mini_path).
#        threshold (int, optional): Maximum allowed time_diff for saving the duplicate mapped copy. Defaults to 10.
#        display_only (bool, optional): If True, displays the overlay (while printing intended save paths);
#                                       if False, saves the overlay image(s).
#     """
#     # Load the mapping JSON file
#     try:
#         with open(mapping_json_path, 'r') as f:
#             mapping = json.load(f)
#     except Exception as e:
#         print(f"Error reading mapping JSON file '{mapping_json_path}': {e}")
#         mapping = {}
        
#     # Create an inverse mapping: candidate rec_path -> mini_path.
#     rec_to_mini = {}
#     for mini_path, entry in mapping.items():
#         rec_path_val = entry.get("rec_path")
#         if rec_path_val:
#             rec_to_mini[rec_path_val] = mini_path

#     # Process each session from the DataFrame.
#     for idx, given_rec in df['path'].items():
#         print(f"\nProcessing session (DataFrame index {idx}): {given_rec}")
        
#         # Construct candidate rec_path using the fixed suffix.
#         candidate_rec = os.path.join(given_rec, "DANNCE", "predict00", "save_data_AVG.mat")
#         mini_path = rec_to_mini.get(candidate_rec, given_rec)
#         if rec_to_mini.get(candidate_rec):
#             print(f"Mapping found for candidate rec_path:\n  {candidate_rec}\n-> Using mini_path: {mini_path}")
#         else:
#             print(f"No mapping found for candidate {candidate_rec}; using the given path as mini_path.")
            
#         # Find netCDF files in the mini_path.
#         nc_pattern = os.path.join(mini_path, "minian_dataset*.nc")
#         nc_files = glob.glob(nc_pattern)
#         if not nc_files:
#             print(f"No netCDF files found in {mini_path} matching pattern: {nc_pattern}")
#             continue
        
#         # Choose the proper .nc file.
#         if len(nc_files) == 1:
#             selected_nc = nc_files[0]
#         else:
#             # If multiple files, prefer one containing an extra identifier (e.g., "denoised")
#             selected_nc = None
#             for fname in nc_files:
#                 if "denoised" in os.path.basename(fname):
#                     selected_nc = fname
#                     break
#             if not selected_nc:
#                 selected_nc = nc_files[0]
                
#         nc_basename = os.path.basename(selected_nc)
#         print(f"Selected netCDF file: {selected_nc}")
        
#         # Determine the naming suffix based on the file name.
#         if nc_basename == "minian_dataset.nc":
#             selection = "plot"
#         else:
#             # Remove base part to get the unique identifier.
#             selection = nc_basename.replace("minian_dataset_", "").replace(".nc", "")
        
#         # # Remove old overlay files from the mini_path.
#         # old_overlays = glob.glob(os.path.join(mini_path, "overlay_*.png"))
#         # for old_file in old_overlays:
#         #     try:
#         #         os.remove(old_file)
#         #         print(f"Removed old overlay file: {old_file}")
#         #     except Exception as e:
#         #         print(f"Error removing file {old_file}: {e}")
#         move_existing_overlays(mini_path)
        
#         # Create a temporary copy of the .nc file.
#         temp_nc = os.path.join(tempfile.gettempdir(), nc_basename)
#         try:
#             shutil.copy2(selected_nc, temp_nc)
#         except Exception as e:
#             print(f"Error copying {selected_nc} to temporary location: {e}")
#             continue
        
#         # Assume that time stamps are stored in "timeStamps.csv" in the mini_path.
#         mini_timestamps = os.path.join(mini_path, "timeStamps.csv")
#         try:
#             data, ts = load_minian_data_specific(temp_nc, mini_timestamps)
#         except Exception as e:
#             print(f"Error loading minian data from {temp_nc}: {e}")
#             os.remove(temp_nc)
#             continue
        
#         # Generate the overlay figure (this helper function should return a figure and an axis).
#         max_proj = data['max_proj'].values
#         fig, ax = overlay_all_roi_edges_no_show(data, max_proj)
        
#         # Compose a title using folder parts and the combination id.
#         try:
#             path_parts = mini_path.split(os.sep)
#             title = "_".join([path_parts[-5], path_parts[-3], path_parts[-2]])
#         except Exception:
#             title = "session"
#         # Extract combination_id from the .nc file name (if not the default, include the extra part).
#         combination_id = "" if nc_basename == "minian_dataset.nc" else selection
#         ax.set_title(f"{title}_{combination_id}")
        
#         # Determine local output file path.
#         local_output_path = os.path.join(mini_path, f"overlay_{selection}.png")
        
#         # --- Saving logic or display info ---
#         # Check if there is a mapping for the mini_path.
#         mapping_entry = mapping.get(mini_path)
#         mapped_output_path = None
#         if mapping_entry:
#             time_diff = mapping_entry.get("time_diff")
#             mapped_rec = mapping_entry.get("rec_path")
#             if time_diff is not None and time_diff < threshold and mapped_rec:
#                 dest_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(mapped_rec))), "MIR_Aligned")
#                 os.makedirs(dest_dir, exist_ok=True)
#                 mapped_output_path = os.path.join(dest_dir, f"overlay_{selection}.png")
        
#         if display_only:
#             # In display mode, just print the intended file paths.
#             print(f"(Display Mode) Would save local overlay to: {local_output_path}")
#             if mapped_output_path:
#                 print(f"(Display Mode) Would save mapped overlay to: {mapped_output_path}")
#             else:
#                 print("(Display Mode) No valid mapping found for a duplicate save.")
#             plt.show()
#         else:
#             # Save the overlay locally.
#             try:
#                 fig.savefig(local_output_path)
#                 print(f"Saved local overlay to: {local_output_path}")
#             except Exception as e:
#                 print(f"Error saving overlay to {local_output_path}: {e}")
#             # Also save the mapped duplicate if applicable.
#             if mapped_output_path:
#                 try:
#                     fig.savefig(mapped_output_path)
#                     print(f"Saved mapped overlay to: {mapped_output_path}")
#                 except Exception as e:
#                     print(f"Error saving mapped overlay to {mapped_output_path}: {e}")
#             else:
#                 print("No mapping found or mapping invalid; skipping mapped save.")
        
#         # Cleanup: close the figure and remove the temporary .nc file.
#         plt.close(fig)
#         try:
#             os.remove(temp_nc)
#         except Exception as e:
#             print(f"Error removing temporary file {temp_nc}: {e}")

#     """
#     Processes sessions from a DataFrame (with a 'path' column) that were not handled via the CSV mapping.
#     For each session directory in df['path']:
    
#       1. Constructs a candidate rec_path by appending "DANNCE/predict00/save_data_AVG.mat" to the value.
#       2. Inverts the mapping JSON (which maps mini_path -> mapping info that includes a 'rec_path')
#          to see if the candidate rec_path is found. If so, the corresponding key (mini_path) is used.
#          If no mapping entry is found, the function falls back on using the given rec_path directly.
#       3. Searches the mini_path directory for netCDF files matching "minian_dataset*.nc".
#          - If only one file is found (typically "minian_dataset.nc"), that file is used.
#          - If multiple files are found, it will choose one that contains a suffix (e.g. "denoised")
#            in its name; otherwise, it defaults to the first file in the list.
#       4. Removes any previously generated overlay PNG files in that directory.
#       5. Copies the selected netCDF file to a temporary location (to avoid file-lock issues),
#          loads it (with an assumed corresponding timeStamps.csv in the same folder),
#          and generates the overlay using your helper functions.
#       6. Sets a title based on folder components and either displays or saves the overlay.
    
#     Parameters:
#        df (pandas.DataFrame): DataFrame containing a 'path' column with session directories.
#        mapping_json_path (str): File path to the JSON file containing mapping information (mini_path keys).
#        threshold (int, optional): Maximum allowed time_diff for saving a duplicate mapped copy. Defaults to 10.
#        display_only (bool, optional): If True, overlays are shown inline; if False, they are saved to file.
#     """
#     # Load the mapping JSON
#     try:
#         with open(mapping_json_path, 'r') as f:
#             mapping = json.load(f)
#     except Exception as e:
#         print(f"Error reading mapping JSON file '{mapping_json_path}': {e}")
#         mapping = {}
        
#     # Create an inverse mapping from candidate rec_path to mini_path.
#     # (In your mapping, keys are the mini_path; mapping values include the rec_path.)
#     rec_to_mini = {}
#     for mini_path, m_entry in mapping.items():
#         rec_path_val = m_entry.get("rec_path")
#         if rec_path_val:
#             rec_to_mini[rec_path_val] = mini_path

#     # Process each session from the DataFrame.
#     for idx, given_rec in df['path'].items():
    
#         print(f"\nProcessing session from DataFrame index {idx}: {given_rec}")
        
#         # Construct the candidate rec_path by appending the fixed suffix.
#         candidate_rec = os.path.join(given_rec, "DANNCE", "predict00", "save_data_AVG.mat")
#         # Try to look up the mini_path from the mapping.
#         mini_path = rec_to_mini.get(candidate_rec)
#         if mini_path:
#             print(f"Found mapping for candidate rec_path: {candidate_rec}")
#         else:
#             print(f"No mapping entry for candidate {candidate_rec}; falling back to using the given path.")
#             # Use the provided path directly. (Adjust if needed.)
#             mini_path = given_rec
        
#         # Now search for the netCDF file in mini_path.
#         pattern = os.path.join(mini_path, "minian_dataset*.nc")
#         nc_files = glob.glob(pattern)
#         if not nc_files:
#             print(f"No netCDF files found in {mini_path} matching pattern: {pattern}")
#             continue
        
#         # If multiple .nc files exist, check for one containing 'denoised'.
#         if len(nc_files) > 1:
#             selected_nc = None
#             for fname in nc_files:
#                 if "denoised" in fname:
#                     selected_nc = fname
#                     break
#             if not selected_nc:
#                 selected_nc = nc_files[0]
#         else:
#             selected_nc = nc_files[0]
            
#         print(f"Selected netCDF file: {selected_nc}")
        
#         # Remove any old overlay PNG files in this directory.
#         old_overlays = glob.glob(os.path.join(mini_path, "overlay_*.png"))
#         for old_file in old_overlays:
#             try:
#                 os.remove(old_file)
#                 print(f"Removed old overlay file: {old_file}")
#             except Exception as e:
#                 print(f"Error removing file {old_file}: {e}")
                
#         # Copy the .nc file to a temporary location.
#         temp_nc = os.path.join(tempfile.gettempdir(), os.path.basename(selected_nc))
#         try:
#             shutil.copy2(selected_nc, temp_nc)
#         except Exception as e:
#             print(f"Error copying {selected_nc} to temporary location: {e}")
#             continue
        
#         # Assume that time stamps are stored in "timeStamps.csv" in the mini_path.
#         mini_timestamps = os.path.join(mini_path, "timeStamps.csv")
#         try:
#             data, ts = load_minian_data_specific(temp_nc, mini_timestamps)
#         except Exception as e:
#             print(f"Error loading minian data from {temp_nc}: {e}")
#             os.remove(temp_nc)
#             continue
        
#         # Create the overlay figure (helper function expected to return a fig, ax).
#         max_proj = data['max_proj'].values
#         fig, ax = overlay_all_roi_edges_no_show(data, max_proj)
        
#         # Create a title based on parts of the mini_path
#         try:
#             # Splitting on os.sep to work on any OS path separator.
#             parts = mini_path.split(os.sep)
#             title = "_".join([parts[-5], parts[-3], parts[-2]])
#         except Exception:
#             title = "session"
#         combination_id = os.path.basename(selected_nc).replace("minian_dataset", "").replace(".nc", "")
#         ax.set_title(f"{title}_{combination_id}")
        
#         # Determine output overlay file path.
#         overlay_out = os.path.join(mini_path, "overlay_plot.png")
#         if display_only:
#             plt.show()
#         else:
#             try:
#                 fig.savefig(overlay_out)
#                 print(f"Saved overlay to {overlay_out}")
#             except Exception as e:
#                 print(f"Error saving overlay to {overlay_out}: {e}")
        
#         # Cleanup: close the figure and remove the temporary .nc file.
#         plt.close(fig)
#         os.remove(temp_nc)

def vis_param_all_with_mapping(csv_file_path, mapping_json_path, threshold=10, display_only=True):
    """
    Processes every session defined in the CSV file and generates overlay images.
    
    对于 CSV 中的每一行（包含 "rec_path" 和 "nc_file" 两列）：
      1. 在 session 目录中查找带有指定 selection 标识的 .nc 文件。
      2. 如果找到了，就生成叠加图（overlay）。
      3. 如果找不到，但目录下存在默认的 minian_dataset.nc，则使用它，输出文件名为 overlay.png。
      4. 如果 session 目录里已经有旧的 overlay 文件，会先把它们移动到子文件夹 "prev_overlays" 中，只留下最新生成的。
      5. 根据 display_only 决定是直接 show 还是保存到本地。
      6. 如果 mapping JSON 中有对应条目且 time_diff < threshold，则：
         a. 在映射目录（.../MIR_Aligned）里也会先把旧的 overlay 文件移动到该目录下的 "prev_overlays" 子文件夹，
            然后再保存最新的 overlay。
    
    参数:
       csv_file_path (str): 包含 'rec_path' 和 'nc_file' 列的 CSV 文件路径。
       mapping_json_path (str): 包含映射信息的 JSON 文件路径。
       threshold (int, optional): 允许的最大 time_diff，默认 10。
       display_only (bool, optional): True 时只 inline 显示 overlay；False 时保存到磁盘。
    """
    # 1. 尝试读取 CSV
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading CSV file '{csv_file_path}': {e}")
        return

    # 2. 尝试读取 mapping JSON
    try:
        with open(mapping_json_path, 'r') as f:
            mapping = json.load(f)
    except Exception as e:
        print(f"Error reading mapping JSON file '{mapping_json_path}': {e}")
        mapping = {}

    # 3. 遍历 CSV 中的每一行
    for index, row in df.iterrows():
        session_dir = row['rec_path']
        selection = row['nc_file']
        print(f"\nProcessing row {index}: session_dir = {session_dir}, selection = {selection}")

        # 3.0 如果 session_dir 不存在，就跳过
        if not os.path.isdir(session_dir):
            print(f"Session directory does not exist: {session_dir}. Skipping.")
            continue

        # 3.1 移动 session 目录下旧的 overlay 文件到 prev_overlays 子文件夹
        prev_dir = os.path.join(session_dir, "prev_overlays")
        os.makedirs(prev_dir, exist_ok=True)
        old_overlay_files = glob.glob(os.path.join(session_dir, "overlay.png")) \
                            + glob.glob(os.path.join(session_dir, "overlay_*.png"))
        for old_file in old_overlay_files:
            try:
                dest = os.path.join(prev_dir, os.path.basename(old_file))
                shutil.move(old_file, dest)
                print(f"Moved old overlay file to prev_overlays: {old_file} -> {dest}")
            except Exception as e:
                print(f"Error moving {old_file} to prev_overlays: {e}")

        # 3.2 首先查找匹配 CSV 中 nc_file 标识的 .nc 文件
        pattern = os.path.join(session_dir, f"*{selection}*.nc")
        nc_files = glob.glob(pattern)
        default_mode = False

        # 3.3 如果没有找到，则尝试使用默认的 minian_dataset.nc
        if not nc_files:
            default_nc = os.path.join(session_dir, "minian_dataset.nc")
            if os.path.exists(default_nc):
                nc_files = [default_nc]
                default_mode = True
                print(f"No matching .nc for '{selection}' found. Using default file: {default_nc}")
            else:
                print(f"No .nc files matching '{pattern}' and no default minian_dataset.nc found in {session_dir}. Skipping.")
                continue
        else:
            print(f"Found {len(nc_files)} matching .nc file(s) for session: {session_dir}")

        # 3.4 依次处理每个找到的 .nc 文件
        for nc_file in nc_files:
            print("Processing file:", nc_file)
            temp_nc = os.path.join(tempfile.gettempdir(), os.path.basename(nc_file))
            try:
                shutil.copy2(nc_file, temp_nc)
            except Exception as e:
                print(f"Error copying {nc_file} to temp: {e}")
                continue

            mini_timestamps = os.path.join(session_dir, 'timeStamps.csv')
            try:
                data, ts = load_minian_data_specific(temp_nc, mini_timestamps)
            except Exception as e:
                print(f"Error loading data from {temp_nc}: {e}")
                try:
                    os.remove(temp_nc)
                except:
                    pass
                continue

            max_proj = data['max_proj'].values
            fig, ax = overlay_all_roi_edges_no_show(data, max_proj)

            # 3.5 根据 default_mode 确定输出文件名和标题
            if default_mode:
                local_filename = "overlay.png"
                title = os.path.basename(session_dir)
                ax.set_title(f"{title}")
            else:
                combination_id = os.path.basename(nc_file).replace("minian_dataset_", "").replace(".nc", "")
                try:
                    title = "_".join([
                        session_dir.split("/")[-5],
                        session_dir.split("/")[-3],
                        session_dir.split("/")[-2]
                    ])
                except IndexError:
                    title = "session"
                ax.set_title(f"{title}_{combination_id}")
                local_filename = f"overlay_{selection}.png"

            local_output_path = os.path.join(session_dir, local_filename)

            if display_only:
                plt.show()
                print(f"Displayed overlay for: {nc_file}")
            else:
                try:
                    fig.savefig(local_output_path)
                    print(f"Saved overlay locally to: {local_output_path}")
                except Exception as e:
                    print(f"Error saving overlay for {nc_file} to {local_output_path}: {e}")

            # 3.6 如果 mapping 存在且满足条件，在映射目录里先清理旧 overlay 再保存一份
            mapping_entry = mapping.get(session_dir)
            if mapping_entry:
                time_diff = mapping_entry.get("time_diff")
                mapped_rec = mapping_entry.get("rec_path")
                if time_diff is not None and time_diff < threshold and mapped_rec:
                    # 先确定映射目录下的 MIR_Aligned 路径
                    dest_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(mapped_rec))), "MIR_Aligned")
                    try:
                        # 创建 MIR_Aligned 文件夹
                        os.makedirs(dest_dir, exist_ok=True)

                        # 在映射目录下，把旧的 overlay 文件移动到 dest_dir/prev_overlays
                        dest_prev = os.path.join(dest_dir, "prev_overlays")
                        os.makedirs(dest_prev, exist_ok=True)
                        old_mapped_overlays = glob.glob(os.path.join(dest_dir, "overlay.png")) \
                                              + glob.glob(os.path.join(dest_dir, "overlay_*.png"))
                        for old_map in old_mapped_overlays:
                            try:
                                moved = os.path.join(dest_prev, os.path.basename(old_map))
                                shutil.move(old_map, moved)
                                print(f"Moved mapped old overlay to prev_overlays: {old_map} -> {moved}")
                            except Exception as e:
                                print(f"Error moving mapped {old_map}: {e}")

                        # 保存新的 overlay 到映射目录
                        mapped_output_path = os.path.join(dest_dir, local_filename)
                        fig.savefig(mapped_output_path)
                        print(f"Saved overlay into mapped directory: {mapped_output_path}")
                    except Exception as e:
                        print(f"Error handling mapped directory '{dest_dir}': {e}")
                else:
                    print("Mapping found but time_diff 太高或 rec_path 缺失；未保存映射副本。")
            else:
                print("No mapping found for session; skipping mapped save.")

            # 3.7 清理：关闭 figure 并删除临时 nc 文件
            plt.close(fig)
            try:
                os.remove(temp_nc)
            except Exception as e:
                print(f"Error removing temp file {temp_nc}: {e}")