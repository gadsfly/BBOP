import os
import glob
import shutil
import tempfile
import matplotlib.pyplot as plt


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
    nc_files = glob.glob(os.path.join(session_dir, "minian_dataset_*.nc"))

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
        title = "_".join(session_dir.split("/")[-5:-2])
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




        