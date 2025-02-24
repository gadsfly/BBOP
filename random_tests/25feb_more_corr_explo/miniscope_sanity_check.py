#!/usr/bin/env python3
import os
import sys
import json

# Adjust sys.path so that ROI spike visualization functions can be imported.
sys.path.append(os.path.abspath('../../'))
from utlis.Ca_tools.roi_spike_vis_utlis import load_minian_data, calculate_dff, overlay_all_roi_edges

def run_minian_sanity_check(minian_path):
    """
    Run the ROI/spike visualization sanity check for a given miniscope path.
    """
    print(f"\nProcessing miniscope path:\n  {minian_path}")
    # Construct the path to the timestamps CSV file.
    mini_timestamps = os.path.join(minian_path, 'timeStamps.csv')
    
    # Load the miniscope data.
    try:
        data, ts = load_minian_data(minian_path, mini_timestamps)
        print("  Data loaded successfully.")
    except Exception as e:
        print(f"  Error loading minian data from {minian_path}: {e}")
        return
    
    # Calculate dF/F.
    try:
        dF_F = calculate_dff(data)
        print("  dF/F calculated successfully.")
    except Exception as e:
        print(f"  Error calculating dF/F for {minian_path}: {e}")
        return
    
    # Retrieve the maximum projection and overlay ROI edges.
    try:
        max_proj = data['max_proj'].values
        overlay_all_roi_edges(data, max_proj)
        print("  ROI edges overlaid successfully.")
    except Exception as e:
        print(f"  Error overlaying ROI edges for {minian_path}: {e}")
        return

def main():
    # Path to the JSON mapping file.
    json_file = "/home/lq53/mir_repos/BBOP/random_tests/25feb_more_corr_explo/mini_to_rec_mapping.json"
    
    if not os.path.exists(json_file):
        print(f"JSON file not found: {json_file}")
        return
    
    with open(json_file, "r") as f:
        mapping_data = json.load(f)
    
    # Loop through each miniscope path in the JSON file.
    for mini_path, mapping in mapping_data.items():
        valid = mapping.get("valid")
        time_diff = mapping.get("time_diff")
        rec_path = mapping.get("rec_path")
        
        # Process only if criteria are met.
        if valid and rec_path is not None and time_diff < 10:
            run_minian_sanity_check(mini_path)
            # Pause between entries to allow manual inspection.
            input("Review the overlay. Press Enter to process the next miniscope (or Ctrl+C to abort)...")
        else:
            print(f"\nSkipping {mini_path} (valid: {valid}, time_diff: {time_diff}, rec_path: {rec_path})")
            
if __name__ == "__main__":
    main()
