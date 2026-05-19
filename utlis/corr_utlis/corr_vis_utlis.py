import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import zscore

# ------------------------------------------------------------------------------
# Optional Utility Function: Load Exclusion Dictionary from JSON file
# ------------------------------------------------------------------------------
def load_exclude_dict(json_path):
    """
    Load a JSON file containing the exclusion dictionary.

    Parameters:
      - json_path: Path to the exclusion JSON file.

    Returns:
      - Dictionary with the exclusion mapping.
    """
    with open(json_path, 'r') as f:
        return json.load(f)

# ------------------------------------------------------------------------------
# 1. Neuron Exclusion Lookup Function
# ------------------------------------------------------------------------------
def get_excluded_neurons_for_session(session_path, exclude_dict):
    """
    Given a session path and the exclusion dictionary, return the list of excluded neuron indices.
    If an exact match is not found, try to match by substring.
    """
    if session_path in exclude_dict:
        return exclude_dict[session_path]
    for key, val in exclude_dict.items():
        if key in session_path:
            return val
    return []  # Default to no exclusions if not found

# ------------------------------------------------------------------------------
# 2. Data Loading with Dynamic HDF5 File Search
# ------------------------------------------------------------------------------
def load_session_data(rec_path):
    """
    Load a single session's HDF5 file by dynamically searching the MIR_Aligned folder.
    
    If only one HDF5 file is present (typically named exactly
    'aligned_predictions_with_ca_and_dF_F.h5'), it is used.
    
    If more than one file exists, the function will look for one that contains
    additional identifying information (e.g., with 'wnd1500' in the filename).
    If found, that file is used; otherwise, the first file is chosen.

    The function extracts metadata from the folder structure and appends it
    to the DataFrame.
    """
    mir_aligned_path = os.path.join(rec_path, 'MIR_Aligned')
    # Search for all HDF5 files in the MIR_Aligned folder
    hdf5_files = glob.glob(os.path.join(mir_aligned_path, '*.h5'))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {mir_aligned_path}")
    if len(hdf5_files) == 1:
        hdf5_file_path = hdf5_files[0]
    else:
        # Look for a file that contains 'wnd1500' in its name
        matching_files = [f for f in hdf5_files if 'wnd1500' in os.path.basename(f)]
        if matching_files:
            hdf5_file_path = matching_files[0]
        else:
            hdf5_file_path = hdf5_files[0]  # fallback to first file if no match found

    df = pd.read_hdf(hdf5_file_path, key='df')
    
    # Extract metadata based on the folder structure:
    # e.g., /data/big_rim/rsync_dcc_sum/Oct3V1/2024_10_25/20241002PMCr2_15_42
    norm_path = os.path.normpath(rec_path)
    session_id = os.path.basename(norm_path)
    recording_date = os.path.basename(os.path.dirname(norm_path))
    experiment_name = os.path.basename(os.path.dirname(os.path.dirname(norm_path)))
    
    df['session_id'] = session_id
    df['recording_date'] = recording_date
    df['experiment'] = experiment_name
    df['session_path'] = rec_path
    df['file_path'] = hdf5_file_path
    
    return df

# ------------------------------------------------------------------------------
# 3. Processing Neuron Activity Data with Optional Exclusion
# ------------------------------------------------------------------------------
def process_neuron_activity(df, exclude_dict=None, manual_exclude_indices=None, apply_exclusion=True):
    """
    Process neuron activity data:
      - Optionally exclude specific neurons.
      - Remove low-variance neurons.
      - Z-score normalize each neuron's time course.
    
    Parameters:
      - df: DataFrame with neuron activity and metadata.
      - exclude_dict: Dictionary with neuron exclusions (optional).
      - manual_exclude_indices: List of neuron indices to exclude manually (optional).
      - apply_exclusion: If False, skip neuron exclusion.
    
    Returns:
      - neuron_activity_normalized: np.array of shape (neurons, timepoints)
      - filtered_neuron_columns: list of neuron column names after filtering
      - df_new: reset-index DataFrame used for plotting (should contain timestamp info)
    """
    # Determine which neurons to exclude
    if apply_exclusion:
        if manual_exclude_indices is not None:
            excluded_indices = manual_exclude_indices
        elif exclude_dict is not None:
            session_path = df['session_path'].iloc[0]
            excluded_indices = get_excluded_neurons_for_session(session_path, exclude_dict)
        else:
            excluded_indices = []
    else:
        excluded_indices = []
    
    # Build the list of column names to exclude based on the neuron indices
    manual_excluded_neurons = [f'dF_F_roi{i}' for i in excluded_indices]
    
    # Select neuron columns that start with 'dF_F_roi' and are not in the excluded list.
    neuron_columns = [
        col for col in df.columns 
        if col.startswith('dF_F_roi') and col not in manual_excluded_neurons
    ]
    
    # Extract neuron activity data and transpose to shape (neurons, timepoints)
    neuron_activity = df[neuron_columns].values.T
    
    # Remove low-variance neurons (keep top 95% variance)
    neuron_variances = np.var(neuron_activity, axis=1)
    threshold = np.percentile(neuron_variances, 5)
    high_variance_indices = neuron_variances > threshold
    neuron_activity_filtered = neuron_activity[high_variance_indices, :]
    filtered_neuron_columns = [col for i, col in enumerate(neuron_columns) if high_variance_indices[i]]
    
    # Z-score normalization along each neuron's time course
    neuron_activity_normalized = zscore(neuron_activity_filtered, axis=1)
    
    # Reset index for plotting convenience (assumes a 'timestamp_ms_mini' column exists)
    df_new = df.reset_index()
    
    return neuron_activity_normalized, filtered_neuron_columns, df_new

# ------------------------------------------------------------------------------
# 4. Plotting Function with Optional Saving
# ------------------------------------------------------------------------------
def plot_session_data(df, exclude_dict=None, manual_exclude_indices=None, apply_exclusion=True, save_filename=None):
    """
    Produce a figure with:
      - A heatmap of hierarchically clustered neuron activity.
      - Center of mass (COM) trajectories.
    
    The overall figure title (suptitle) includes the session name and a formatted HDF5 filename.
    For an HDF5 filename starting with "aligned_predictions_with_ca_and_dF_F_", only the part after the prefix is displayed.
    
    Parameters:
      - df: DataFrame with session data.
      - exclude_dict: Dictionary for neuron exclusions (optional).
      - manual_exclude_indices: List of neuron indices to exclude manually (optional).
      - apply_exclusion: If False, plot all neurons without exclusion.
      - save_filename: Optional filename (with path) to save the plot.
    
    Expects the DataFrame to have a 'timestamp_ms_mini' column for the x-axis.
    """
    # Process neuron activity data
    neuron_activity_normalized, filtered_neuron_columns, df_new = process_neuron_activity(
        df, exclude_dict=exclude_dict, manual_exclude_indices=manual_exclude_indices, apply_exclusion=apply_exclusion
    )
    time = df_new['timestamp_ms_mini']
    
    # Create the figure and layout
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[2, 1, 1.5])
    
    # --- Build title information from the DataFrame ---
    # Get session id from the folder path (e.g., "20241002PMCr2_15_42")
    session_id = df['session_id'].iloc[0]
    
    # Extract the .h5 file name from the full file_path
    h5_filepath = df['file_path'].iloc[0]
    h5_filename = os.path.basename(h5_filepath)
    base, ext = os.path.splitext(h5_filename)
    
    # Process the h5 file info: if the base name starts with the standard prefix,
    # extract only the dynamic part that follows.
    if base == 'aligned_predictions_with_ca_and_dF_F':
        file_info = base
    elif base.startswith('aligned_predictions_with_ca_and_dF_F_'):
        file_info = base.split('aligned_predictions_with_ca_and_dF_F_')[1]
    else:
        file_info = base

    # Add a suptitle to the figure with both the session and file information.
    fig.suptitle(f"Session: {session_id} | File: {file_info}", fontsize=16, y=0.98)
    
    # --- Plot 1: Ca²⁺ Heatmap of Clustered Neuron Activity ---
    ax1 = fig.add_subplot(gs[0, 0])
    if neuron_activity_normalized is not None and filtered_neuron_columns is not None:
        # Hierarchical clustering using Ward linkage
        Z = linkage(neuron_activity_normalized, method='ward')
        neuron_order = leaves_list(Z)
        neuron_activity_ordered = neuron_activity_normalized[neuron_order, :]
        ordered_neuron_columns = [filtered_neuron_columns[i] for i in neuron_order]
        
        neuron_indices = np.arange(len(neuron_activity_ordered))
        ax1.pcolormesh(
            time, neuron_indices, neuron_activity_ordered,
            cmap='viridis', shading='auto'
        )
        ax1.set_title('Hierarchically Clustered Neuron Activity (Filtered & Z-scored)')
        ax1.set_ylabel('Neurons (clustered)')
    else:
        ax1.text(0.5, 0.5, 'No neuron activity data available.',
                 transform=ax1.transAxes, ha='center', va='center')
        ax1.set_title('No Neuron Activity Plot')
    ax1.tick_params(labelbottom=False)
    
    # --- Plot 2: Center of Mass (COM) Trajectories ---
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    if {'com_x', 'com_y', 'com_z'}.issubset(df_new.columns):
        ax2.plot(time, df_new['com_x'], color='red', label='COM X')
        ax2.plot(time, df_new['com_y'], color='green', label='COM Y')
        ax2.set_ylabel('COM (mm)')
        ax2.set_title('Center of Mass Trajectories')
        
        # Secondary y-axis for COM Z
        ax2_right = ax2.twinx()
        ax2_right.plot(time, df_new['com_z'], color='blue', label='COM Z')
        ax2_right.set_ylabel('COM Z (mm)', color='blue')
        ax2_right.tick_params(axis='y', labelcolor='blue')
        
        # Combine legends from both axes
        lines_ax2, labels_ax2 = ax2.get_legend_handles_labels()
        lines_ax2_right, labels_ax2_right = ax2_right.get_legend_handles_labels()
        ax2.legend(lines_ax2 + lines_ax2_right, labels_ax2 + labels_ax2_right, loc='upper right')
    else:
        print("COM columns not found (com_x, com_y, com_z). Check your DataFrame.")
    ax2.tick_params(labelbottom=False)
    
    # --- Additional plotting sections can be added here if needed ---
    plt.tight_layout()
    
    # Save the figure if a filename is provided
    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight')
    plt.show()


# ------------------------------------------------------------------------------
# 5. Wrapper Function to Run and Save a Single Session Plot
# ------------------------------------------------------------------------------
def run_session_plot(session_path, exclude_dict=None, manual_exclude_indices=None, apply_exclusion=True, save_plot=True):
    """
    Loads a single session's data, processes it, creates the plot,
    and then displays and optionally saves the plot.
    
    Parameters:
      - session_path: Path to the session folder.
      - exclude_dict: Dictionary with neuron exclusions.
      - manual_exclude_indices: Optional list of neuron indices to exclude manually.
      - apply_exclusion: If False, all neurons are plotted.
      - save_plot: If True, the plot will be saved to disk. The filename is derived from the HDF5 file name and saved in the same directory as that file.
    """
    df_session = load_session_data(session_path)
    if save_plot:
        h5_filepath = df_session['file_path'].iloc[0]
        h5_dir = os.path.dirname(h5_filepath)
        h5_filename = os.path.basename(h5_filepath)
        base, ext = os.path.splitext(h5_filename)
        # If the HDF5 file name is the standard one
        if base == 'aligned_predictions_with_ca_and_dF_F':
            filename = f"session_plot_{base}.png"
        # If it starts with the standard name but contains extra identifying info,
        # then use a '*' to indicate the variable (middle) part.
        elif base.startswith('aligned_predictions_with_ca_and_dF_F'):
            filename = "session_plot_aligned_predictions_with_ca_and_dF_F_*.png"
        else:
            # Fall back to using the session id if the file name doesn't match expected patterns.
            session_id = df_session['session_id'].iloc[0]
            filename = f"session_plot_{session_id}.png"
        # Save plot in the same directory as the .h5 file
        save_filename = os.path.join(h5_dir, filename)
    else:
        save_filename = None

    plot_session_data(df_session,
                      exclude_dict=exclude_dict,
                      manual_exclude_indices=manual_exclude_indices,
                      apply_exclusion=apply_exclusion,
                      save_filename=save_filename)

# ------------------------------------------------------------------------------
# 6. Example Usage for a Single Session
# ------------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Specify your single session folder path here.
#     session_path = '/data/big_rim/rsync_dcc_sum/Oct3V1/2024_10_25/20241002PMCr2_15_42'
    
#     # Option 1: If you want to load the exclusion dictionary from a JSON file:
#     # exclude_json_path = '/path/to/neuro_exclude.json'
#     # exclude_dict = load_exclude_dict(exclude_json_path)
    
#     # Option 2: If you already have an exclusion dictionary, simply pass it in.
#     # For example, if you don't need to exclude any neurons, you can pass None.
#     exclude_dict = None  # or use your loaded exclusion dictionary
    
#     # Call the wrapper function: this will load the data, plot it, and save the figure if enabled.
#     run_session_plot(session_path, exclude_dict)
