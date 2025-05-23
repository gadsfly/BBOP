import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from scipy.sparse import csc_matrix
import xarray as xr
import glob
import os


def load_minian_data(minian_path, timestamps_path):
    """Load minian data and timestamps from given paths."""
    nc_files = glob.glob(f"{minian_path}/*.nc")
    if not nc_files:
        raise FileNotFoundError("No .nc files found in the specified path.")
    data = xr.open_dataset(nc_files[0])
    timestamps = pd.read_csv(timestamps_path)
    return data, timestamps['Time Stamp (ms)'].values

def calculate_dff(data):
    """Calculate ΔF/F for calcium signals."""
    C = data['C'].values
    F0 = np.percentile(C, 20, axis=1, keepdims=True)  # Calculate baseline (20th percentile)
    dF_F = (C - F0) / F0
    return dF_F

def plot_all_neurons(C, ts, title="C values for all neurons", neurons_per_plot=20):
    """Plot all neurons' calcium signals sorted by mean value."""
    num_neuron, _ = C.shape
    mean_C = np.mean(C, axis=1)
    sorted_indices = np.argsort(mean_C)[::-1]
    chunks = [sorted_indices[i:i + neurons_per_plot] for i in range(0, len(sorted_indices), neurons_per_plot)]

    for chunk_idx, chunk in enumerate(chunks):
        plt.figure(figsize=(12, 8))
        for i, neuron_index in enumerate(chunk):
            plt.plot(ts, C[neuron_index] + i * 10, color='black')
            plt.text(ts[-1], C[neuron_index][-1] + i * 10, f'Neuron {neuron_index}', color='red')
        plt.xlabel('Time Stamp (ms)')
        plt.ylabel('C values (shifted for each neuron)')
        plt.title(f'{title}: Neurons {chunk_idx * neurons_per_plot + 1} to {(chunk_idx + 1) * neurons_per_plot}')
        plt.show()
        print(f"Neuron indices for plot {chunk_idx + 1}:")
        print(chunk)

def plot_all_dff(dF_F, ts, title="ΔF/F for all neurons", neurons_per_plot=20):
    """Plot all neurons' ΔF/F signals sorted by mean value."""
    num_neuron, _ = dF_F.shape
    mean_dF_F = np.mean(dF_F, axis=1)
    sorted_indices = np.argsort(mean_dF_F)[::-1]
    chunks = [sorted_indices[i:i + neurons_per_plot] for i in range(0, len(sorted_indices), neurons_per_plot)]

    for chunk_idx, chunk in enumerate(chunks):
        plt.figure(figsize=(12, 8))
        for i, neuron_index in enumerate(chunk):
            plt.plot(ts, dF_F[neuron_index] + i * 10, color='black')
            plt.text(ts[-1], dF_F[neuron_index][-1] + i * 10, f'Neuron {neuron_index}', color='red')
        plt.xlabel('Time Stamp (ms)')
        plt.ylabel('ΔF/F (shifted for each neuron)')
        plt.title(f'{title}: Neurons {chunk_idx * neurons_per_plot + 1} to {(chunk_idx + 1) * neurons_per_plot}')
        plt.show()
        print(f"Neuron indices for plot {chunk_idx + 1}:")
        print(chunk)

def plot_calcium_signals(df, selected_neurons, title="Calcium Signals"):
    """Plot calcium signals for selected neurons."""
    plt.figure(figsize=(12, 8))
    for i, neuron_idx in enumerate(selected_neurons):
        plt.plot(df.index, df[f'calcium_roi{neuron_idx}'] + i * 20, label=f'Neuron {neuron_idx}')
    plt.xlabel('Time (s)')
    plt.ylabel('Calcium signal (shifted)')
    plt.title(title)
    plt.legend()
    plt.show()

def plot_dff(dF_F, ts, selected_neurons, title="ΔF/F for Selected Neurons"):
    """Plot ΔF/F for selected neurons."""
    fig, axes = plt.subplots(len(selected_neurons), 1, figsize=(12, 8), sharex=True)
    for i, neuron_index in enumerate(selected_neurons):
        axes[i].plot(ts, dF_F[neuron_index] * 100, label=f'Neuron {neuron_index}')
        axes[i].grid(True)
        axes[i].set_ylabel('ΔF/F (%)')
        axes[i].legend(loc='upper right')
    axes[-1].set_xlabel('Time Stamp (ms)')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

#black version
# def plot_dff(dF_F, ts, selected_neurons, title="ΔF/F for Selected Neurons"):
#     """Plot ΔF/F for selected neurons."""
#     plt.figure(figsize=(12, 8))
#     for i, neuron_idx in enumerate(selected_neurons):
#         plt.plot(ts, dF_F[neuron_idx] + i * 10, color='black', label=f'Neuron {neuron_idx}')
#         plt.text(ts[-1], dF_F[neuron_idx][-1] + i * 10, f'Neuron {neuron_idx}', color='red')
#     plt.xlabel('Time Stamp (ms)')
#     plt.ylabel('ΔF/F (shifted)')
#     plt.title(title)
#     plt.legend()
#     plt.show()

def overlay_roi_edges_exclude(data, max_proj, exclude_neurons=None):
    """
    Overlay ROI edges on the max projection image, excluding specified neurons.

    Parameters:
        data : dict
            Dictionary containing the ROI masks under key 'A'. The ROI masks can be in
            either a dense (NumPy array) or sparse (e.g., csc_matrix) format.
        max_proj : 2D array
            The max projection image.
        exclude_neurons : list, optional
            List of ROI indices to exclude from overlaying and labeling.
            If None, all neurons will be included.
    """
    if exclude_neurons is None:
        exclude_neurons = []
        
        
    # Convert data['A'] to a dense array if needed
    if hasattr(data['A'], 'values'):
        A_dense = data['A'].values
    else:
        A_dense = csc_matrix((data['A'].data, data['A'].indices, data['A'].indptr),
                             shape=data['A'].shape).toarray()
    
    overlay_edges = np.zeros_like(max_proj)
    n_rois = A_dense.shape[0]
    
    # Process each ROI from all neurons except those to be excluded
    for roi in range(n_rois):
        if roi in exclude_neurons:
            continue
        roi_mask = A_dense[roi].reshape(max_proj.shape) > 0
        roi_edge = roi_mask ^ binary_erosion(roi_mask)
        overlay_edges += roi_edge
        
    overlay_edges = np.clip(overlay_edges, 0, 1)
    
    # Plot the max projection and overlay the ROI edges
    plt.figure(figsize=(10, 10))
    plt.imshow(max_proj, interpolation='nearest')
    plt.imshow(overlay_edges, cmap='Reds', alpha=0.3, interpolation='nearest')
    
    # Add ROI labels for all neurons (except those excluded)
    for roi in range(n_rois):
        if roi in exclude_neurons:
            continue
        roi_mask = A_dense[roi].reshape(max_proj.shape) > 0
        coords = np.argwhere(roi_mask)
        if coords.size > 0:
            centroid = coords.mean(axis=0)
            plt.text(centroid[1], centroid[0] + 20, str(roi), color='blue', fontsize=12,
                     ha='center', va='center')
                     
    plt.title('Max Projection with ROI Edges Overlay (Excluding Specified Neurons)')
    plt.axis('off')
    plt.show()


def overlay_roi_edges(data, max_proj, manual_selection):
    """Overlay ROI edges on the max projection image."""
    A_dense = data['A'].values if hasattr(data['A'], 'values') else csc_matrix(
        (data['A'].data, data['A'].indices, data['A'].indptr), shape=data['A'].shape).toarray()
    overlay_edges = np.zeros_like(max_proj)
    for roi in manual_selection:
        roi_mask = A_dense[roi].reshape(max_proj.shape) > 0
        roi_edge = roi_mask ^ binary_erosion(roi_mask)
        overlay_edges += roi_edge
    overlay_edges = np.clip(overlay_edges, 0, 1)
    plt.figure(figsize=(10, 10))
    plt.imshow(max_proj, interpolation='nearest')
    plt.imshow(overlay_edges, cmap='Reds', alpha=0.3, interpolation='nearest')
    for roi in manual_selection:
        roi_mask = A_dense[roi].reshape(max_proj.shape) > 0
        coords = np.argwhere(roi_mask)
        if coords.size > 0:
            centroid = coords.mean(axis=0)
            plt.text(centroid[1], centroid[0] + 20, str(roi), color='blue', fontsize=12, ha='center', va='center')
    plt.title('Max Projection with ROI Edges Overlay')
    plt.axis('off')
    plt.show()

def overlay_all_roi_edges(data, max_proj):
    """Overlay edges of all ROIs on the max projection image."""
    # Extract the dense array for ROIs
    A_dense = data['A'].values if hasattr(data['A'], 'values') else csc_matrix(
        (data['A'].data, data['A'].indices, data['A'].indptr), shape=data['A'].shape).toarray()
    
    overlay_edges = np.zeros_like(max_proj)
    
    # Iterate through all ROIs
    for roi in range(A_dense.shape[0]):  # Assuming each row in A_dense corresponds to a neuron
        roi_mask = A_dense[roi].reshape(max_proj.shape) > 0
        roi_edge = roi_mask ^ binary_erosion(roi_mask)
        overlay_edges += roi_edge
    
    overlay_edges = np.clip(overlay_edges, 0, 1)
    
    # Plot the results
    plt.figure(figsize=(10, 10))
    plt.imshow(max_proj, interpolation='nearest')
    plt.imshow(overlay_edges, cmap='Reds', alpha=0.3, interpolation='nearest')
    
    # Optionally label centroids
    for roi in range(A_dense.shape[0]):
        roi_mask = A_dense[roi].reshape(max_proj.shape) > 0
        coords = np.argwhere(roi_mask)
        if coords.size > 0:
            centroid = coords.mean(axis=0)
            plt.text(centroid[1], centroid[0] + 20, str(roi), color='blue', fontsize=12, ha='center', va='center')
    
    plt.title('Max Projection with All ROI Edges Overlay')
    plt.axis('off')
    plt.show()

def load_minian_data_specific(nc_path, timestamps_path):
    """
    Load minian data from a specific NetCDF file path, and load timestamps from a CSV path.
    
    Parameters
    ----------
    nc_path : str
        Full path to the .nc file to open.
    timestamps_path : str
        Path to the CSV file containing the time stamps.
    
    Returns
    -------
    data : xarray.Dataset
        The loaded dataset from the .nc file.
    timestamps : np.ndarray
        The 'Time Stamp (ms)' column from the timestamps CSV as a NumPy array.
    """
    if not os.path.isfile(nc_path):
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")
    if not os.path.isfile(timestamps_path):
        raise FileNotFoundError(f"Timestamps CSV not found: {timestamps_path}")
    
    # Open the dataset and load it fully into memory
    data = xr.open_dataset(nc_path)
    data = data.load()
    
    # Load timestamps from CSV
    timestamps_df = pd.read_csv(timestamps_path)
    if 'Time Stamp (ms)' not in timestamps_df.columns:
        raise ValueError("Expected column 'Time Stamp (ms)' in timestamps CSV.")
    
    return data, timestamps_df['Time Stamp (ms)'].values

def overlay_all_roi_edges_no_show(data, max_proj):
    """
    Overlay edges of all ROIs on the max projection image without calling plt.show().
    Returns the figure and axes so you can display or save as needed.
    
    Parameters
    ----------
    data : xarray.Dataset
        Must contain 'A': either a dense array [n_rois, width*height] or a sparse csc.
    max_proj : np.ndarray
        2D array (height, width) representing the max projection image.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
        The created figure and axes.
    """
    # 1) Convert 'A' to a dense array if it's sparse.
    A_var = data['A']
    if hasattr(A_var, 'values'):
        A_dense = A_var.values
    else:
        from scipy.sparse import csc_matrix
        A_dense = csc_matrix((A_var.data, A_var.indices, A_var.indptr),
                             shape=A_var.shape).toarray()
    
    # 2) Prepare an empty overlay array (same shape as max_proj).
    overlay_edges = np.zeros_like(max_proj, dtype=np.uint8)
    
    # 3) For each ROI, extract the mask, compute its edge, and add to overlay.
    n_rois = A_dense.shape[0]
    for roi in range(n_rois):
        roi_mask = A_dense[roi].reshape(max_proj.shape) > 0
        roi_edge = roi_mask ^ binary_erosion(roi_mask)
        overlay_edges += roi_edge.astype(np.uint8)
    
    # Clip to avoid intensities > 1
    overlay_edges = np.clip(overlay_edges, 0, 1)
    
    # 4) Create a new figure and axes, and plot the max projection.
    fig, ax = plt.subplots(figsize=(10, 10))
    # Using 'viridis' instead of 'gray'
    ax.imshow(max_proj, interpolation='nearest', cmap='viridis')
    
    # 5) Overlay the ROI edges in red with partial transparency.
    ax.imshow(overlay_edges, cmap='Reds', alpha=0.3, interpolation='nearest')
    
    # 6) Optionally label ROI centroids.
    for roi in range(n_rois):
        roi_mask = A_dense[roi].reshape(max_proj.shape) > 0
        coords = np.argwhere(roi_mask)
        if coords.size > 0:
            centroid = coords.mean(axis=0)
            ax.text(
                centroid[1], centroid[0] + 20, str(roi),
                color='blue', fontsize=12, ha='center', va='center'
            )
    
    # 7) Final styling.
    ax.set_title('Max Projection with All ROI Edges Overlay')
    ax.axis('off')
    
    # 8) Return the figure and axes.
    return fig, ax




def load_session_data(miniscope_folder: str, tolerance_ms: int = 5) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      - roi_0 … roi_{R‑1}  (ΔF/F for each ROI)
      - time_ms             (common timestamp)
      - qw, qx, qy, qz      (head–orientation quaternion)
    """
    # 1) Head orientation
    head_csv = os.path.join(miniscope_folder, 'headOrientation.csv')
    head = pd.read_csv(head_csv).rename(columns={'Time Stamp (ms)': 'time_ms'})

    # 2) Ca data + timestamps
    time_csv = os.path.join(miniscope_folder, 'timeStamps.csv')
    data, ts = load_minian_data(miniscope_folder, time_csv)

    # 3) ΔF/F (shape = [n_rois, n_frames])
    dF_F = calculate_dff(data)

    # 4) Build Ca DataFrame (transpose so rows = frames)
    n_rois, n_frames = dF_F.shape
    ca_df = pd.DataFrame(
        dF_F.T,
        columns=[f'roi_{i}' for i in range(n_rois)]
    )
    ca_df['time_ms'] = ts  # now len(ts)==n_frames matches ca_df

    # 5) Align via nearest‑neighbor merge
    # aligned = pd.merge_asof(
    #     ca_df.sort_values('time_ms'),
    #     head.sort_values('time_ms'),
    #     on='time_ms',
    #     direction='nearest',
    #     tolerance=tolerance_ms
    # )

    # new: exact‐timestamp join
    aligned = pd.merge(
        ca_df,
        head[['time_ms','qw','qx','qy','qz']],
        on='time_ms',
        how='inner'
    )


    return aligned


def visualize_session(miniscope_folder: str):
    # 1) load & align
    df = load_session_data(miniscope_folder, tolerance_ms=0)

    # 2) pull raw A & max projection
    data, _ = load_minian_data(miniscope_folder,
                               os.path.join(miniscope_folder, 'timeStamps.csv'))
    max_proj = data['max_proj'].values

    # 3) overlay & show
    overlay_all_roi_edges(data, max_proj)

    return df
