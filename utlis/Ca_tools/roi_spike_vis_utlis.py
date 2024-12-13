import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from scipy.sparse import csc_matrix
import xarray as xr
import glob

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


