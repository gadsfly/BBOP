# below are not validated.. at all... so 
# import numpy as np
# import os
# import matplotlib.pyplot as plt

# def calculate_distances(marker1_id, marker2_id, pts_location):
#     """
#     Calculate the Euclidean distances between two specified markers across all frames.

#     Parameters:
#     - marker1_id: int, ID of the first marker.
#     - marker2_id: int, ID of the second marker.
#     - pts_location: numpy array of shape (n_frames, 3, n_markers), the 3D positions of markers.

#     Returns:
#     - distances: numpy array of shape (n_frames,), distances between the two markers for each frame.
#     """
#     # Extract the positions of the two markers
#     marker1_positions = pts_location[:, :, marker1_id]  # Shape: (n_frames, 3)
#     marker2_positions = pts_location[:, :, marker2_id]  # Shape: (n_frames, 3)

#     # Check for NaN values and create a mask
#     valid_mask = ~np.isnan(marker1_positions[:, 0]) & ~np.isnan(marker2_positions[:, 0])

#     # Compute distances only for valid frames
#     distances = np.full(pts_location.shape[0], np.nan)  # Initialize with NaN
#     valid_distances = np.linalg.norm(marker1_positions[valid_mask] - marker2_positions[valid_mask], axis=1)
#     distances[valid_mask] = valid_distances

#     return distances




# def analyze_segment_lengths(prediction_data, ground_truth_means, ground_truth_stds, marker_pairs, labels, save_path, title):
#     """
#     Analyze segment lengths by calculating distances between marker pairs, computing mean and std,
#     and plotting comparison with ground truth.

#     Parameters:
#     - prediction_data: numpy array of shape (n_frames, 3, n_markers), the predicted 3D positions.
#     - ground_truth_means: list of float, ground truth average distances for each segment.
#     - ground_truth_stds: list of float, ground truth standard deviations for each segment.
#     - marker_pairs: list of tuples, each containing two marker IDs (marker1_id, marker2_id).
#     - labels: list of strings, labels for each segment.
#     - save_path: string, path to save the plot.
#     - title: string, title for the plot.

#     Returns:
#     - None
#     """
#     num_segments = len(marker_pairs)
#     pred_means = []
#     pred_stds = []

#     # Calculate distances and statistics for each segment
#     for marker1_id, marker2_id in marker_pairs:
#         distances = calculate_distances(marker1_id, marker2_id, prediction_data)
#         valid_distances = distances[~np.isnan(distances)]
#         pred_means.append(np.mean(valid_distances))
#         pred_stds.append(np.std(valid_distances))

#     # Plotting
#     total_width, n = 0.5, 2  # Total bar width and number of bars per group
#     x = np.arange(num_segments)
#     width = total_width / n
#     x_offsets = x - (total_width - width) / 2

#     plt.figure(figsize=(10, 6))
#     plt.bar(x_offsets, ground_truth_means, width=width, yerr=ground_truth_stds, label='Ground Truth', capsize=5)
#     plt.bar(x_offsets + width, pred_means, width=width, yerr=pred_stds, label='Prediction', capsize=5)
#     plt.xticks(x, labels)
#     plt.ylabel('Distance (mm)')
#     plt.title(title)
#     plt.legend(loc='best')
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.show()
#     plt.close()
