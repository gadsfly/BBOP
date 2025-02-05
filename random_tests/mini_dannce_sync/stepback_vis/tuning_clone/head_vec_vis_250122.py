import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import os

# Define your HDF5 file path
# hdf5_file_path = "C:/Users/shiny/Desktop/ShinySw/School/MouseProject/mice_winder/aligned_predictions_with_ca_and_dF_F.h5"
rec_path = '/data/big_rim/rsync_dcc_sum/Oct3V1/2024_10_25/20241002PMCr2_17_05'

hdf5_file_path = os.path.join(rec_path, 'MIR_Aligned/aligned_predictions_with_ca_and_dF_F.h5')
# Read the HDF5 file into a DataFrame
df_merged_with_dF_F = pd.read_hdf(hdf5_file_path, key='df')

# Define head keypoints
head_keypoints = [1, 2, 3, 4]  # EarL, EarR, Snout, SpineF

# Extract the relevant coordinates
head_coords = df_merged_with_dF_F[[f"kp{idx}_{axis}" for idx in head_keypoints for axis in ['x', 'y', 'z']]].copy()

# Define labels for better readability
keypoint_labels = {
    1: 'EarL',
    2: 'EarR',
    3: 'Snout',
    4: 'SpineF'
}

def draw_arc(ax, center, radius, start_angle, end_angle, axis, color='g', label=None):
    """Draws a 3D arc in the specified plane."""
    # Generate points along the arc
    angles = np.linspace(np.radians(start_angle), np.radians(end_angle), 100)
    if axis == 'xy':
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        z = np.full_like(x, center[2])  # Fixed Z-coordinate
    elif axis == 'xz':
        x = center[0] + radius * np.cos(angles)
        z = center[2] + radius * np.sin(angles)
        y = np.full_like(x, center[1])  # Fixed Y-coordinate
    elif axis == 'yz':
        y = center[1] + radius * np.cos(angles)
        z = center[2] + radius * np.sin(angles)
        x = np.full_like(y, center[0])  # Fixed X-coordinate
    else:
        raise ValueError("Axis must be one of 'xy', 'xz', or 'yz'.")

    # Plot the arc
    ax.plot(x, y, z, color=color, label=label, linewidth=2)


# Extract the first timestamp
first_timestamp = head_coords.index[10]
print(f"First Timestamp: {first_timestamp}")

# Extract coordinates for the first timestamp
coords = head_coords.loc[first_timestamp]

# Extract coordinates for the keypoints
earl = np.array([coords["kp1_x"], coords["kp1_y"], coords["kp1_z"]])
earr = np.array([coords["kp2_x"], coords["kp2_y"], coords["kp2_z"]])
snout = np.array([coords["kp3_x"], coords["kp3_y"], coords["kp3_z"]])

# Compute vectors in the plane
v1 = earr - earl  # Vector from EarL to EarR
v2 = snout - earl  # Vector from EarL to Snout

# Compute the normal vector to the plane
normal = np.cross(v1, v2)
normal_norm = np.linalg.norm(normal)
if normal_norm == 0:
    raise ValueError("The three keypoints are colinear; cannot define a plane.")
normal /= normal_norm  # Normalize the normal vector

# Define the local coordinate system
x_local = v1 / np.linalg.norm(v1)  # Local X-axis (EarL to EarR)
z_local = normal  # Local Z-axis (normal to the plane)
y_local = np.cross(z_local, x_local)  # Local Y-axis

# Create the rotation matrix from local to global axes
rotation_matrix = np.vstack([x_local, y_local, z_local]).T  # Columns are local axes in global coordinates

# Create a Rotation object
rotation = R.from_matrix(rotation_matrix)

# Extract Euler angles (Yaw, Pitch, Roll) in degrees using 'xyz' order
euler_angles = rotation.as_euler('xyz', degrees=True)  # [roll, pitch, yaw]

roll, pitch, yaw = euler_angles
print(f"Yaw: {yaw:.2f}°")
print(f"Pitch: {pitch:.2f}°")
print(f"Roll: {roll:.2f}°")

# Prepare lists of x, y, z coordinates for keypoints
keypoints = [1, 2, 3, 4]  # Corresponding to kp1 to kp4
x = [coords[f"kp{kp}_x"] for kp in keypoints]
y = [coords[f"kp{kp}_y"] for kp in keypoints]
z = [coords[f"kp{kp}_z"] for kp in keypoints]

# Calculate the midpoint between EarL (kp1) and EarR (kp2)
midpoint_x = (coords["kp1_x"] + coords["kp2_x"]) / 2
midpoint_y = (coords["kp1_y"] + coords["kp2_y"]) / 2
midpoint_z = (coords["kp1_z"] + coords["kp2_z"]) / 2

midpoint = (midpoint_x, midpoint_y, midpoint_z)
print(f"Midpoint of EarL and EarR: {midpoint}")

# Coordinates of Snout (kp3) already defined as 'snout'

# Calculate vector components from midpoint to Snout
vector_x = snout[0] - midpoint_x
vector_y = snout[1] - midpoint_y
vector_z = snout[2] - midpoint_z

vector = (vector_x, vector_y, vector_z)
print(f"Vector from Midpoint to Snout: {vector}")

# Calculate the magnitude of the vector
vector_length = np.sqrt(vector_x**2 + vector_y**2 + vector_z**2)
print(f"Vector Length: {vector_length:.2f}")

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of keypoints
ax.scatter(x, y, z, c='r', marker='o', s=100, label='Keypoints')

# Annotate each keypoint
for i, kp in enumerate(keypoints):
    ax.text(x[i], y[i], z[i], keypoint_labels[kp], size=12, zorder=1, color='k') 

# Remove the blue lines by not plotting connections

# Plot the midpoint
ax.scatter(midpoint_x, midpoint_y, midpoint_z, c='g', marker='^', s=100, label='Midpoint')

# Annotate the midpoint
ax.text(midpoint_x, midpoint_y, midpoint_z, 'Midpoint', size=12, zorder=1, color='k') 

# Draw the vector from midpoint to Snout using quiver
ax.quiver(
    midpoint_x, midpoint_y, midpoint_z,  # Starting point
    vector_x, vector_y, vector_z,          # Vector components
    color='m',                            # Vector color
    linewidth=2,
    arrow_length_ratio=0.1,               # Size of the arrow head
    label='Midpoint to Snout Vector'
)

# Annotate the Euler angles on the plot
# Position the text slightly above the midpoint for visibility
annotation_text = f"Yaw: {yaw:.2f}°\nPitch: {pitch:.2f}°\nRoll: {roll:.2f}°"
# ax.text(midpoint_x, midpoint_y, midpoint_z + vector_z * 0.1, annotation_text, 
        # size=12, color='k', bbox=dict(facecolor='white', alpha=0.6))

midpoint = (earl + earr) / 2
draw_arc(ax, midpoint, radius=0.5, start_angle=0, end_angle=yaw, axis='xy', color='b', label='Yaw (θ)')
draw_arc(ax, snout, radius=0.5, start_angle=0, end_angle=pitch, axis='xz', color='r', label='Pitch (φ)')
draw_arc(ax, earl, radius=0.5, start_angle=0, end_angle=roll, axis='yz', color='g', label='Roll (ψ)')

# Annotate angles
ax.text(midpoint[0] + 0.5, midpoint[1], midpoint[2], f"Yaw: {yaw:.2f}°", color='blue', fontsize=12)
ax.text(snout[0] + 0.5, snout[1], snout[2], f"Pitch: {pitch:.2f}°", color='red', fontsize=12)
ax.text(earl[0], earl[1] + 0.5, earl[2], f"Roll: {roll:.2f}°", color='green', fontsize=12)


# Set labels for axes
ax.set_xlabel('X Coordinate', fontsize=12)
ax.set_ylabel('Y Coordinate', fontsize=12)
ax.set_zlabel('Z Coordinate', fontsize=12)

# Set title
ax.set_title(f'3D Plot of Head Keypoints at Timestamp {first_timestamp}', fontsize=14)

# Add legend
ax.legend()

# Show the plot
plt.show()
