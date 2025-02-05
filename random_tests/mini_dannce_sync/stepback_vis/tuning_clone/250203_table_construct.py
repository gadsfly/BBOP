#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# =====================================================
# 1. Data Loading
# =====================================================
# Change the file path as needed.
hdf5_file_path = "/data/big_rim/rsync_dcc_sum/Oct3V1/2024_10_25/20241002PMCr2_17_05/MIR_Aligned/aligned_predictions_with_ca_and_dF_F.h5"
# Load the DataFrame and assign it to df_new
df_new = pd.read_hdf(hdf5_file_path, key='df')
print(f"Loaded DataFrame with {len(df_new)} frames.")

# =====================================================
# 2. XY Plane Circle Fitting (Your Original Code)
# =====================================================
# Assumes COM data are in columns 'com_x' and 'com_y'
x = np.array(df_new['com_x'])
y = np.array(df_new['com_y'])
points = np.column_stack((x, y))

# Compute the convex hull to approximate the outer boundary
hull = ConvexHull(points)
hull_points = points[hull.vertices]
hx, hy = hull_points[:, 0], hull_points[:, 1]

def residuals(params, x_data, y_data):
    xc, yc, r = params
    return np.sqrt((x_data - xc)**2 + (y_data - yc)**2) - r

# Initial guess based on the hull points
initial_guess = [
    np.mean(hx), 
    np.mean(hy), 
    np.mean(np.sqrt((hx - np.mean(hx))**2 + (hy - np.mean(hy))**2))
]

result = least_squares(residuals, initial_guess, args=(hx, hy))
xc, yc, r = result.x

# Generate points for the fitted circle in XY
theta = np.linspace(0, 2*np.pi, 100)
circle_x = xc + r * np.cos(theta)
circle_y = yc + r * np.sin(theta)

print(f"The computed radius of the fitted circle is: {r} mm")

# =====================================================
# 3. Determine the Table’s Z Axis from Paw Contact Points
# =====================================================
# We'll use four paw keypoints:
#   - ForepawL, ForepawR, HindpawL, HindpawR
#
# Their order is defined by the following keypoint labels (1-indexed):
keypoint_labels = [
    'EarL',       # 1
    'EarR',       # 2
    'Snout',      # 3
    'SpineF',     # 4
    'SpineM',     # 5
    'Tail(base)', # 6
    'Tail(mid)',  # 7
    'Tail(end)',  # 8
    'ForepawL',   # 9
    'WristL',     # 10
    'ElbowL',     # 11
    'ShoulderL',  # 12
    'ForepawR',   # 13
    'WristR',     # 14
    'ElbowR',     # 15
    'ShoulderR',  # 16
    'HindpawL',   # 17
    'AnkleL',     # 18
    'KneeL',      # 19
    'HindpawR',   # 20
    'AnkleR',     # 21
    'KneeR'       # 22
]
paw_labels = [ 'HindpawL', 'HindpawR']# 'ForepawL', 'ForepawR',
# Create a mapping from paw label to its (1-indexed) column index.
paw_indices = { label: keypoint_labels.index(label) + 1 for label in paw_labels }

# Define a threshold (in mm) above the minimum z for candidate contact points.
threshold = 5.0

candidate_points = []  # Will hold [x, y, z] for candidate contact points.
for label in paw_labels:
    idx = paw_indices[label]
    xs = np.array(df_new[f'kp{idx}_x'])
    ys = np.array(df_new[f'kp{idx}_y'])
    zs = np.array(df_new[f'kp{idx}_z'])
    
    # Find the minimum z for this paw.
    min_z = np.min(zs)
    
    # Select frames where the paw's z is within 'threshold' mm of its minimum.
    candidate_idx = np.where(zs <= min_z + threshold)[0]
    for i in candidate_idx:
        candidate_points.append([xs[i], ys[i], zs[i]])

candidate_points = np.array(candidate_points)
print("Total candidate contact points from paws:", candidate_points.shape[0])

# Fit a plane to these candidate points using singular value decomposition (SVD)
centroid = np.mean(candidate_points, axis=0)
U, S, Vt = np.linalg.svd(candidate_points - centroid)
normal = Vt[-1]
# (Optional) Ensure the normal points upward (i.e., positive z)
if normal[2] < 0:
    normal = -normal

# Compute the table's z coordinate at the XY circle center (xc, yc) from the plane equation.
# The plane equation: normal[0]*(x - centroid[0]) + normal[1]*(y - centroid[1]) + normal[2]*(z - centroid[2]) = 0
# Solve for z:
table_z = centroid[2] - (normal[0]*(xc - centroid[0]) + normal[1]*(yc - centroid[1])) / normal[2]
print("Fitted table z coordinate at circle center:", table_z)

# --- Compute the Tilt Angle Relative to the Ground ---
ground_normal = np.array([0, 0, 1])
dot_prod = np.clip(np.dot(normal, ground_normal), -1.0, 1.0)
tilt_angle_rad = np.arccos(dot_prod)
tilt_angle_deg = np.degrees(tilt_angle_rad)
print("Table tilt angle relative to the ground: {:.2f} degrees".format(tilt_angle_deg))

# =====================================================
# 4. Construct the Visual Table (Circle on the Table Plane)
# =====================================================
# Define a helper to compute two orthonormal vectors spanning the table plane.
def plane_basis(n):
    # Choose an arbitrary vector that is not colinear with n.
    if np.abs(n[0]) < 1e-6 and np.abs(n[1]) < 1e-6:
        v = np.array([1, 0, 0])
    else:
        v = np.array([0, 0, 1])
    b1 = np.cross(n, v)
    b1 = b1 / np.linalg.norm(b1)
    b2 = np.cross(n, b1)
    b2 = b2 / np.linalg.norm(b2)
    return b1, b2

# Use the fitted table plane normal to define the in-plane axes.
b1, b2 = plane_basis(normal)

# The table circle center in 3D is taken as (xc, yc, table_z)
table_center = np.array([xc, yc, table_z])
# Parameterize the circle on the table plane.
circle3d = np.array([table_center + r * (np.cos(t)*b1 + np.sin(t)*b2) for t in theta])

# =====================================================
# 5. Visualization: Combined 2D and 3D
# =====================================================
fig = plt.figure(figsize=(16, 7))

# ----- Left Panel: 2D XY View (Same as Original) -----
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(x, y, 'o', ms=3, alpha=0.3, label='Original COM Data')
ax1.plot(hx, hy, 'go', ms=5, label='Convex Hull Points')
ax1.plot(circle_x, circle_y, 'r-', lw=2, label='Fitted Circle')
ax1.plot(xc, yc, 'bo', label='Computed Center')
ax1.set_aspect('equal', 'box')
ax1.set_xlabel('X Position (mm)')
ax1.set_ylabel('Y Position (mm)')
ax1.set_title('2D Circle Fit Using Convex Hull Points')
ax1.legend()

# ----- Right Panel: 3D View -----
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
# Plot candidate contact points
ax2.scatter(candidate_points[:, 0], candidate_points[:, 1], candidate_points[:, 2],
            color='r', s=20, label='Paw Candidate Points')
# Plot the fitted plane as a surface
# Create a grid over the candidate points' x and y range.
grid_x = np.linspace(np.min(candidate_points[:, 0]), np.max(candidate_points[:, 0]), 20)
grid_y = np.linspace(np.min(candidate_points[:, 1]), np.max(candidate_points[:, 1]), 20)
grid_x, grid_y = np.meshgrid(grid_x, grid_y)
grid_z = centroid[2] - (normal[0]*(grid_x - centroid[0]) + normal[1]*(grid_y - centroid[1])) / normal[2]
ax2.plot_surface(grid_x, grid_y, grid_z, alpha=0.5, color='cyan')
# Mark the plane's centroid
ax2.scatter(centroid[0], centroid[1], centroid[2],
            color='b', s=100, label='Candidate Points Centroid')
# Plot the computed table circle (projected onto the table plane)
ax2.plot(circle3d[:, 0], circle3d[:, 1], circle3d[:, 2],
         'm-', lw=3, label='Table Circle on Fitted Plane')
# Mark the table center
ax2.scatter(table_center[0], table_center[1], table_center[2],
            color='k', s=100, marker='x', label='Table Center (COM)')

ax2.set_xlabel("X (mm)")
ax2.set_ylabel("Y (mm)")
ax2.set_zlabel("Z (mm)")
ax2.set_title("3D View: Paw Contact Points, Fitted Plane & Table Circle")
ax2.legend()

plt.tight_layout()
plt.show()

# =====================================================
# 6. Save the Table Parameters for Future Reconstruction
# =====================================================
table_params = {
    'table_center': table_center,  # [xc, yc, table_z]
    'radius': r,
    'plane_normal': normal,
    'b1': b1,
    'b2': b2
}
# Save to an NPZ file
np.savez('table_params.npz', **table_params)
print("Saved table parameters to 'table_params.npz'.")
