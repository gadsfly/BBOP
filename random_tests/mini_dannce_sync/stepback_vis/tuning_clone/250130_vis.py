import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os
import sys
sys.path.append(os.path.abspath('../../../..'))
import numpy as np
import pandas as pd
import optitracking_lib as orig
import adaped_try_df as dfrev

# from utlis.corr_utlis.processed_syned_load import load_filtered_data_from_h5

# rec_path = '/data/big_rim/rsync_dcc_sum/Oct3V1/2024_10_25/20241002PMCr2_17_05'

# hdf5_file_path = os.path.join(rec_path, 'MIR_Aligned/aligned_predictions_with_ca_and_dF_F.h5')

# # Load the DataFrame from the HDF5 file
# df_merged_with_dF_F = pd.read_hdf(hdf5_file_path, key='df')

# df_test = df_merged_with_dF_F.copy()
# df_test = df_test.reset_index()

# df_results = dfrev.process_tracking_data_df(
#     df_test,
#     spineF_kp=4,   # example keypoint indices
#     tailB_kp=6,
#     spineM_kp=5,
#     earL_kp=1,  # Example: keypoint 1 is 'EarL'
#     earR_kp=2,  # Example: keypoint 2 is 'EarR'
#     snout_kp=3,
#     frame_rate=30.0
# )


# # Suppose we have nf frames
# nf = 200  # example number of frames

# # 1) Example data (replace with your own!)
# #    Let's say each center is just random for demonstration:
# head_center = np.random.rand(nf, 3)
# body_center = np.random.rand(nf, 3)

# # For the rotation matrices, let's just make identity for demonstration,
# # or small random rotations. In real usage, you'd have your actual 3x3s.
# global_head_rm = np.tile(np.eye(3), (nf, 1, 1))  # shape: (nf, 3, 3)
# global_body_rm = np.tile(np.eye(3), (nf, 1, 1))

# 2) Create a figure and 3D axes
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(projection='3d')

# We'll plot up to some bounding region:
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Head & Body Movement (3D)")

# 3) Initialize lines/quivers
# We can't directly store quiver objects in a way that updates easily.
# We'll store them as separate 3D line segments, or we can do a small workaround.

# Let's store the line objects for each axis: 
# Each axis has 3 segments: one for HEAD x, HEAD y, HEAD z
# then one for BODY x, BODY y, BODY z
head_x_line, = ax.plot([], [], [], color='r', lw=2, label='Head X-axis')
head_y_line, = ax.plot([], [], [], color='g', lw=2, label='Head Y-axis')
head_z_line, = ax.plot([], [], [], color='b', lw=2, label='Head Z-axis')

body_x_line, = ax.plot([], [], [], color='r', lw=2, ls='--', label='Body X-axis')
body_y_line, = ax.plot([], [], [], color='g', lw=2, ls='--', label='Body Y-axis')
body_z_line, = ax.plot([], [], [], color='b', lw=2, ls='--', label='Body Z-axis')

# We'll put them in a list for convenience
head_lines = [head_x_line, head_y_line, head_z_line]
body_lines = [body_x_line, body_y_line, body_z_line]

# This helps us avoid overlapping labels in the legend
ax.legend(loc='upper right')

# 4) Define the update function
def update(frame):
    # HEAD
    cx, cy, cz = head_center[frame]
    # Extract rows from the rotation matrix
    # R = [hx; hy; hz], each row is a direction
    hx = global_head_rm[frame, 0, :]  # (3,)
    hy = global_head_rm[frame, 1, :]
    hz = global_head_rm[frame, 2, :]

    # We'll scale the length so the axes are visible. 
    # For example, length = 0.05
    length = 0.05
    hx_ends = [cx + length*hx[0], cy + length*hx[1], cz + length*hx[2]]
    hy_ends = [cx + length*hy[0], cy + length*hy[1], cz + length*hy[2]]
    hz_ends = [cx + length*hz[0], cy + length*hz[1], cz + length*hz[2]]

    # Update the data for lines
    # X-axis for HEAD
    head_lines[0].set_data_3d([cx, hx_ends[0]], 
                              [cy, hx_ends[1]], 
                              [cz, hx_ends[2]])
    # Y-axis for HEAD
    head_lines[1].set_data_3d([cx, hy_ends[0]],
                              [cy, hy_ends[1]],
                              [cz, hy_ends[2]])
    # Z-axis for HEAD
    head_lines[2].set_data_3d([cx, hz_ends[0]],
                              [cy, hz_ends[1]],
                              [cz, hz_ends[2]])

    # BODY
    bx, by, bz = body_center[frame]
    bx_vec = global_body_rm[frame, 0, :]
    by_vec = global_body_rm[frame, 1, :]
    bz_vec = global_body_rm[frame, 2, :]

    bx_ends = [bx + length*bx_vec[0], by + length*bx_vec[1], bz + length*bx_vec[2]]
    by_ends = [bx + length*by_vec[0], by + length*by_vec[1], bz + length*by_vec[2]]
    bz_ends = [bx + length*bz_vec[0], by + length*bz_vec[1], bz + length*bz_vec[2]]

    body_lines[0].set_data_3d([bx, bx_ends[0]], 
                              [by, bx_ends[1]],
                              [bz, bx_ends[2]])
    body_lines[1].set_data_3d([bx, by_ends[0]], 
                              [by, by_ends[1]],
                              [bz, by_ends[2]])
    body_lines[2].set_data_3d([bx, bz_ends[0]],
                              [by, bz_ends[1]],
                              [bz, bz_ends[2]])

    return head_lines + body_lines

# 5) Create the animation
anim = FuncAnimation(fig, update, frames=nf, interval=50, blit=False)

# Show animation in a Jupyter notebook (if using %matplotlib inline)
plt.show()
