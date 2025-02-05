# import os
# import sys
# sys.path.append(os.path.abspath('../../../..'))
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from mpl_toolkits.mplot3d import Axes3D
# # from utlis.corr_utlis.processed_syned_load import load_filtered_data_from_h5
# import pandas as pd
# import adaped_try_df as dfrev

# rec_path = '/data/big_rim/rsync_dcc_sum/Oct3V1/2024_10_25/20241002PMCr2_17_05'

# hdf5_file_path = os.path.join(rec_path, 'MIR_Aligned/aligned_predictions_with_ca_and_dF_F.h5')

# # Load the DataFrame from the HDF5 file
# df_merged_with_dF_F = pd.read_hdf(hdf5_file_path, key='df')

# df_test = df_merged_with_dF_F.copy()
# df_test = df_test.reset_index()

# results = dfrev.process_tracking_data_df(
#     df_test,
#     spineF_kp=4,   # example keypoint indices
#     tailB_kp=6,
#     spineM_kp=5,
#     earL_kp=1,  # Example: keypoint 1 is 'EarL'
#     earR_kp=2,  # Example: keypoint 2 is 'EarR'
#     snout_kp=3,
#     frame_rate=30.0
# )

# # Extract data from results
# global_head_rm = results['global_head_rm']
# r_roots = results['r_roots']
# head_center = np.column_stack((results['dx_cum'], results['dy_cum'], np.zeros(len(results['dx_cum']))))
# body_center = head_center + np.array([0, 0, 0.1])  # Offset body center for visibility

# # Set up the figure and axes
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(projection='3d')

# # Set plot limits and labels
# ax.set_xlim([-1, 1])
# ax.set_ylim([-1, 1])
# ax.set_zlim([-1, 1])
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title("Head and Body Movement (3D)")

# # Initialize plot elements (lines for axes)
# head_lines = [ax.plot([], [], [], color=c, lw=2)[0] for c in ['r', 'g', 'b']]
# body_lines = [ax.plot([], [], [], color=c, lw=2, ls='--')[0] for c in ['r', 'g', 'b']]

# # Update function for animation
# def update(frame):
#     cx, cy, cz = head_center[frame]
#     bx, by, bz = body_center[frame]

#     # Head rotation vectors
#     hx = global_head_rm[frame, 0, :]
#     hy = global_head_rm[frame, 1, :]
#     hz = global_head_rm[frame, 2, :]

#     # Body rotation vectors
#     bx_vec = r_roots[frame, 0, :]
#     by_vec = r_roots[frame, 1, :]
#     bz_vec = r_roots[frame, 2, :]

#     # Scale for visibility
#     length = 0.05

#     # Update head lines
#     head_lines[0].set_data_3d([cx, cx + length * hx[0]], [cy, cy + length * hx[1]], [cz, cz + length * hx[2]])
#     head_lines[1].set_data_3d([cx, cx + length * hy[0]], [cy, cy + length * hy[1]], [cz, cz + length * hy[2]])
#     head_lines[2].set_data_3d([cx, cx + length * hz[0]], [cy, cy + length * hz[1]], [cz, cz + length * hz[2]])

#     # Update body lines
#     body_lines[0].set_data_3d([bx, bx + length * bx_vec[0]], [by, by + length * bx_vec[1]], [bz, bz + length * bx_vec[2]])
#     body_lines[1].set_data_3d([bx, bx + length * by_vec[0]], [by, by + length * by_vec[1]], [bz, bz + length * by_vec[2]])
#     body_lines[2].set_data_3d([bx, bx + length * bz_vec[0]], [by, by + length * bz_vec[1]], [bz, bz + length * bz_vec[2]])

#     return head_lines + body_lines

# # Create the animation
# anim = FuncAnimation(fig, update, frames=len(head_center), interval=50, blit=False)

# # Show the animation
# plt.show()

# # Save animation as an MP4 if needed
# # anim.save('head_body_movement.mp4', writer='ffmpeg')


