# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:01:45 2022

@author: 13190044511
"""
import os
import numpy as np
import scipy.io as sio
import imageio
import tqdm
from projection import *
import connectivity

import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle

def find_calib_file(base_folder):
    for file_name in os.listdir(base_folder):
        if file_name.endswith('label3d_dannce.mat'):
            return os.path.join(base_folder, file_name)
    return None
# /home/lq53/mir_data/24summ/2024_06_26/1686940_left/DANNCE/predict_results/240726_mir_label_from_demo

# /hpc/group/tdunn/lq53/dannce_chris_calib/240503rec_240229V1left/result_folder/train_newcom_70frames_100epo/DANNCE/predict_results
###############################################################################################################
base_path =  '/home/lq53/mir_data/24summ/2024_06_26/1686940_left'
video_path = os.path.join(base_path, 'videos/Camera2/0.mp4')
label3d_path = find_calib_file(base_path)
pred_folder = 'DANNCE/predict_results/240729_mir_label6_from_arnav'
# label3d_path = '/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_06_28/1686941_left_right_2/pos_synced_1686941_left_right_2_2024_06_28_1686941_left_label3d_dannce.mat' #calib
pred_path = os.path.join(base_path, pred_folder, 'save_data_AVG0.mat') #  smoothed_prediction_AVG0.mat save_data_AVG0.mat
N_FRAMES = 500
START_FRAME = 0
ANIMAL= 'mouse20' #'mouse22'
cam = 'Camera2' 
vid_title = 'combined_cam2_500'
VID_NAME = vid_title + '.mp4'
COLOR = connectivity.COLOR_DICT[ANIMAL]
CONNECTIVITY = connectivity.CONNECTIVITY_DICT[ANIMAL]
save_path = os.path.join(base_path, pred_folder, 'vis') #os.path.join(pred_path, 'vis')
if not os.path.exists(save_path):
    os.makedirs(save_path)

com_file = os.path.join(base_path,pred_folder,'com3d_used.mat')
com_data = sio.loadmat(com_file)
###############################################################################################################

# def project_to_2d(pts, K, r, t):
#     """
#     Project 3D points to 2D using camera parameters.
#     :param pts: 3D points (N x 3)
#     :param K: Camera intrinsic matrix
#     :param r: Rotation matrix
#     :param t: Translation vector
#     :return: 2D projected points (N x 2)
#     """
#     pts_homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))])  # Convert to homogeneous coordinates
#     camera_matrix = np.dot(K, np.hstack([r, t.reshape(-1, 1)]))  # Camera projection matrix
#     pts_2d_homogeneous = np.dot(camera_matrix, pts_homogeneous.T).T  # Apply projection
#     pts_2d = pts_2d_homogeneous[:, :2] / pts_2d_homogeneous[:, 2][:, np.newaxis]  # Convert back to 2D
#     # print(pts_2d.shape)
#     return pts_2d

############################################3
# load camera parameterss
cameras = load_cameras(label3d_path)

# get dannce predictions
pred_3d = sio.loadmat(pred_path)['pred'][START_FRAME: START_FRAME+N_FRAMES]
# print(pred_3d.shape)

# compute projections
pred_2d = {}
pose_3d = np.transpose(pred_3d, (0, 2, 1))
# print(pose_3d.shape)
pts = np.reshape(pose_3d, (-1, 3))
# print(pts.shape)


# get the 2d projection
projpts = project_to_2d(pts,
                        cameras[cam]["K"],
                        cameras[cam]["r"],
                        cameras[cam]["t"])[:, :2]

projpts = distortPoints(projpts,
                        cameras[cam]["K"],
                        np.squeeze(cameras[cam]["RDistort"]),
                        np.squeeze(cameras[cam]["TDistort"]))
projpts = projpts.T
projpts = np.reshape(projpts, (N_FRAMES, -1, 2))
pred_2d[cam] = projpts


del projpts, pred_3d

###############################3
# for com
pts_com = com_data['com'][START_FRAME: START_FRAME+N_FRAMES]
pred_2d_com = {}
pred_2d_com_box = {}

# # Add the box points around com
# box_offsets = np.array([[-120, -120, -120], [120, -120, -120], [-120, 120, -120], [120, 120, -120],
#                         [-120, -120, 120], [120, -120, 120], [-120, 120, 120], [120, 120, 120]])
# pts_com_with_box = np.concatenate([pts_com[:, np.newaxis, :] + offset for offset in box_offsets], axis=1)
# # print("com", pts_com)
# # print("box", pts_com_with_box)
# pts_com_flat = np.reshape(pts_com_with_box, (-1, 3))
# # print("pts_com_flat", pts_com_flat.shape)

# # Project the 3D box points to 2D
# projpts_com = project_to_2d(pts_com_flat,
#                             cameras[cam]["K"],
#                             cameras[cam]["r"],
#                             cameras[cam]["t"])[:, :2]

# projpts_com = distortPoints(projpts_com,
#                             cameras[cam]["K"],
#                             np.squeeze(cameras[cam]["RDistort"]),
#                             np.squeeze(cameras[cam]["TDistort"]))
# projpts_com = projpts_com.T
# projpts_com = np.reshape(projpts_com, (N_FRAMES, -1, 2))
# # projpts_com = projpts_com.reshape(N_FRAMES, -1, 2)
# pred_2d_com_box[cam] = projpts_com
# print("box", projpts_com)
# del projpts_com

# Get the 2d projection for com
projpts_com = project_to_2d(pts_com,
                            cameras[cam]["K"],
                            cameras[cam]["r"],
                            cameras[cam]["t"])[:, :2]

projpts_com = distortPoints(projpts_com,
                            cameras[cam]["K"],
                            np.squeeze(cameras[cam]["RDistort"]),
                            np.squeeze(cameras[cam]["TDistort"]))
projpts_com = projpts_com.T
projpts_com = np.reshape(projpts_com, (N_FRAMES, -1, 2))
pred_2d_com[cam] = projpts_com
# print("com", projpts_com)
del projpts_com
#####################3


# open videos
vids = imageio.get_reader(video_path)

# set up video writer
metadata = dict(title='combined_visualization', artist='Matplotlib')
writer = FFMpegWriter(fps=20, metadata=metadata) # orig fps = 30.

###############################################################################################################
fig = plt.figure()
plt.rcParams['figure.figsize'] = (6, 6)


def adjust_viewport(kpts_2d, margin=70):
    """
    Adjust the plot's viewport based on keypoints.
    :param kpts_2d: Keypoints for the current frame.
    :param margin: Extra space around the keypoints to ensure they are not on the edge.
    """
    # This method is way too shaky
    # min_x, max_x = np.min(kpts_2d[:, 0]), np.max(kpts_2d[:, 0])
    # min_y, max_y = np.min(kpts_2d[:, 1]), np.max(kpts_2d[:, 1])
    # plt.xlim([min_x - margin, max_x + margin])
    # plt.ylim([max_y + margin, min_y - margin])

    center_x = np.mean(kpts_2d[:, 0])
    center_y = np.mean(kpts_2d[:, 1])
    plt.xlim([center_x - margin, center_x + margin])
    plt.ylim([center_y + margin, center_y - margin])


# # Define the edges of the box (connectivity)
# box_edges = [
#     (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom square
#     (4, 5), (5, 7), (7, 6), (6, 4),  # Top square
#     (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical connections
# ]

# # Define box edge colors (all blue in this case)
# box_edge_colors = ['blue'] * len(box_edges)



with writer.saving(fig, os.path.join(save_path, "vis_"+VID_NAME), dpi=300):
    for curr_frame in tqdm.tqdm(range(N_FRAMES)):
        plt.clf()
        # grab images
        imgs = [vids.get_data(curr_frame+START_FRAME)][0]
        kpts_2d = pred_2d[cam][curr_frame]
        
        temp_kpts_2d = np.r_[kpts_2d[0:6,:],kpts_2d[8:,:]]

        # Plot com keypoints
        kpts_2d_com = pred_2d_com[cam][curr_frame]
        # kpts_2d_comb = pred_2d_com_box[cam][curr_frame]
        
        # Zoom in based on keypoints
        adjust_viewport(temp_kpts_2d, margin=450)  # Adjust margin as needed for best fit 150 is good.

        plt.imshow(imgs)
        
        # Plot com points
        plt.scatter(kpts_2d_com[:, 0], kpts_2d_com[:, 1], marker='.', color='red', linewidths=2, alpha=0.5)
        # # Plot the box points
        # plt.scatter(kpts_2d_comb[:, 0], kpts_2d_comb[:, 1], marker='.', color='blue', linewidths=2, alpha=0.5)
        # # Draw the edges of the box
        # for color, (index_from, index_to) in zip(box_edge_colors, box_edges):
        #     xs, ys = [np.array([kpts_2d_comb[index_from, j], kpts_2d_comb[index_to, j]]) for j in range(2)]
        #     plt.plot(xs, ys, c=color, lw=1)
        #     del xs, ys


        plt.scatter(kpts_2d[:, 0], kpts_2d[:, 1], marker='.', color='white', linewidths=2, alpha=0.5) #point size

        for color, (index_from, index_to) in zip(COLOR, CONNECTIVITY):
            xs, ys = [np.array([kpts_2d[index_from, j], kpts_2d[index_to, j]]) for j in range(2)]
            plt.plot(xs, ys, c=color, lw=2)  # line error
            del xs, ys

        plt.title(vid_title)
        plt.axis("off")
        
        writer.grab_frame()
