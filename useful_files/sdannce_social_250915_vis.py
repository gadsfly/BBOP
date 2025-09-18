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

def find_calib_file(base_folder):
    for file_name in os.listdir(base_folder):
        if file_name.endswith('label3d_dannce.mat'):
            return os.path.join(base_folder, file_name)
    return None
# /hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_06_26/1686940_left/DANNCE/predict_results/six_points_multi_test_0calib
# /hpc/group/tdunn/lq53/dannce_chris_calib/240503rec_240229V1left/result_folder/train_newcom_70frames_100epo/DANNCE/predict_results
###############################################################################################################
base_path =  "/data/big_rim/rsync_dcc_sum/Oct3V1/2024_10_31/2social_mini_20240819V1r1_femalebleach_11_48"
#'/hpc/group/tdunn/Bryan_Rigs/BigOpenField/Oct3V1/2024_11_13/20240910v1r_cricket_cyliner_test_16_17' #'/hpc/group/tdunn/Bryan_Rigs/BigOpenField/Oct3V1/2024_11_01/20240910V1r_BO_11_35' #'/hpc/group/tdunn/Bryan_Rigs/BigOpenField/Oct3V1/2024_11_13/20240910v1r_cricket_cyliner_test_16_17'
cammm = 3
video_path = os.path.join(base_path, f'videos/Camera{cammm}/0.mp4')
label3d_path = find_calib_file(base_path)
pred_folder = 'SDANNCE/predict00'
# label3d_path = '/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_06_28/1686941_left_right_2/pos_synced_1686941_left_right_2_2024_06_28_1686941_left_label3d_dannce.mat' #calib
pred_path = os.path.join(base_path, pred_folder, 'save_data_AVG.mat') #  smoothed_prediction_AVG0.mat
N_FRAMES = 100
START_FRAME = 1000
ANIMAL= 'mouse20' #'mouse22'
cam = f'Camera{cammm}' 
vid_title = f'combined_cam{cammm}_{N_FRAMES}_after{START_FRAME}' #after500
VID_NAME = vid_title + '.mp4'
COLOR = connectivity.COLOR_DICT[ANIMAL]
CONNECTIVITY = connectivity.CONNECTIVITY_DICT[ANIMAL]
save_path = os.path.join(base_path, pred_folder, 'vis') #os.path.join(pred_path, 'vis')
if not os.path.exists(save_path):
    os.makedirs(save_path)

com_file = os.path.join(base_path,pred_folder,'com3d_used.mat')
com_data = sio.loadmat(com_file)
###############################################################################################################
# load camera parameterss
cameras = load_cameras(label3d_path)

# get dannce predictions  -> supports (F, 1, 3, 22) and (F, 2, 3, 22)
pred_raw = sio.loadmat(pred_path)['pred'][START_FRAME: START_FRAME+N_FRAMES]
pred_raw = np.squeeze(pred_raw)  # (F,3,22) or (F,2,3,22)

pred_2d = {}
camK, camr, camt = cameras[cam]["K"], cameras[cam]["r"], cameras[cam]["t"]
Rdist, Tdist = np.squeeze(cameras[cam]["RDistort"]), np.squeeze(cameras[cam]["TDistort"])

if pred_raw.ndim == 3:  # (F, 3, 22) -> single animal
    pose_3d = np.transpose(pred_raw, (0, 2, 1))          # (F, 22, 3)
    pts = pose_3d.reshape(-1, 3)                          # (F*22, 3)
    projpts = project_to_2d(pts, camK, camr, camt)[:, :2]
    projpts = distortPoints(projpts, camK, Rdist, Tdist).T
    projpts = projpts.reshape(N_FRAMES, 22, 2)            # (F, 22, 2)
    pred_2d[cam] = projpts
else:                  # (F, 2, 3, 22) -> two animals
    pose_3d = np.transpose(pred_raw, (0, 1, 3, 2))        # (F, 2, 22, 3)
    pts = pose_3d.reshape(-1, 3)                          # (F*2*22, 3)
    projpts = project_to_2d(pts, camK, camr, camt)[:, :2]
    projpts = distortPoints(projpts, camK, Rdist, Tdist).T
    projpts = projpts.reshape(N_FRAMES, 2, 22, 2)         # (F, 2, 22, 2)
    pred_2d[cam] = projpts


# # get dannce predictions
# pred_3d = sio.loadmat(pred_path)['pred'][START_FRAME: START_FRAME+N_FRAMES]
# pred_3d = np.squeeze(pred_3d, axis=1) # added for sdannce...
# # print("Shape of pred_3d:", pred_3d.shape)

# # compute projections
# pred_2d = {}
# pose_3d = np.transpose(pred_3d, (0, 2, 1))
# pts = np.reshape(pose_3d, (-1, 3))


# # get the 2d projection
# projpts = project_to_2d(pts,
#                         cameras[cam]["K"],
#                         cameras[cam]["r"],
#                         cameras[cam]["t"])[:, :2]

# projpts = distortPoints(projpts,
#                         cameras[cam]["K"],
#                         np.squeeze(cameras[cam]["RDistort"]),
#                         np.squeeze(cameras[cam]["TDistort"]))
# projpts = projpts.T
# projpts = np.reshape(projpts, (N_FRAMES, -1, 2))
# pred_2d[cam] = projpts


# del projpts, pred_3d


###############################3
# for com

# --- COM projection (robust to (F,3,2) | (F,2,3) | (F,3)) ---
pts_com_raw = np.asarray(com_data['com'][START_FRAME: START_FRAME+N_FRAMES])  # e.g. (F,3,2)
pred_2d_com = {}

camK, camr, camt = cameras[cam]["K"], cameras[cam]["r"], cameras[cam]["t"]
Rdist, Tdist = np.squeeze(cameras[cam]["RDistort"]), np.squeeze(cameras[cam]["TDistort"])

if pts_com_raw.ndim == 3:
    # (F,3,2) -> (F,2,3), or keep (F,2,3) as-is
    if pts_com_raw.shape[1] == 3 and pts_com_raw.shape[2] == 2:
        pts_F_A_C = np.transpose(pts_com_raw, (0, 2, 1))      # (F, 2, 3)
    elif pts_com_raw.shape[1] == 2 and pts_com_raw.shape[2] == 3:
        pts_F_A_C = pts_com_raw                               # (F, 2, 3)
    else:
        raise ValueError(f"Unexpected COM shape {pts_com_raw.shape}; expected (F,3,2) or (F,2,3).")
elif pts_com_raw.ndim == 2 and pts_com_raw.shape[1] == 3:
    # single animal (F,3) -> (F,1,3)
    pts_F_A_C = pts_com_raw[:, None, :]
else:
    raise ValueError(f"Unexpected COM shape {pts_com_raw.shape}.")

n_animals = pts_F_A_C.shape[1]
proj_com = project_to_2d(pts_F_A_C.reshape(-1, 3), camK, camr, camt)[:, :2]
proj_com = distortPoints(proj_com, camK, Rdist, Tdist).T
proj_com = proj_com.reshape(N_FRAMES, n_animals, 2)           # (F, A, 2)

pred_2d_com[cam] = proj_com


# pts_com = com_data['com'][START_FRAME: START_FRAME+N_FRAMES]
# pred_2d_com = {}
# # Get the 2d projection for com
# projpts_com = project_to_2d(pts_com,
#                             cameras[cam]["K"],
#                             cameras[cam]["r"],
#                             cameras[cam]["t"])[:, :2]

# projpts_com = distortPoints(projpts_com,
#                             cameras[cam]["K"],
#                             np.squeeze(cameras[cam]["RDistort"]),
#                             np.squeeze(cameras[cam]["TDistort"]))
# projpts_com = projpts_com.T
# projpts_com = np.reshape(projpts_com, (N_FRAMES, -1, 2))
# pred_2d_com[cam] = projpts_com
# del projpts_com
#####################3


# open videos
vids = imageio.get_reader(video_path)

# set up video writer
metadata = dict(title='combined_visualization', artist='Matplotlib')
writer = FFMpegWriter(fps=20, metadata=metadata) # orig fps = 30.

###############################################################################################################
fig = plt.figure()
plt.rcParams['figure.figsize'] = (6, 6)

def adjust_viewport(kpts_2d_a1, kpts_2d_a2=None, margin=70):
    """
    Center around the mean of visible points from one or both animals.
    kpts_2d_a*: (22,2) arrays (NaN allowed).
    """
    if kpts_2d_a2 is None:
        stack = kpts_2d_a1
    else:
        stack = np.vstack([kpts_2d_a1, kpts_2d_a2])

    # drop NaNs
    valid = ~np.isnan(stack).any(axis=1)
    pts = stack[valid] if valid.any() else stack

    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    plt.xlim([cx - margin, cx + margin])
    plt.ylim([cy + margin, cy - margin])


# def adjust_viewport(kpts_2d, margin=70):
#     """
#     Adjust the plot's viewport based on keypoints.
#     :param kpts_2d: Keypoints for the current frame.
#     :param margin: Extra space around the keypoints to ensure they are not on the edge.
#     """
#     # This method is way too shaky
#     # min_x, max_x = np.min(kpts_2d[:, 0]), np.max(kpts_2d[:, 0])
#     # min_y, max_y = np.min(kpts_2d[:, 1]), np.max(kpts_2d[:, 1])
#     # plt.xlim([min_x - margin, max_x + margin])
#     # plt.ylim([max_y + margin, min_y - margin])

#     center_x = np.mean(kpts_2d[:, 0])
#     center_y = np.mean(kpts_2d[:, 1])
#     plt.xlim([center_x - margin, center_x + margin])
#     plt.ylim([center_y + margin, center_y - margin])

with writer.saving(fig, os.path.join(save_path, "vis_"+VID_NAME), dpi=300):
    for curr_frame in tqdm.tqdm(range(N_FRAMES)):
        plt.clf()

        # frame image
        img = vids.get_data(curr_frame + START_FRAME)
        plt.imshow(img)

        # keypoints
        k = pred_2d[cam][curr_frame]
        # k shape: (22,2) if 1 animal; (2,22,2) if 2 animals
        if k.ndim == 2:
            k_a1, k_a2 = k, None
        else:
            k_a1, k_a2 = k[0], k[1]                        # NEW

        # for viewport: drop tail(mid/end) like your original
        def _trim_tail(xy):
            return np.r_[xy[0:6, :], xy[8:, :]]            # CHG: keep your style

        temp_a1 = _trim_tail(k_a1)
        temp_a2 = _trim_tail(k_a2) if k_a2 is not None else None
        # adjust_viewport(temp_a1, temp_a2, margin=450)      # CHG: include both

        # COM points
        k_com = pred_2d_com[cam][curr_frame]               # (1,2) or (2,2)
        plt.scatter(k_com[:, 0], k_com[:, 1], marker='.', color='red', linewidths=2, alpha=0.5)

        # scatter points
        plt.scatter(k_a1[:, 0], k_a1[:, 1], marker='.', color='white', linewidths=2, alpha=0.8)
        if k_a2 is not None:
            plt.scatter(k_a2[:, 0], k_a2[:, 1], marker='.', color='cyan', linewidths=2, alpha=0.8)  # NEW

        # skeleton lines
        for color, (index_from, index_to) in zip(COLOR, CONNECTIVITY):
            xs, ys = [np.array([k_a1[index_from, j], k_a1[index_to, j]]) for j in range(2)]
            plt.plot(xs, ys, c=color, lw=2)
            if k_a2 is not None:                                                                # NEW
                xs2, ys2 = [np.array([k_a2[index_from, j], k_a2[index_to, j]]) for j in range(2)]
                plt.plot(xs2, ys2, c=color, lw=2, alpha=0.9)                                    # NEW

        plt.title(vid_title)
        plt.axis("off")
        writer.grab_frame()


# with writer.saving(fig, os.path.join(save_path, "vis_"+VID_NAME), dpi=300):
#     for curr_frame in tqdm.tqdm(range(N_FRAMES)):
#         plt.clf()
#         # grab images
#         imgs = [vids.get_data(curr_frame+START_FRAME)][0]
#         kpts_2d = pred_2d[cam][curr_frame]
        
#         temp_kpts_2d = np.r_[kpts_2d[0:6,:],kpts_2d[8:,:]]

#         # Plot com keypoints
#         kpts_2d_com = pred_2d_com[cam][curr_frame]
#         temp_kpts_2d_com = np.r_[kpts_2d_com[0:6,:],kpts_2d_com[8:,:]]
        
#         # Zoom in based on keypoints
#         adjust_viewport(temp_kpts_2d, margin=450)  # Adjust margin as needed for best fit 150 is good.


#         plt.imshow(imgs)
        
#         # Plot com points
#         plt.scatter(kpts_2d_com[:, 0], kpts_2d_com[:, 1], marker='.', color='red', linewidths=2, alpha=0.5)

#         plt.scatter(temp_kpts_2d[:, 0], temp_kpts_2d[:, 1], marker='.', color='white', linewidths=2, alpha=0.5) #point size

#         for color, (index_from, index_to) in zip(COLOR, CONNECTIVITY):
#             xs, ys = [np.array([kpts_2d[index_from, j], kpts_2d[index_to, j]]) for j in range(2)]
#             plt.plot(xs, ys, c=color, lw=2) #line error
#             del xs, ys

#         plt.title(vid_title)
#         plt.axis("off")
        
#         writer.grab_frame()
