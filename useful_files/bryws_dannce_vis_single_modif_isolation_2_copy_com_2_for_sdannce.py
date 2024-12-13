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
base_path =  '/hpc/group/tdunn/Bryan_Rigs/BigOpenField/Oct3V1/2024_11_13/20240910v1r_cricket_cyliner_test_16_17' #'/hpc/group/tdunn/Bryan_Rigs/BigOpenField/Oct3V1/2024_11_01/20240910V1r_BO_11_35' #'/hpc/group/tdunn/Bryan_Rigs/BigOpenField/Oct3V1/2024_11_13/20240910v1r_cricket_cyliner_test_16_17'
cammm = 2
video_path = os.path.join(base_path, f'videos/Camera{cammm}/0.mp4')
label3d_path = find_calib_file(base_path)
pred_folder = 'DANNCE/predict00'
# label3d_path = '/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_06_28/1686941_left_right_2/pos_synced_1686941_left_right_2_2024_06_28_1686941_left_label3d_dannce.mat' #calib
pred_path = os.path.join(base_path, pred_folder, 'save_data_AVG.mat') #  smoothed_prediction_AVG0.mat
N_FRAMES = 100
START_FRAME = 22050
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

# get dannce predictions
pred_3d = sio.loadmat(pred_path)['pred'][START_FRAME: START_FRAME+N_FRAMES]
pred_3d = np.squeeze(pred_3d, axis=1) # added for sdannce...
# print("Shape of pred_3d:", pred_3d.shape)

# compute projections
pred_2d = {}
pose_3d = np.transpose(pred_3d, (0, 2, 1))
pts = np.reshape(pose_3d, (-1, 3))


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


with writer.saving(fig, os.path.join(save_path, "vis_"+VID_NAME), dpi=300):
    for curr_frame in tqdm.tqdm(range(N_FRAMES)):
        plt.clf()
        # grab images
        imgs = [vids.get_data(curr_frame+START_FRAME)][0]
        kpts_2d = pred_2d[cam][curr_frame]
        
        temp_kpts_2d = np.r_[kpts_2d[0:6,:],kpts_2d[8:,:]]

        # Plot com keypoints
        kpts_2d_com = pred_2d_com[cam][curr_frame]
        temp_kpts_2d_com = np.r_[kpts_2d_com[0:6,:],kpts_2d_com[8:,:]]
        
        # Zoom in based on keypoints
        adjust_viewport(temp_kpts_2d, margin=450)  # Adjust margin as needed for best fit 150 is good.


        plt.imshow(imgs)
        
        # Plot com points
        plt.scatter(kpts_2d_com[:, 0], kpts_2d_com[:, 1], marker='.', color='red', linewidths=2, alpha=0.5)

        plt.scatter(temp_kpts_2d[:, 0], temp_kpts_2d[:, 1], marker='.', color='white', linewidths=2, alpha=0.5) #point size

        for color, (index_from, index_to) in zip(COLOR, CONNECTIVITY):
            xs, ys = [np.array([kpts_2d[index_from, j], kpts_2d[index_to, j]]) for j in range(2)]
            plt.plot(xs, ys, c=color, lw=2) #line error
            del xs, ys

        plt.title(vid_title)
        plt.axis("off")
        
        writer.grab_frame()
