import os
import re
import numpy as np
import scipy.io as sio
import imageio
import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
# from utlis.projection import *                                 
from utlis.com_traga_utlis import load_data, plot_3d_trajectory, detect_jumps, generate_jump_video,generate_com_video #generate_com_video_choice#*
# from utlis.com_traga_utlis import plot_com_all, process_folders, temp_change_calib_pos_to_0, generate_dannce_vid_seq# *
import 

weired_folders = [
#     # "/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_07_03/1691486_left_right_caffeine_1448",
#     # "/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_07_03/1691486_left_caffeine_1051",
#     # "/home/lq53/mir_data/24summ/2024_07_10/1691485_no_hole_habituation_13_59"
#     # "/home/lq53/mir_data/24summ/2024_06_26/1686940_left"
#     # base_paths = [ 
#     # below 3 need to process from strach.
#     # '/home/lq53/mir_data/24summ/2024_06_26/1686940_left', #validation set
#     # '/home/lq53/mir_data/24summ/2024_06_28/1686941_left_right_2', 
#     # '/home/lq53/mir_data/24summ/2024_07_12/240605PMC1_right_hole_11_27',
#     # # labels
#     "/home/lq53/mir_data/24summ/2024_07_10/1691485_no_hole_habituation_13_59",
#     "/home/lq53/mir_data/24summ/2024_07_11/1691485BMCFF1505",
#     "/home/lq53/mir_data/24summ/2024_07_10/1691485_left_hole_saline_10_35",
#     "/home/lq53/mir_data/24summ/2024_07_03/1691486_left_right_habituation",
#     "/home/lq53/mir_data/24summ/2024_07_19/240605PMC_window2_right2holes_12_14",
#     #below are my labels.
#     # "/home/lq53/mir_data/24summ/2024_07_03/1691486_left_right_habituation",
#     # "/home/lq53/mir_data/24summ/2024_07_12/240605PMC1_right_hole_11_27" #this one did not work
#     "/home/lq53/mir_data/24summ/2024_07_16/1691485RMHBN1405",
#     "/home/lq53/mir_data/24summ/2024_07_15/1691485RMPBS1659",
#     "/home/lq53/mir_data/24summ/2024_07_19/240605PMC_window2_right2holes_11_30",
#     "/home/lq53/mir_data/24summ/2024_07_15/1691485RMHBN1425",
#     "/home/lq53/mir_data/24summ/2024_07_15/1691485RMPBF1531"
#     # 'G:\\Videos\\6cam\\lq53\\Mir_intrinsics\\pyxy3d_noT'
#     # '/n/olveczky_lab_tier1/Lab/dannce_rig2/data/M1-M7_photometry/Alone', SYNTAX_ERROR_UPDATE_THIS_WITH_YOUR_OWN_FILES
#     # '/n/holylabs/LABS/olveczky_lab/Lab/dannce-dev/hannah-data/COM_DANNCE_TRAINING'
# # ]
# ]
    "/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_07_10/1691485_no_hole_habituation_13_59",
    "/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_07_11/1691485BMCFF1505",
    "/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_07_10/1691485_left_hole_saline_10_35",
    "/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_07_03/1691486_left_right_habituation",
    "/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_07_19/240605PMC_window2_right2holes_12_14",
    #below are my labels.
    # "/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_07_03/1691486_left_right_habituation"
    # "/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_07_12/240605PMC1_right_hole_11_27" #this one did not work
    "/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_07_16/1691485RMHBN1405",
    "/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_07_15/1691485RMPBS1659",
    "/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_07_19/240605PMC_window2_right2holes_11_30",
    "/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_07_15/1691485RMHBN1425",
    "/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_07_15/1691485RMPBF1531",
]

for wie in weired_folders:
    # note that there exist a function plot_com_all which i wrote before that can easily do below, 
    # but for somereason i used this just that we can be a bit more flexible in terms of the name of the folders and stuff...

    # /home/lq53/mir_data/24summ/2024_06_26/1686940_left/COM_df/predict_results//vis/vis_combined_Camera2_1000_from_0.mp4
    com_foler = os.path.join(wie, com_fn, 'predict_results')
    com_path = os.path.join(com_foler, 'com3d.mat')   
    com_folder_save = os.path.join(com_foler, 'vis')
    if not os.path.exists(com_folder_save):
        os.makedirs(com_folder_save)
    graph_title = "z_com3d_vis_"
    
    com_data = load_data(com_path)

    plot_3d_trajectory(com_data, graph_title, com_folder_save)
    jump_indices = detect_jumps(com_data, com_folder_save)

    # # # produce video, which is not necessary if not labeling more com to detect what's wrong


    save_path = com_folder_save # os.path.join(com_foler, 'vis') #os.path.join(pred_path, 'vis')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    base_folder = wie
    base_base_folder = os.path.dirname(base_folder)
    generate_jump_video(com_data, base_folder, jump_indices, graph_title, save_path, cam='Camera1')
    # generate_com_video(com_data, base_folder, graph_title, save_path, cam='Camera1')
# generate_com_video_choice # without that much things, just generate. quick iteration made for one click pipeline...

    