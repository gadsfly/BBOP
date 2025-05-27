import os
import re
import numpy as np
import scipy.io as sio
import imageio
import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import sys
sys.path.append(os.path.abspath('../..'))
import utlis.connectivity as connectivity
from utlis.projection import *
from utlis.sync_utlis.sync_df_utlis import find_calib_file
import shutil
from scipy.signal import medfilt

##### functions to use: generate_dannce_vid_seq, plot_com_all

def load_com(file_path):
    data = sio.loadmat(file_path)
    return data['com']

def plot_3d_trajectory_com(com_data, graph_title, com_folder_save, zmin, zmax):
    # Extract positions
    x_positions = com_data[:, 0]
    y_positions = com_data[:, 1]
    z_positions = com_data[:, 2]
    # Use the index as a proxy for time to generate a color gradient
    time_steps = np.arange(len(x_positions))

    # Plotting the 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create a color map
    cmap = plt.get_cmap('viridis')
    # Normalize the time_steps to use with the color map
    norm = plt.Normalize(time_steps.min(), time_steps.max())

    scatter = ax.scatter(x_positions, y_positions, z_positions, c=time_steps, cmap=cmap, marker='o', norm=norm)
    ax.set_title(graph_title)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_zlim(zmin, zmax)
    fig.colorbar(scatter, ax=ax, label='Time Step')
    plt.savefig(os.path.join(com_folder_save,'com_3d_trajectory_plot.jpg'), format='jpg')
    plt.show()
    plt.close()

def detect_jumps(com_data, com_folder_save):
    x_positions = com_data[:, 0]
    y_positions = com_data[:, 1]
    z_positions = com_data[:, 2]

    # Calculate differences between consecutive positions
    dx = np.diff(x_positions)
    dy = np.diff(y_positions)
    dz = np.diff(z_positions)

    # Compute the magnitude of the differences
    d_magnitude = np.sqrt(dx**2 + dy**2 + dz**2)

    # Define a threshold for significant jumps (this can be adjusted based on your data)
    threshold = np.mean(d_magnitude) + 6 * np.std(d_magnitude) # used to be 2*, but can change. changed to 10 when started fintuning

    # Identify frames with significant jumps
    significant_jumps = d_magnitude > threshold
    jump_indices = np.where(significant_jumps)[0]

    # Plotting the magnitude of differences
    plt.figure(figsize=(12, 6))
    plt.plot(d_magnitude, label='Magnitude of Differences')
    plt.scatter(jump_indices, d_magnitude[jump_indices], color='red', label='Significant Jumps')
    plt.axhline(threshold, color='orange', linestyle='--', label='Threshold')
    plt.xlabel('Frame')
    plt.ylabel('Magnitude of Position Difference')
    plt.title('Detection of Jumps in COM Trajectory')
    plt.legend()
    plt.savefig(os.path.join(com_folder_save, 'com_trajectory_jumps_plot.jpg'), format='jpg')
    plt.show()
    plt.close()

    np.save(os.path.join(com_folder_save, 'com_jump_indices.npy'), jump_indices)

    # Output frames with significant jumps
    print("saved into npy. Frames with significant jumps:", jump_indices)
    return jump_indices


def analyze_com_trajectory(com_data, save_folder):
    """
    Analyze the COM trajectory data by smoothing, plotting the COM coordinates over time,
    calculating speed, creating a speed histogram, and saving the plots.

    Parameters:
    - com_data: numpy array of shape (n_frames, 3), the COM data with columns x, y, z.
    - save_folder: string, path to the folder where the plots will be saved.

    Returns:
    - None
    """

    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Smooth the COM data using a median filter
    # Adjust the kernel size as needed (must be odd)
    kernel_size = 31  # This can be adjusted based on your data
    com_data_smoothed = medfilt(com_data, kernel_size=(kernel_size, 1))

    # Plot the smoothed COM coordinates over time
    plt.figure(figsize=(12, 6))
    plt.plot(com_data_smoothed)
    plt.title('Smoothed COM Trajectory')
    plt.xlabel('Frame')
    plt.ylabel('Position (mm)')
    plt.legend(['X', 'Y', 'Z'])
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'com_trajectory_plot.jpg'), format='jpg')
    plt.show()
    plt.close()

    # Calculate differences between consecutive frames
    diff_x = np.diff(com_data_smoothed[:, 0])
    diff_y = np.diff(com_data_smoothed[:, 1])
    diff_z = np.diff(com_data_smoothed[:, 2])

    # Calculate speed (displacement magnitude)
    speed = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)

    # Create speed histogram
    max_speed = 20  # Maximum speed to display in the histogram (mm/frame)
    bin_size = 0.1  # Bin size (mm/frame)
    bins = np.arange(0, max_speed + bin_size, bin_size)

    # Histogram of speed
    speed_counts, bin_edges = np.histogram(speed, bins=bins)

    # Plot the speed histogram
    plt.figure(figsize=(12, 6))
    plt.plot(bin_edges[:-1], speed_counts)
    plt.title('Speed Distribution')
    plt.xlabel('Speed (mm/frame)')
    plt.ylabel('Number of Frames')
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'speed_histogram.jpg'), format='jpg')
    plt.show()
    plt.close()

    # Optionally, save the speed data for further analysis
    np.save(os.path.join(save_folder, 'speed_data.npy'), speed)

    # Output completion message
    print("Analysis complete. Plots saved to:", save_folder)


def generate_jump_video(com_data, base_folder, jump_indices, graph_title, save_path, cam='Camera1'):
    # if jump_indices == []:c#somehow caused error....
    if len(jump_indices) == 0:
        return
    
    label3d_path = find_calib_file(base_folder)
    video_path = os.path.join(base_folder, 'videos/Camera1/0.mp4')
    vid_title = graph_title
    VID_NAME = vid_title + '.mp4'

    ###############################################################################################################
    # load camera parameters
    cameras = load_cameras(label3d_path)

    # get dannce predictions
    # pred_3d = sio.loadmat(pred_path)['pred'][START_FRAME: START_FRAME+N_FRAMES]
    pts = com_data[jump_indices]
    print(len(pts))
    N_FRAMES = len(jump_indices) #jump_indices
    print('N_FRAMES', N_FRAMES)
    if N_FRAMES ==0:
        return
    # compute projections
    pred_2d = {}
    # pose_3d = np.transpose(pred_3d, (0, 2, 1))
    # pts = np.reshape(pose_3d, (-1, 3))

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
    print('pred_2d',len(projpts))


    del projpts#, pred_3d

    # open videos
    vids = imageio.get_reader(video_path)

    # set up video writer
    metadata = dict(title='dannce_visualization', artist='Matplotlib')
    writer = FFMpegWriter(fps=4, metadata=metadata) # orig fps = 30., sihan uses 20, use 0.5 for com debug

    ###############################################################################################################
    fig = plt.figure()
    plt.rcParams['figure.figsize'] = (6, 6)

    with writer.saving(fig, os.path.join(save_path, "vis_"+VID_NAME), dpi=300):
        for curr_frame, i in enumerate(tqdm.tqdm(jump_indices)):
            # print(curr_frame, i)
            plt.clf()
            # grab images
            imgs = vids.get_data(i)
            kpts_2d = pred_2d[cam][curr_frame]
            
            # temp_kpts_2d = np.r_[kpts_2d[0:6,:],kpts_2d[8:,:]]
            
            # Zoom in based on keypoints
            # adjust_viewport(temp_kpts_2d, margin=150)  # Adjust margin as needed for best fit


            plt.imshow(imgs)
            plt.scatter(kpts_2d[:, 0], kpts_2d[:, 1], marker='.', color='white', linewidths=2, alpha=0.5) #point size

            # for color, (index_from, index_to) in zip(COLOR, CONNECTIVITY):
            #     xs, ys = [np.array([kpts_2d[index_from, j], kpts_2d[index_to, j]]) for j in range(2)]
            #     plt.plot(xs, ys, c=color, lw=2) #line error
            #     del xs, ys

            plt.title(vid_title)
            plt.axis("off")
            
            
            writer.grab_frame()


def generate_com_video(com_data, base_folder, graph_title, save_path, cam='Camera1'):
    #######
    N_FRAMES = 2
    START_FRAME = 0
    #######
    label3d_path = find_calib_file(base_folder)
    video_path = os.path.join(base_folder, f'videos/{cam}/0.mp4')
    vid_title = graph_title
    VID_NAME = vid_title + 'continued.mp4'

    ###############################################################################################################
    # load camera parameters
    cameras = load_cameras(label3d_path)

    # get dannce predictions
    # pred_3d = sio.loadmat(pred_path)['pred'][START_FRAME: START_FRAME+N_FRAMES]
    pts = com_data[START_FRAME: START_FRAME+N_FRAMES] #
    # print(len(pts))
    # N_FRAMES = len(jump_indices) #jump_indices
    # print('N_FRAMES', N_FRAMES)
    # compute projections
    pred_2d = {}
    # pose_3d = np.transpose(pred_3d, (0, 2, 1))
    # pts = np.reshape(pose_3d, (-1, 3))

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
    print('pred_2d',len(projpts))


    del projpts#, pred_3d

    # open videos
    vids = imageio.get_reader(video_path)

    # set up video writer
    metadata = dict(title='dannce_visualization', artist='Matplotlib')
    writer = FFMpegWriter(fps=30, metadata=metadata) # orig fps = 30., sihan uses 20, use 0.5 for com debug

    ###############################################################################################################
    fig = plt.figure()
    plt.rcParams['figure.figsize'] = (6, 6)

    with writer.saving(fig, os.path.join(save_path, "vis_"+VID_NAME), dpi=300):
        for curr_frame, i in enumerate(tqdm.tqdm(range(N_FRAMES))): #jump_indices , i
            # print(curr_frame, i)
            plt.clf()
            # grab images
            # imgs = vids.get_data(i)
            imgs = vids.get_data(i+START_FRAME)
            # imgs = [vids.get_data(curr_frame+START_FRAME)][0]
            kpts_2d = pred_2d[cam][curr_frame]
            
            # temp_kpts_2d = np.r_[kpts_2d[0:6,:],kpts_2d[8:,:]]
            
            # Zoom in based on keypoints
            # adjust_viewport(temp_kpts_2d, margin=150)  # Adjust margin as needed for best fit


            plt.imshow(imgs)
            plt.scatter(kpts_2d[:, 0], kpts_2d[:, 1], marker='.', color='white', linewidths=2, alpha=0.5) #point size

            # for color, (index_from, index_to) in zip(COLOR, CONNECTIVITY):
            #     xs, ys = [np.array([kpts_2d[index_from, j], kpts_2d[index_to, j]]) for j in range(2)]
            #     plt.plot(xs, ys, c=color, lw=2) #line error
            #     del xs, ys

            plt.title(vid_title)
            plt.axis("off")
            
            
            writer.grab_frame()


def plot_com_all(base_folder, com_folder_name='COM/predict00', perform_jump_indices=False, perform_video_generation=False, perform_generate_com_video=False, zmin=-10, zmax=30):
    # base_base_folder = os.path.dirname(os.path.normpath(base_folder))
    folder_name = os.path.basename(os.path.normpath(base_folder))
    graph_title = f'COM_{folder_name}' #240716weightsCOM_
    com_folder = os.path.join(base_folder, com_folder_name) #COM_240716weights/predict_results f'{com_folder_name}/predict_results'
    com_path = os.path.join(com_folder, 'com3d0.mat')
    
    if os.path.exists(com_path):
        com_folder_save = os.path.join(com_folder, 'vis')
        if not os.path.exists(com_folder_save):
            os.makedirs(com_folder_save)
        
        com_data = load_com(com_path)
        print(f"plotting com_traga for {base_folder}")
        plot_3d_trajectory_com(com_data, graph_title, com_folder_save, zmin, zmax)
        analyze_com_trajectory(com_data, com_folder_save)
        
        if perform_jump_indices:
            jump_indices = detect_jumps(com_data, com_folder_save)
        
            if perform_video_generation:
                # save_path = os.path.join(com_folder, 'vis')
                # if not os.path.exists(save_path):
                #     os.makedirs(save_path)
                generate_jump_video(com_data, base_folder, jump_indices, graph_title, com_folder_save, cam='Camera1')
        
        if perform_generate_com_video:
            generate_com_video(base_folder, graph_title, com_folder_save, cam='Camera1')
    else:
        print(f"no com file found for {base_folder}")




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


# def generate_dannce_vid_seq(base_path, pred_name="AVG", cam="Camera6", N_FRAMES=1, START_FRAME=30, smooth = False):
#     ###############################################################################################################

#     pred_folder = "DANNCE/predict00"
#     video_path = os.path.join(base_path, f'videos/{cam}/0.mp4')
#     label3d_path = find_calib_file(base_path)
#     smoothed = "smoothed_prediction_AVG0.mat"
#     avg0 = f"save_data_{pred_name}.mat"
#     if smooth:
#         pred_path = os.path.join(base_path, pred_folder, smoothed)
#         vid_title = f'combined_{cam}_smoothed_{N_FRAMES}_start{START_FRAME}'
#     else:
#         pred_path = os.path.join(base_path, pred_folder, avg0)
#         vid_title = f'combined_{cam}_{pred_name}_{N_FRAMES}_start{START_FRAME}'
     
#     # N_FRAMES = 1000
#     # START_FRAME = 0
#     ANIMAL= 'mouse20'
    
#     VID_NAME = vid_title + '.mp4'
#     COLOR = connectivity.COLOR_DICT[ANIMAL]
#     CONNECTIVITY = connectivity.CONNECTIVITY_DICT[ANIMAL]
#     save_path = os.path.join(base_path, pred_folder, 'vis') #os.path.join(pred_path, 'vis')
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     com_file = os.path.join(base_path,pred_folder,'com3d_used.mat')
#     com_data = sio.loadmat(com_file)
#     ###############################################################################################################
#     # load camera parameterss
#     cameras = load_cameras(label3d_path)

#     # get dannce predictions
#     pred_3d = sio.loadmat(pred_path)['pred'][START_FRAME: START_FRAME+N_FRAMES]

#     # compute projections
#     pred_2d = {}
#     pose_3d = np.transpose(pred_3d, (0, 2, 1))
#     pts = np.reshape(pose_3d, (-1, 3))


#     # get the 2d projection
#     projpts = project_to_2d(pts,
#                             cameras[cam]["K"],
#                             cameras[cam]["r"],
#                             cameras[cam]["t"])[:, :2]

#     projpts = distortPoints(projpts,
#                             cameras[cam]["K"],
#                             np.squeeze(cameras[cam]["RDistort"]),
#                             np.squeeze(cameras[cam]["TDistort"]))
#     projpts = projpts.T
#     projpts = np.reshape(projpts, (N_FRAMES, -1, 2))
#     pred_2d[cam] = projpts


#     del projpts, pred_3d


#     ###############################3
#     # for com
#     pts_com = com_data['com'][START_FRAME: START_FRAME+N_FRAMES]
#     pred_2d_com = {}
#     # Get the 2d projection for com
#     projpts_com = project_to_2d(pts_com,
#                                 cameras[cam]["K"],
#                                 cameras[cam]["r"],
#                                 cameras[cam]["t"])[:, :2]

#     projpts_com = distortPoints(projpts_com,
#                                 cameras[cam]["K"],
#                                 np.squeeze(cameras[cam]["RDistort"]),
#                                 np.squeeze(cameras[cam]["TDistort"]))
#     projpts_com = projpts_com.T
#     projpts_com = np.reshape(projpts_com, (N_FRAMES, -1, 2))
#     pred_2d_com[cam] = projpts_com
#     del projpts_com
#     #####################3


#     # open videos
#     vids = imageio.get_reader(video_path)

#     # set up video writer
#     metadata = dict(title='combined_visualization', artist='Matplotlib')
#     writer = FFMpegWriter(fps=20, metadata=metadata) # orig fps = 30.

#     ###############################################################################################################
#     fig = plt.figure()
#     plt.rcParams['figure.figsize'] = (6, 6)





#     with writer.saving(fig, os.path.join(save_path, "vis_"+VID_NAME), dpi=300):
#         for curr_frame in tqdm.tqdm(range(N_FRAMES)):
#             plt.clf()
#             # grab images
#             imgs = [vids.get_data(curr_frame+START_FRAME)][0]
#             kpts_2d = pred_2d[cam][curr_frame]
            
#             temp_kpts_2d = np.r_[kpts_2d[0:6,:],kpts_2d[8:,:]]

#             # Plot com keypoints
#             kpts_2d_com = pred_2d_com[cam][curr_frame]
#             temp_kpts_2d_com = np.r_[kpts_2d_com[0:6,:],kpts_2d_com[8:,:]]
            
#             # Zoom in based on keypoints
#             adjust_viewport(temp_kpts_2d, margin=450)  # Adjust margin as needed for best fit 150 is good.


#             plt.imshow(imgs)
            
#             # Plot com points
#             plt.scatter(kpts_2d_com[:, 0], kpts_2d_com[:, 1], marker='.', color='red', linewidths=2, alpha=0.5)

#             plt.scatter(temp_kpts_2d[:, 0], temp_kpts_2d[:, 1], marker='.', color='white', linewidths=2, alpha=0.5) #point size

#             for color, (index_from, index_to) in zip(COLOR, CONNECTIVITY):
#                 xs, ys = [np.array([kpts_2d[index_from, j], kpts_2d[index_to, j]]) for j in range(2)]
#                 plt.plot(xs, ys, c=color, lw=2) #line error
#                 del xs, ys

#             plt.title(vid_title)
#             plt.axis("off")
            
#             writer.grab_frame()

# def generate_dannce_vid_seq(base_path, pred_name="AVG", cam="Camera6", N_FRAMES=1, START_FRAME=30, smooth=False):
#     pred_folder = "DANNCE/predict00"
#     video_path = os.path.join(base_path, f'videos/{cam}/0.mp4')
#     label3d_path = find_calib_file(base_path)
#     smoothed = "smoothed_prediction_AVG0.mat"
#     avg0 = f"save_data_{pred_name}.mat"
#     if smooth:
#         pred_path = os.path.join(base_path, pred_folder, smoothed)
#         vid_title = f'combined_{cam}_smoothed_{N_FRAMES}_start{START_FRAME}'
#     else:
#         pred_path = os.path.join(base_path, pred_folder, avg0)
#         vid_title = f'combined_{cam}_{pred_name}_{N_FRAMES}_start{START_FRAME}'
    
#     # Define save path
#     save_path = os.path.join(base_path, pred_folder, 'vis')
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     com_file = os.path.join(base_path, pred_folder, 'com3d_used.mat')

#     # Check if com_file exists
#     if not os.path.exists(com_file):
#         print(f"Skipping {base_path} due to missing {com_file}")
#         return

#     # Load the necessary data with error handling
#     try:
#         com_data = sio.loadmat(com_file)
#         pred_3d = sio.loadmat(pred_path)['pred'][START_FRAME: START_FRAME+N_FRAMES]
#     except Exception as e:
#         print(f"Error loading data in {base_path}: {e}")
#         return

#     # Debugging output for shapes
#     print(f"Initial shape of pred_3d: {pred_3d.shape}")
#     print(f"Shape of com_data['com']: {com_data['com'].shape}")

#     # Adjust pred_3d to the expected shape if necessary
#     # if pred_3d.shape[0] == 1 and pred_3d.shape[1] == 1:
#     #     pred_3d = np.squeeze(pred_3d, axis=(0, 1))  # Remove singleton dimensions

#     # print(f"Adjusted shape of pred_3d: {pred_3d.shape}")
    
#     # if pred_3d.shape[1:] != (3, 22):
#     #     print(f"Skipping {base_path} due to incompatible pred_3d shape: {pred_3d.shape}")
#     #     return

#     # Proceed with projections and further processing
#     cameras = load_cameras(label3d_path)

#     try:
#         # Reshape and project points
#         pose_3d = np.transpose(pred_3d, (0, 2, 1))
#         pts = np.reshape(pose_3d, (-1, 3))
        
#         projpts = project_to_2d(pts,
#                                 cameras[cam]["K"],
#                                 cameras[cam]["r"],
#                                 cameras[cam]["t"])[:, :2]
#         projpts = distortPoints(projpts,
#                                 cameras[cam]["K"],
#                                 np.squeeze(cameras[cam]["RDistort"]),
#                                 np.squeeze(cameras[cam]["TDistort"]))
#         projpts = projpts.T
#         projpts = np.reshape(projpts, (N_FRAMES, -1, 2))
#         pred_2d = {cam: projpts}
        
#     except ValueError as e:
#         print("pts", pts)
#         print(f"Array shape mismatch in {base_path}: {e}")
#         return

#     # Processing com data for 2D projection
#     try:
#         pts_com = com_data['com'][START_FRAME: START_FRAME+N_FRAMES]
#         projpts_com = project_to_2d(pts_com,
#                                     cameras[cam]["K"],
#                                     cameras[cam]["r"],
#                                     cameras[cam]["t"])[:, :2]
#         projpts_com = distortPoints(projpts_com,
#                                     cameras[cam]["K"],
#                                     np.squeeze(cameras[cam]["RDistort"]),
#                                     np.squeeze(cameras[cam]["TDistort"]))
#         projpts_com = projpts_com.T
#         projpts_com = np.reshape(projpts_com, (N_FRAMES, -1, 2))
#         pred_2d_com = {cam: projpts_com}
        
#     except ValueError as e:
#         print(f"Error projecting com data in {base_path}: {e}")
#         return

#     # Open video reader
#     vids = imageio.get_reader(video_path)
#     metadata = dict(title='combined_visualization', artist='Matplotlib')
#     writer = FFMpegWriter(fps=20, metadata=metadata)

#     fig = plt.figure()
#     plt.rcParams['figure.figsize'] = (6, 6)

#     with writer.saving(fig, os.path.join(save_path, "vis_" + vid_title + '.mp4'), dpi=300):
#         for curr_frame in tqdm.tqdm(range(N_FRAMES)):
#             plt.clf()
#             imgs = vids.get_data(curr_frame + START_FRAME)
#             kpts_2d = pred_2d[cam][curr_frame]
#             kpts_2d_com = pred_2d_com[cam][curr_frame]
            
#             # Adjust the view to the detected keypoints
#             adjust_viewport(kpts_2d, margin=450)

#             plt.imshow(imgs)
#             plt.scatter(kpts_2d_com[:, 0], kpts_2d_com[:, 1], marker='.', color='red', linewidths=2, alpha=0.5)
#             plt.scatter(kpts_2d[:, 0], kpts_2d[:, 1], marker='.', color='white', linewidths=2, alpha=0.5)

#             for color, (index_from, index_to) in zip(COLOR, CONNECTIVITY):
#                 xs, ys = [np.array([kpts_2d[index_from, j], kpts_2d[index_to, j]]) for j in range(2)]
#                 plt.plot(xs, ys, c=color, lw=2)

#             plt.title(vid_title)
#             plt.axis("off")
#             writer.grab_frame()
