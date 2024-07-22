import os
import re
import numpy as np
import scipy.io as sio
import imageio
import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
# from utlis.projection import *
from utlis.projection import *
import shutil

def load_data(file_path):
    data = sio.loadmat(file_path)
    return data['com']

def plot_3d_trajectory(com_data, graph_title, com_folder_save):
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
    threshold = np.mean(d_magnitude) + 10 * np.std(d_magnitude) # used to be 2*, but can change

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


# def calculate_velocity(com_data, time_interval=1):
#     # Calculate the difference between consecutive positions
#     delta_positions = np.diff(com_data, axis=0)
#     # Calculate the velocity by dividing by the time interval
#     velocities = delta_positions / time_interval
#     return velocities

# def plot_velocity(velocities, graph_title, com_folder_save):
#     # Calculate the magnitude of velocity
#     velocity_magnitudes = np.linalg.norm(velocities, axis=1)
#     # Use the index as a proxy for time
#     time_steps = np.arange(len(velocity_magnitudes))

#     # Plotting the velocity magnitude
#     plt.figure(figsize=(10, 6))
#     plt.plot(time_steps, velocity_magnitudes, marker='o', linestyle='-')
#     plt.title(graph_title)
#     plt.xlabel('Time Step')
#     plt.ylabel('Velocity Magnitude')
#     plt.grid(True)
#     plt.savefig(os.path.join(com_folder_save, 'com_velocity_plot.jpg'), format='jpg')
#     plt.show()
#     plt.close()

def find_missing_folders(synced_name_path):
    missing_folders = []
    for npy_file in os.listdir(synced_name_path):
        if npy_file.endswith('_missing_folders.npy'):
            npy_file_path = os.path.join(synced_name_path, npy_file)
            if os.path.isfile(npy_file_path):
                npy_missing_folders = np.load(npy_file_path)
                missing_folders.extend(npy_missing_folders.tolist())
    return missing_folders

def process_folders(dates, synced_name_path, summ_folder, operations):
    # Retrieve missing folders
    missing_folders = find_missing_folders(synced_name_path)

    # Print the missing folders to verify
    print("Missing folders:", len(missing_folders))

    # Process each folder
    processed_count = 0
    for date in dates:
        base_base_folder = os.path.join(summ_folder, date)
        for folder_name in os.listdir(base_base_folder):
            base_folder = os.path.join(base_base_folder, folder_name)
            if os.path.isdir(base_folder):
                if re.match(r'^\d', folder_name) and base_folder not in missing_folders:
                    print('Processing folder:', base_folder)
                    processed_count += 1
                    for operation_func, params in operations:
                        if params:
                            operation_func(base_folder, **params)
                        else:
                            operation_func(base_folder)

    print("processed folder: ", processed_count)




def find_calib_file(base_folder):
    for file_name in os.listdir(base_folder):
        if file_name.endswith('label3d_dannce.mat'):
            return os.path.join(base_folder, file_name)
    return None

def generate_jump_video(com_data, base_folder, jump_indices, graph_title, save_path, cam='Camera1'):
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
            
            temp_kpts_2d = np.r_[kpts_2d[0:6,:],kpts_2d[8:,:]]
            
            # Zoom in based on keypoints
            # adjust_viewport(temp_kpts_2d, margin=150)  # Adjust margin as needed for best fit


            plt.imshow(imgs)
            plt.scatter(temp_kpts_2d[:, 0], temp_kpts_2d[:, 1], marker='.', color='white', linewidths=2, alpha=0.5) #point size

            # for color, (index_from, index_to) in zip(COLOR, CONNECTIVITY):
            #     xs, ys = [np.array([kpts_2d[index_from, j], kpts_2d[index_to, j]]) for j in range(2)]
            #     plt.plot(xs, ys, c=color, lw=2) #line error
            #     del xs, ys

            plt.title(vid_title)
            plt.axis("off")
            
            
            writer.grab_frame()


def plot_com_all(base_folder, com_folder_name, perform_jump_indices=False, perform_video_generation=False):
    # base_base_folder = os.path.dirname(os.path.normpath(base_folder))
    folder_name = os.path.basename(os.path.normpath(base_folder))
    graph_title = f'24_07_05_COM_{folder_name}'
    com_folder = os.path.join(base_folder, f'{com_folder_name}/predict_results') #COM_240716weights/predict_results
    com_path = os.path.join(com_folder, 'com3d.mat')
    if os.path.exists(com_path):
        com_folder_save = os.path.join(com_folder, 'vis')
        if not os.path.exists(com_folder_save):
            os.makedirs(com_folder_save)
        
        com_data = load_data(com_path)
        plot_3d_trajectory(com_data, graph_title, com_folder_save)
        
        if perform_jump_indices:
            jump_indices = detect_jumps(com_data, com_folder_save)
        
        if perform_video_generation:
            save_path = os.path.join(com_folder, 'vis')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            generate_jump_video(com_data, base_folder, jump_indices, graph_title, save_path, cam='Camera1')


def temp_change_calib_pos_to_0(base_folder):
    for file in os.listdir(base_folder):
        if file.startswith('0') and file.endswith('.mat'):
            print("Calibration file starting with '0' already exists. Skipping processing.")
            return
    calib_path = find_calib_file(base_folder)
    if calib_path is None:
        print("No calib file found.")
        return

    
    mat_data = sio.loadmat(calib_path)
    sync = mat_data["sync"]

    for cam_idx in range(6):
        camera_key = f'Camera{cam_idx+1}'
        # print(sync[cam_idx][0]['data_sampleID'][0][0][0])
        sync[cam_idx][0]['data_sampleID'][0][0][0] = np.nan_to_num(sync[cam_idx][0]['data_sampleID'][0][0][0], nan=0)
        # print(2)
        # print(sync[cam_idx][0]['data_sampleID'][0][0][0])

    mat_data["sync"] = sync

    original_filename = os.path.basename(calib_path)
    new_filename = original_filename.replace('pos', '0')
    new_calib_path = os.path.join(base_folder, new_filename)

    sio.savemat(new_calib_path, mat_data)
    
    # Create a new folder for the original .mat file
    new_folder = os.path.join(base_folder, 'prev_calib')
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    
    # Move the original .mat file to the new folder
    shutil.move(calib_path, os.path.join(new_folder, os.path.basename(calib_path)))

    print(f"Original .mat file moved to {new_folder}")
    print(f"Modified .mat file saved as {new_calib_path}")



def generate_dannce_vid_seq(base_path, pred_folder, cam, vid_title):
    video_path = os.path.join(base_path, f'videos/{cam}/0.mp4')
    label3d_path = find_calib_file(base_path)


    