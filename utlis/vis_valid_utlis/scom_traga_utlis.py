import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import medfilt
import imageio
from matplotlib.animation import FFMpegWriter
import tqdm
import utlis.connectivity as connectivity
from utlis.projection import *
from utlis.sync_utlis.sync_df_utlis import find_calib_file
import shutil
from scipy.signal import medfilt
# ——————————————————————————————————————————————————————————
# assume the following single-animal functions are defined as before:
#   load_com, plot_3d_trajectory_com, analyze_com_trajectory,
#   detect_jumps, generate_jump_video, adjust_viewport, generate_com_video
# ——————————————————————————————————————————————————————————

def load_com(file_path):
    data = sio.loadmat(file_path)
    return data['com']  # shape (n_frames, 3) or (n_frames, 3 * n_animals)

# … your original plot_3d_trajectory_com, analyze_com_trajectory, etc. …

# ——————————————————————————————————————————————————————————
# New multi-animal ("social") functions
# ——————————————————————————————————————————————————————————

def plot_social_3d_trajectory(com_data, graph_title, save_folder, zmin, zmax):
    """
    For each animal channel in com_data (n_frames × 3 × n_animals),
    create a separate 3D time-colored scatter.
    """
    n_animals = com_data.shape[2]
    time_steps = np.arange(com_data.shape[0])
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(time_steps.min(), time_steps.max())

    for i in range(n_animals):
        x, y, z = com_data[:,0,i], com_data[:,1,i], com_data[:,2,i]
        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection='3d')
        sc  = ax.scatter(x, y, z, c=time_steps, cmap=cmap, norm=norm, marker='o')
        ax.set_title(f'{graph_title}_animal{i+1}')
        ax.set_xlabel('X Position'); ax.set_ylabel('Y Position'); ax.set_zlabel('Z Position')
        ax.set_zlim(zmin, zmax)
        fig.colorbar(sc, ax=ax, label='Time Step')
        out = os.path.join(save_folder, f'com_animal{i+1}_3d_trajectory_plot.jpg')
        plt.savefig(out, format='jpg')
        plt.show(); plt.close()

def analyze_social_com_trajectory(com_data, save_folder):
    """
    Smooth each animal’s COM over time, plot trajectories, 
    compute and histogram speeds, and save results.
    """
    n_animals   = com_data.shape[2]
    kernel_size = 31

    for i in range(n_animals):
        data = medfilt(com_data[:,:,i], kernel_size=(kernel_size,1))
        # smoothed trajectory
        plt.figure(figsize=(12,6))
        plt.plot(data)
        plt.title(f'Smoothed COM Trajectory Animal{i+1}')
        plt.xlabel('Frame'); plt.ylabel('Position (mm)')
        plt.legend(['X','Y','Z']); plt.tight_layout()
        out1 = os.path.join(save_folder, f'com_animal{i+1}_trajectory_plot.jpg')
        plt.savefig(out1, format='jpg'); plt.show(); plt.close()

        # speed histogram
        diffs = np.diff(data, axis=0)
        speed = np.linalg.norm(diffs, axis=1)
        bins  = np.arange(0, 20+0.1, 0.1)
        counts, edges = np.histogram(speed, bins=bins)

        plt.figure(figsize=(12,6))
        plt.plot(edges[:-1], counts)
        plt.title(f'Speed Distribution Animal{i+1}')
        plt.xlabel('Speed (mm/frame)'); plt.ylabel('Number of Frames')
        plt.tight_layout()
        out2 = os.path.join(save_folder, f'speed_histogram_animal{i+1}.jpg')
        plt.savefig(out2, format='jpg'); plt.show(); plt.close()

        np.save(os.path.join(save_folder, f'speed_data_animal{i+1}.npy'), speed)

def detect_social_jumps(com_data, save_folder, threshold_factor=6):
    """
    Find frames where COM ‘jumps’ for each animal.
    Returns dict: { animal_index: jump_indices_array }.
    """
    n_animals = com_data.shape[2]
    jumps_dict = {}

    for i in range(n_animals):
        pos = com_data[:,:,i]
        diffs = np.diff(pos, axis=0)
        mag   = np.linalg.norm(diffs, axis=1)
        thresh = mag.mean() + threshold_factor * mag.std()
        idxs   = np.where(mag > thresh)[0]
        jumps_dict[i] = idxs

        plt.figure(figsize=(12,6))
        plt.plot(mag, label='Magnitude of Differences')
        plt.scatter(idxs, mag[idxs], c='red', label='Significant Jumps')
        plt.axhline(thresh, color='orange', linestyle='--', label='Threshold')
        plt.xlabel('Frame'); plt.ylabel('Magnitude of Position Difference')
        plt.title(f'Detection of Jumps Animal{i+1}')
        plt.legend()
        out = os.path.join(save_folder, f'com_trajectory_jumps_animal{i+1}.jpg')
        plt.savefig(out, format='jpg'); plt.show(); plt.close()

        np.save(os.path.join(save_folder, f'com_jump_indices_animal{i+1}.npy'), idxs)

    print("Frames with significant jumps per animal:", jumps_dict)
    return jumps_dict

def generate_social_jump_videos(com_data, base_folder, graph_title, save_folder, cam='Camera1'):
    """
    For each animal, render short videos centered on jump frames.
    """
    jumps = detect_social_jumps(com_data, save_folder)
    calib = find_calib_file(base_folder)
    cams  = load_cameras(calib)
    vid_p = os.path.join(base_folder, f'videos/{cam}/0.mp4')
    reader= imageio.get_reader(vid_p)
    meta  = dict(title='dannce_visualization', artist='Matplotlib')

    for i, idxs in jumps.items():
        if idxs.size == 0:
            continue
        pts = com_data[idxs, :, i]
        proj = project_to_2d(pts, cams[cam]["K"], cams[cam]["r"], cams[cam]["t"])[:,:2]
        proj = distortPoints(proj, cams[cam]["K"],
                             np.squeeze(cams[cam]["RDistort"]),
                             np.squeeze(cams[cam]["TDistort"]))
        proj = proj.T.reshape(len(idxs), -1, 2)

        writer = FFMpegWriter(fps=4, metadata=meta)
        fig    = plt.figure(figsize=(6,6))
        out_v  = os.path.join(save_folder, f'vis_{graph_title}_animal{i+1}.mp4')

        with writer.saving(fig, out_v, dpi=300):
            for frame_idx, jump_frame in enumerate(idxs):
                plt.clf()
                img  = reader.get_data(jump_frame)
                kpts = proj[frame_idx]
                plt.imshow(img)
                plt.scatter(kpts[:,0], kpts[:,1], marker='.', alpha=0.5)
                plt.axis('off')
                writer.grab_frame()

def generate_social_com_video(com_data, base_folder, graph_title, save_folder, cam='Camera1', num_frames=2):
    """
    For each animal, render the first num_frames of COM overlaid on the video.
    """
    calib = find_calib_file(base_folder)
    cams  = load_cameras(calib)
    vid_p = os.path.join(base_folder, f'videos/{cam}/0.mp4')
    reader= imageio.get_reader(vid_p)
    meta  = dict(title='dannce_visualization', artist='Matplotlib')
    n_animals = com_data.shape[2]

    for i in range(n_animals):
        pts = com_data[:num_frames, :, i]
        proj = project_to_2d(pts, cams[cam]["K"], cams[cam]["r"], cams[cam]["t"])[:,:2]
        proj = distortPoints(proj, cams[cam]["K"],
                             np.squeeze(cams[cam]["RDistort"]),
                             np.squeeze(cams[cam]["TDistort"]))
        proj = proj.T.reshape(num_frames, -1, 2)

        writer = FFMpegWriter(fps=30, metadata=meta)
        fig    = plt.figure(figsize=(6,6))
        out_v  = os.path.join(save_folder, f'vis_{graph_title}_continued_animal{i+1}.mp4')

        with writer.saving(fig, out_v, dpi=300):
            for f in range(num_frames):
                plt.clf()
                img  = reader.get_data(f)
                kpts = proj[f]
                plt.imshow(img)
                plt.scatter(kpts[:,0], kpts[:,1], marker='.', alpha=0.5)
                plt.axis('off')
                writer.grab_frame()

# ——————————————————————————————————————————————————————————
# Updated master function
# ——————————————————————————————————————————————————————————

def plot_com_all_social(
    base_folder,
    com_folder_name='COM/predict00',
    perform_jump_indices=False,
    perform_video_generation=False,
    perform_generate_com_video=False,
    zmin=-10,
    zmax=30
):
    """
    Detect single vs. multi-animal COM, reshape accordingly,
    and run either the single-animal or social pipeline.
    """
    folder    = os.path.basename(os.path.normpath(base_folder))
    title     = f'COM_{folder}'
    com_dir   = os.path.join(base_folder, com_folder_name)
    com_mat   = os.path.join(com_dir, 'com3d0.mat')
    if not os.path.exists(com_mat):
        print(f"No COM file found for {base_folder}")
        return

    raw = load_com(com_mat)
    print(f"plotting com_traga for {base_folder}")
    # if your MAT stores shape (frames, 3, n_animals) already, just use it
    if raw.ndim == 3 and raw.shape[1] == 3:
        com_data = raw

    # otherwise if it’s flattened (frames, 3*n_animals), reshape it
    elif raw.ndim == 2 and raw.shape[1] % 3 == 0:
        n_animals = raw.shape[1] // 3
        com_data  = raw.reshape(-1, 3, n_animals)

    else:
        raise ValueError(f"Unexpected COM array shape {raw.shape}, " +
                         "expected (frames,3) or (frames,3*n_animals) or (frames,3,n_animals)")

    # now both branches yield com_data.shape == (frames, 3, n_animals)
    n_animals = com_data.shape[2]


    vis_folder = os.path.join(com_dir, 'vis')
    os.makedirs(vis_folder, exist_ok=True)

    if n_animals == 1:
        # single-animal branch
        single = com_data[:, :, 0]
        plot_3d_trajectory_com(single, title, vis_folder, zmin, zmax)
        analyze_com_trajectory(single, vis_folder)
        if perform_jump_indices:
            jumps = detect_jumps(single, vis_folder)
            if perform_video_generation:
                generate_jump_video(single, base_folder, jumps, title, vis_folder)
        if perform_generate_com_video:
            generate_com_video(single, base_folder, title, vis_folder)
    else:
        # multi-animal (social) branch
        plot_social_3d_trajectory(com_data, title, vis_folder, zmin, zmax)
        analyze_social_com_trajectory(com_data, vis_folder)
        if perform_jump_indices:
            detect_social_jumps(com_data, vis_folder)
            if perform_video_generation:
                generate_social_jump_videos(com_data, base_folder, title, vis_folder)
        if perform_generate_com_video:
            generate_social_com_video(com_data, base_folder, title, vis_folder)
