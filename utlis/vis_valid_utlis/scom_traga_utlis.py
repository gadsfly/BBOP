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

from utlis.vis_valid_utlis.com_trag_updated import plot_3d_trajectory_com, analyze_com_trajectory, detect_jumps,generate_jump_video,generate_com_video

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


def generate_social_com_video(
    com_data,
    base_folder,
    graph_title,
    save_folder,
    cam='Camera1',
    start_frame=0,
    num_frames=100,
    flip_x=False
):
    """
    Render COM trajectories for all animals on one video,
    starting at `start_frame` for both data and video.
    """
    # load calibration & camera params
    calib = find_calib_file(base_folder)
    cams  = load_cameras(calib)
    K     = cams[cam]["K"]
    Rdst  = np.squeeze(cams[cam]["RDistort"])
    Tdst  = np.squeeze(cams[cam]["TDistort"])
    r_vec = cams[cam]["r"]
    t_vec = cams[cam]["t"]

    # open video reader
    vid_path = os.path.join(base_folder, f'videos/{cam}/0.mp4')
    reader   = imageio.get_reader(vid_path)

    # project COM into 2D once, for the requested slice
    n_animals = com_data.shape[2]
    projs = []
    for ai in range(n_animals):
        # slice COM frames, then project
        slice_3d = com_data[start_frame:start_frame+num_frames, :, ai]
        uvw = project_to_2d(slice_3d, K, r_vec, t_vec)      # (num_frames, 3)
        xy  = uvw[:, :2]                                   # (num_frames, 2)
        distorted = distortPoints(xy, K, Rdst, Tdst)       # (num_frames, 2)
        proj_i    = distorted.T.reshape(num_frames, -1, 2) # (num_frames, 1, 2)
        projs.append(proj_i)

    # prepare writer
    metadata = dict(title='dannce_visualization', artist='Matplotlib')
    out_name = f'vis_{graph_title}_combined_start{start_frame}.mp4'
    out_path = os.path.join(save_folder, out_name)
    writer   = FFMpegWriter(fps=30, metadata=metadata)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')

    with writer.saving(fig, out_path, dpi=300):
        for idx in range(num_frames):
            frame_idx = start_frame + idx
            ax.clear()
            frame = reader.get_data(frame_idx)
            ax.imshow(frame)
            h, w, _ = frame.shape

            # overlay each animal
            for ai, proj_i in enumerate(projs):
                pts = proj_i[idx]  # shape (1,2)
                if flip_x:
                    pts = pts.copy()
                    pts[:, 0] = w - pts[:, 0]
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    s=20,
                    alpha=0.8,
                    label=f'Animal {ai+1}'
                )

            ax.legend(loc='upper right', fontsize='small')
            writer.grab_frame()

    plt.close(fig)
    print(f"Saved combined COM video starting at frame {start_frame} → {out_path}")

# below function will generate the videos separately, which is unnecessary for a smooth generation
# def (com_data, base_folder, graph_title, save_folder, cam='Camera1', num_frames=100):
#     """
#     For each animal, render the first num_frames of COM overlaid on the video.
#     """
#     calib = find_calib_file(base_folder)
#     cams  = load_cameras(calib)
#     vid_p = os.path.join(base_folder, f'videos/{cam}/0.mp4')
#     reader= imageio.get_reader(vid_p)
#     meta  = dict(title='dannce_visualization', artist='Matplotlib')
#     n_animals = com_data.shape[2]

#     for i in range(n_animals):
#         pts = com_data[:num_frames, :, i]
#         proj = project_to_2d(pts, cams[cam]["K"], cams[cam]["r"], cams[cam]["t"])[:,:2]
#         proj = distortPoints(proj, cams[cam]["K"],
#                              np.squeeze(cams[cam]["RDistort"]),
#                              np.squeeze(cams[cam]["TDistort"]))
#         proj = proj.T.reshape(num_frames, -1, 2)

#         writer = FFMpegWriter(fps=30, metadata=meta)
#         fig    = plt.figure(figsize=(6,6))
#         out_v  = os.path.join(save_folder, f'vis_{graph_title}_continued_animal{i+1}.mp4')

#         with writer.saving(fig, out_v, dpi=300):
#             for f in range(num_frames):
#                 plt.clf()
#                 img  = reader.get_data(f)
#                 kpts = proj[f]
#                 plt.imshow(img)
#                 plt.scatter(kpts[:,0], kpts[:,1], marker='.', alpha=0.5)
#                 plt.axis('off')
#                 writer.grab_frame()

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



def standalone_generate_social_com_video(base_folder,
    com_folder_name='COM/predict00',    cam='Camera1',
    start_frame=100,
    num_frames=150):
    """
    Detect single vs. multi-animal COM, reshape accordingly,
    and run either the single-animal or social pipeline.
    """
    folder    = os.path.basename(os.path.normpath(base_folder))
    title     = f'2COM_{folder}_{start_frame}_{num_frames}'
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

    generate_social_com_video(com_data, base_folder, title, vis_folder,    cam,
    start_frame,
    num_frames)

#################33 new functions for com distance comparison##########################

import os
import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt

def com_distance_qc(
    base_folder,
    com_folder_name='COM/predict00',
    prefer_filtered=False,#True,    # try vis/com_filtered_win*.npy first
    win=None,                # choose a specific window size (int); if None pick the smallest available
    jump_k=6.0,              # threshold = mean(|Δdist|) + k * std(|Δdist|)
    first_n=None,            # only evaluate the first N frames if set
    save_csv=True,
    save_plot=True,
    show_plot=False
):
    """
    Quick QC of inter-animal distance.

    Loads COM (prefers filtered .npy if available), computes dist(t) = ||A0(t)-A1(t)||,
    flags 'jump' frames where |Δdist| exceeds mean + k*std, and writes a small CSV.

    Returns
    -------
    out : dict with keys
        'dist'         : np.ndarray [T]
        'jump_frames'  : np.ndarray [M]   # frames where the jump is flagged (uses Δdist, so indices are +1)
        'summary_df'   : pd.DataFrame(1 row)
        'used_source'  : 'filtered_winXX' or 'raw'
        'used_path'    : path to loaded array
    """
    com_dir   = os.path.join(base_folder, com_folder_name)
    vis_dir   = os.path.join(com_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    # ---- pick source: filtered npy or raw mat ----
    used_source = 'raw'
    used_path   = None
    com = None

    if prefer_filtered:
        # list vis/com_filtered_win*.npy
        if os.path.isdir(vis_dir):
            def extract_w(fname):
                # "com_filtered_win05.npy" -> 5
                name = os.path.splitext(fname)[0]
                if name.startswith('com_filtered_win'):
                    return int(name.replace('com_filtered_win', ''))
                return None
            candidates = []
            for fn in os.listdir(vis_dir):
                if fn.startswith('com_filtered_win') and fn.endswith('.npy'):
                    w = extract_w(fn)
                    if w is not None:
                        candidates.append((w, os.path.join(vis_dir, fn)))
            if candidates:
                if win is not None:
                    # exact window requested
                    matches = [p for (w,p) in candidates if w == int(win)]
                    if matches:
                        used_path = matches[0]
                    else:
                        # fall back to smallest if requested not found
                        used_path = sorted(candidates, key=lambda t: t[0])[0][1]
                else:
                    used_path = sorted(candidates, key=lambda t: t[0])[0][1]  # smallest window
                com = np.load(used_path)            # (T, 3, n_animals)
                used_source = f'filtered_{os.path.basename(used_path)}'

    if com is None:
        # fall back to raw
        mat_path = os.path.join(com_dir, 'com3d0.mat')
        if not os.path.isfile(mat_path):
            raise FileNotFoundError(f"No COM file found at {mat_path} and no filtered .npy in {vis_dir}")
        data = sio.loadmat(mat_path)
        if 'com' not in data:
            raise KeyError(f"'com' key not in {mat_path}")
        raw = data['com']
        if raw.ndim == 3 and raw.shape[1] == 3:
            com = raw
        elif raw.ndim == 2 and raw.shape[1] % 3 == 0:
            n_animals = raw.shape[1] // 3
            com = raw.reshape(-1, 3, n_animals)
        else:
            raise ValueError(f"Unexpected COM shape {raw.shape}")
        used_source = 'raw'
        used_path   = mat_path

    # ---- require at least 2 animals ----
    if com.shape[2] < 2:
        raise ValueError(f"Need >=2 animals; got shape {com.shape}")

    # ---- optionally trim to first_n ----
    if first_n is not None:
        com = com[:first_n, :, :]

    # ---- compute distances between animal 0 and 1 ----
    a0 = com[:, :, 0]     # (T, 3)
    a1 = com[:, :, 1]     # (T, 3)
    dist = np.linalg.norm(a0 - a1, axis=1)  # (T,)

    # ---- detect abrupt changes in the distance curve ----
    d_dist = np.abs(np.diff(dist))          # (T-1,)
    mu  = np.nanmean(d_dist)
    sd  = np.nanstd(d_dist)
    thr = mu + jump_k * sd
    jump_idx = np.where(d_dist > thr)[0] + 1   # assign the jump to the *later* frame

    # ---- small summary ----
    summary = {
        'source'      : [used_source],
        'n_frames'    : [int(dist.size)],
        'mean'        : [float(np.nanmean(dist))],
        'median'      : [float(np.nanmedian(dist))],
        'std'         : [float(np.nanstd(dist))],
        'max'         : [float(np.nanmax(dist))],
        'min'         : [float(np.nanmin(dist))],
        'jump_k'      : [float(jump_k)],
        'jump_thresh' : [float(thr)],
        'n_jumps'     : [int(jump_idx.size)],
    }
    df = pd.DataFrame(summary)

    # ---- save outputs ----
    if save_csv:
        df.to_csv(os.path.join(vis_dir, 'com_distance_qc_summary.csv'), index=False)
        np.save(os.path.join(vis_dir, 'com_distance_qc_jump_frames.npy'), jump_idx)

    if save_plot or show_plot:
        plt.figure(figsize=(10,4))
        plt.plot(np.arange(dist.size), dist, lw=1)
        if jump_idx.size:
            plt.scatter(jump_idx, dist[jump_idx], s=15)
        plt.xlabel('Frame')
        plt.ylabel('Distance (mm)')
        plt.title('Inter-animal COM distance')
        plt.tight_layout()
        if save_plot:
            plt.savefig(os.path.join(vis_dir, 'com_distance_qc_plot.png'), dpi=150)
        if show_plot:
            plt.show()
        plt.close()

    return {
        'dist': dist,
        'jump_frames': jump_idx,
        'summary_df': df,
        'used_source': used_source,
        'used_path': used_path,
    }




#####################################################scom smooth, useless in the future#####################################


# def generate_social_com_video_smooth(
#     com_data,
#     base_folder,
#     graph_title,
#     save_folder,
#     cam='Camera1',
#     start_frame=0,
#     num_frames=60,
#     window_sizes=(1, 5, 15),
# ):
#     """
#     For each median-filter window size in window_sizes, render the COM (smoothed)
#     overlaid on the video frames [start_frame : start_frame+num_frames].
#     Assumes the following are already imported:
#       - find_calib_file, load_cameras, project_to_2d, distortPoints
#       - scipy.signal.medfilt
#       - imageio, numpy as np, matplotlib.pyplot as plt, FFMpegWriter

#     Parameters
#     ----------
#     com_data : np.ndarray
#         Shape (n_frames, 3, n_animals).
#     base_folder : str
#         Root folder containing 'videos/{cam}/0.mp4'.
#     graph_title : str
#         Title fragment for output filenames.
#     save_folder : str
#         Directory where output videos will be saved.
#     cam : str, default 'Camera1'
#         Subfolder under 'videos/' containing "0.mp4".
#     start_frame : int, default 0
#         Index of the first frame to visualize.
#     num_frames : int, default 60
#         Number of consecutive frames (from start_frame) to overlay.
#     window_sizes : iterable of int, default (1,5,15)
#         List of median-filter window sizes (in frames). w=1 means no filtering.
#     """
#     # 1. Load calibration & camera parameters
#     calib = find_calib_file(base_folder)
#     cams  = load_cameras(calib)

#     # 2. Prepare video reader
#     vid_path = os.path.join(base_folder, f'videos/{cam}/0.mp4')
#     if not os.path.exists(vid_path):
#         raise FileNotFoundError(f"No video at {vid_path}")
#     reader = imageio.get_reader(vid_path)
#     fps    = reader.get_meta_data().get('fps', 30)

#     # 3. Ensure save_folder exists
#     os.makedirs(save_folder, exist_ok=True)

#     n_animals = com_data.shape[2]
#     meta      = dict(title='dannce_visualization', artist='Matplotlib')

#     # 4. Loop over each smoothing window
#     for w in window_sizes:
#         # 4a. Apply median filter if w > 1; else copy raw
#         if w > 1:
#             smoothed = np.empty_like(com_data)
#             for ai in range(n_animals):
#                 smoothed[:, :, ai] = medfilt(
#                     com_data[:, :, ai], kernel_size=(w, 1)
#                 )
#         else:
#             smoothed = com_data.copy()

#         # 5. Project smoothed COM to 2D (only the [start_frame:start_frame+num_frames] segment)
#         proj_all = {}
#         for ai in range(n_animals):
#             segment_3d = smoothed[start_frame : start_frame + num_frames, :, ai]
#             xy = project_to_2d(
#                 segment_3d,
#                 cams[cam]['K'],
#                 cams[cam]['r'],
#                 cams[cam]['t']
#             )[:, :2]
#             xy = distortPoints(
#                 xy,
#                 cams[cam]['K'],
#                 np.squeeze(cams[cam]['RDistort']),
#                 np.squeeze(cams[cam]['TDistort'])
#             )
#             proj_all[ai] = xy.reshape(num_frames, -1, 2)

#         # 6. Write one MP4 per animal
#         for ai in range(n_animals):
#             out_fname = f"vis_{graph_title}_win{w:02d}_animal{ai+1}.mp4"
#             out_path  = os.path.join(save_folder, out_fname)
#             writer    = FFMpegWriter(fps=fps, metadata=meta)

#             fig = plt.figure(figsize=(6, 6))
#             with writer.saving(fig, out_path, dpi=200):
#                 for fi in range(num_frames):
#                     plt.clf()
#                     frame_idx = start_frame + fi
#                     frame_img = reader.get_data(frame_idx)
#                     kpts2d    = proj_all[ai][fi]  # shape (n_points, 2)
#                     plt.imshow(frame_img)
#                     plt.scatter(
#                         kpts2d[:, 0], kpts2d[:, 1],
#                         s=15, marker='.', alpha=0.6
#                     )
#                     plt.axis('off')
#                     writer.grab_frame()
#             plt.close(fig)


# def run_social_com_smooths(
#     base_folder,
#     cam='Camera1',
#     start_frame=0,
#     num_frames=60,
#     window_sizes=(1, 3, 5, 9, 15, 31, 61),
#     com_folder_name='COM/predict00'
# ):
#     """
#     Wrapper that, given only base_folder, loads multi-animal COM data and produces
#     overlay videos for multiple median-filter window sizes.

#     Parameters
#     ----------
#     base_folder : str
#         Path to the session folder containing:
#           - "{base_folder}/{com_folder_name}/com3d0.mat"
#           - "{base_folder}/videos/{cam}/0.mp4"
#     cam : str, default 'Camera1'
#         Subfolder under 'videos/' in which "0.mp4" is stored.
#     start_frame : int, default 0
#         Index of the first frame to visualize.
#     num_frames : int, default 60
#         Number of frames (starting from start_frame) to overlay in each output video.
#     window_sizes : iterable of int, default (1,3,5,9,15,31,61)
#         List of median-filter window sizes (in frames) to test. Window size = 1 means no filtering.
#     com_folder_name : str, default 'COM/predict00'
#         Relative path under base_folder where "com3d0.mat" lives.

#     Side Effects
#     ------------
#     - Creates (if needed) a "vis" subfolder under "{base_folder}/{com_folder_name}".
#     - Writes one MP4 per animal, per window_size, named:
#         vis_COM_<session>_win<ws>_animal<idx>.mp4
#     """
#     # 1. Determine session tag and vis folder
#     session_name = os.path.basename(os.path.normpath(base_folder))
#     graph_title  = f"COM_{session_name}"
#     com_dir      = os.path.join(base_folder, com_folder_name)
#     com_mat      = os.path.join(com_dir, "com3d0.mat")
#     if not os.path.exists(com_mat):
#         raise FileNotFoundError(f"No COM file found at {com_mat}")

#     vis_folder = os.path.join(com_dir, "vis")
#     os.makedirs(vis_folder, exist_ok=True)

#     # 2. Load raw COM and reshape to (frames, 3, n_animals)
#     raw = load_com(com_mat)
#     if raw.ndim == 3 and raw.shape[1] == 3:
#         com_data = raw.copy()
#     elif raw.ndim == 2 and raw.shape[1] % 3 == 0:
#         n_animals = raw.shape[1] // 3
#         com_data  = raw.reshape(-1, 3, n_animals)
#     else:
#         raise ValueError(f"Unexpected COM shape {raw.shape}")

#     # 3. Delegate to the smoothing video generator
#     generate_social_com_video_smooth(
#         com_data=com_data,
#         base_folder=base_folder,
#         graph_title=graph_title,
#         save_folder=vis_folder,
#         cam=cam,
#         start_frame=start_frame,
#         num_frames=num_frames,
#         window_sizes=window_sizes
#     )

import pandas as pd

def compare_com_distances(
    base_folder,
    com_folder_name='COM/predict00',
    vis_dir='vis',
    frame_count=300,
    save_csv=True
):
    """
    载入之前保存的平滑后 COM (.npy)，计算前 frame_count 帧中 
    动物 1 与动物 2 的欧式距离，绘制距离随帧数变化的曲线，并输出以下定量指标：
      - mean_dist
      - median_dist
      - max_dist
      - min_dist
      - std_dist

    最后会将这些指标保存到 CSV（默认保存在 vis 文件夹下 com_distance_summary.csv），
    并在控制台打印 DataFrame。

    参数
    ----
    base_folder : str
        最外层会话文件夹路径，包含 "{com_folder_name}/vis/" 子目录。
    com_folder_name : str, default 'COM/predict00'
        COM 文件所在的相对路径（例如 "COM/predict00"）。
    vis_dir : str, default 'vis'
        在 "{base_folder}/{com_folder_name}" 下，保存 .npy 和输出的子文件夹名。
    frame_count : int, default 300
        只计算并可视化前 frame_count 帧的距离。
    save_csv : bool, default True
        是否将定量指标保存为 CSV（文件名：com_distance_summary.csv）。

    要求
    ----
    - "{base_folder}/{com_folder_name}/vis/" 中已经存在多个名为 
      "com_filtered_winXX.npy"（XX 为两位窗口大小）的文件。
    - 每个 .npy 文件的数组形状为 (n_frames, 3, n_animals)，且至少有两只动物。

    返回
    ----
    pd.DataFrame
        包含每个窗口大小对应的（mean, median, max, min, std）距离指标。
    """
    # 构造 vis 文件夹路径
    vis_path = os.path.join(base_folder, com_folder_name, vis_dir)
    if not os.path.isdir(vis_path):
        raise FileNotFoundError(f"找不到 vis 文件夹：{vis_path}")

    # 找到所有以 com_filtered_win 开头、.npy 结尾的文件
    files = [
        fn for fn in os.listdir(vis_path)
        if fn.startswith("com_filtered_win") and fn.endswith(".npy")
    ]
    if not files:
        raise FileNotFoundError(f"在 {vis_path} 中未找到任何 'com_filtered_win*.npy' 文件")

    # 按窗口大小排序
    def extract_window_size(fname):
        # 例如 "com_filtered_win05.npy" → 返回整数 5
        name = os.path.splitext(fname)[0]  # "com_filtered_win05"
        suffix = name.replace("com_filtered_win", "")  # "05"
        return int(suffix)

    files_sorted = sorted(files, key=extract_window_size)

    # 存储每个窗口大小的定量指标
    summary_list = []

    plt.figure(figsize=(10, 6))
    plotted_any = False

    for fn in files_sorted:
        w = extract_window_size(fn)
        path_npy = os.path.join(vis_path, fn)
        com_arr = np.load(path_npy)  # 形状 (n_frames, 3, n_animals)

        if com_arr.ndim != 3 or com_arr.shape[2] < 2:
            # 至少需要两只动物才能计算彼此距离
            continue

        n_frames = com_arr.shape[0]
        n = min(frame_count, n_frames)
        # 提取前 n 帧，动物 0 和 1 的 (x,y,z)
        a0 = com_arr[:n, :, 0]  # (n, 3)
        a1 = com_arr[:n, :, 1]  # (n, 3)
        # 计算每帧的欧氏距离
        dists = np.linalg.norm(a0 - a1, axis=1)  # 形状 (n,)

        # 计算定量指标
        mean_dist   = np.nanmean(dists)
        median_dist = np.nanmedian(dists)
        max_dist    = np.nanmax(dists)
        min_dist    = np.nanmin(dists)
        std_dist    = np.nanstd(dists)

        summary_list.append({
            'window_size': w,
            'mean_dist':   mean_dist,
            'median_dist': median_dist,
            'max_dist':    max_dist,
            'min_dist':    min_dist,
            'std_dist':    std_dist
        })

        # 绘制距离曲线
        frames = np.arange(n)
        plt.plot(frames, dists, label=f"win{w}")
        plotted_any = True

    if not plotted_any:
        raise RuntimeError("没有可用的 COM 数据或没有足够数量的动物进行距离计算。")

    plt.xlabel("Frame Index")
    plt.ylabel("Distance Between Animal 1 and 2")
    plt.title(f"COM Distance Over First {frame_count} Frames")
    plt.legend(title="Window Size")
    plt.tight_layout()
    plt.show()

    # 转为 DataFrame 并打印
    df_summary = pd.DataFrame(summary_list).sort_values('window_size')
    print("\nQuantitative distance summary:\n")
    print(df_summary.to_string(index=False))

    # 保存为 CSV（可选）
    if save_csv:
        csv_path = os.path.join(vis_path, "com_distance_summary.csv")
        df_summary.to_csv(csv_path, index=False)
        print(f"\n已将定量指标保存到：{csv_path}")

    return df_summary


def plot_raw_com_distance(
    base_folder,
    com_folder_name='COM/predict00',
    save_csv=True
):
    """
    从原始 COM mat 文件中载入多动物数据，计算动物 1 和动物 2 
    在整个会话每一帧的欧式距离，并绘制距离随帧索引的曲线。

    同时计算并输出全程（所有帧）的定量指标：
      - mean_dist
      - median_dist
      - max_dist
      - min_dist
      - std_dist

    可选地将这些指标保存为 CSV（保存路径： 
    {base_folder}/{com_folder_name}/vis/com_raw_distance_summary.csv）。

    参数
    ----
    base_folder : str
        会话最外层文件夹路径，例如 "/path/to/session/"。
    com_folder_name : str, default 'COM/predict00'
        相对于 base_folder 的 COM mat 文件所在子文件夹。
        最终会尝试加载：
          "{base_folder}/{com_folder_name}/com3d0.mat"
        如果不存在，会抛出 FileNotFoundError。
    save_csv : bool, default True
        是否将定量指标保存到 CSV 文件中。

    返回
    ----
    pd.DataFrame
        包含单个“raw”窗口、针对全帧的距离指标：
          window_type='raw',
          mean_dist, median_dist, max_dist, min_dist, std_dist, n_frames
    """
    # 1. 构造 com mat 文件路径
    com_dir = os.path.join(base_folder, com_folder_name)
    com_mat = os.path.join(com_dir, 'com3d0.mat')
    if not os.path.isfile(com_mat):
        raise FileNotFoundError(f"未找到 COM 文件: {com_mat}")

    # 2. 载入 mat 并提取 'com' 变量
    data = sio.loadmat(com_mat)
    if 'com' not in data:
        raise KeyError(f"'com' 键在文件 {com_mat} 中未找到")
    raw = data['com']  # 可能是 (n_frames, 3, n_animals) 或 (n_frames, 3*n_animals)

    # 3. 如果是二维扁平形式，需要重塑
    if raw.ndim == 3 and raw.shape[1] == 3:
        com_data = raw  # 已经是 (n_frames, 3, n_animals)
    elif raw.ndim == 2 and raw.shape[1] % 3 == 0:
        n_animals = raw.shape[1] // 3
        com_data = raw.reshape(-1, 3, n_animals)
    else:
        raise ValueError(f"Unexpected COM shape {raw.shape}; 期望 (frames,3,n_animals) 或 (frames,3*n_animals)")

    n_frames, _, n_animals = com_data.shape
    if n_animals < 2:
        raise ValueError("至少需要两只动物才能计算彼此距离")

    # 4. 提取动物 0 和动物 1 的 (x,y,z)
    a0 = com_data[:, :, 0]  # 形状 (n_frames, 3)
    a1 = com_data[:, :, 1]  # 形状 (n_frames, 3)

    # 5. 计算每帧的欧式距离
    dists = np.linalg.norm(a0 - a1, axis=1)  # 形状 (n_frames,)

    # 6. 绘制距离随帧索引变化的曲线
    frames = np.arange(n_frames)
    plt.figure(figsize=(10, 4))
    plt.plot(frames, dists, color='tab:blue')
    plt.xlabel("Frame Index")
    plt.ylabel("Raw COM Distance (mm)")
    plt.title("Raw COM Distance Between Animal 1 and 2 Over Entire Session")
    plt.tight_layout()
    plt.show()

    # 7. 计算全程定量指标
    mean_dist   = np.nanmean(dists)
    median_dist = np.nanmedian(dists)
    max_dist    = np.nanmax(dists)
    min_dist    = np.nanmin(dists)
    std_dist    = np.nanstd(dists)

    summary = {
        'window_type': ['raw'],
        'mean_dist':   [mean_dist],
        'median_dist': [median_dist],
        'max_dist':    [max_dist],
        'min_dist':    [min_dist],
        'std_dist':    [std_dist],
        'n_frames':    [n_frames]
    }
    df_summary = pd.DataFrame(summary)

    print("\nRaw COM Distance Summary (entire session):\n")
    print(df_summary.to_string(index=False))

    # 8. 保存 CSV（若需要）
    if save_csv:
        vis_path = os.path.join(com_dir, 'vis')
        os.makedirs(vis_path, exist_ok=True)
        csv_path = os.path.join(vis_path, 'com_raw_distance_summary.csv')
        df_summary.to_csv(csv_path, index=False)
        print(f"\n已将定量指标保存到：{csv_path}")

    return df_summary


import os
import numpy as np
import scipy.io as sio
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares
from scipy.signal import medfilt
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# Assumes these are already imported somewhere:
#   find_calib_file, load_cameras, project_to_2d, distortPoints
#   detect_social_jumps

def _fit_circle_xy(x, y):
    """
    Fit a circle (xc, yc, r) to points (x, y) using convex hull + least squares.
    Returns xc, yc, r.
    """
    pts = np.stack([x, y], axis=1)
    hull = ConvexHull(pts)
    hx, hy = pts[hull.vertices, 0], pts[hull.vertices, 1]
    init = [
        hx.mean(),
        hy.mean(),
        np.mean(np.hypot(hx - hx.mean(), hy - hy.mean()))
    ]
    def resid(p, xh, yh):
        return np.hypot(xh - p[0], yh - p[1]) - p[2]
    res = least_squares(resid, init, args=(hx, hy))
    return res.x  # (xc, yc, r)

def _filter_and_interpolate(
    smoothed,  # shape (n_frames, 3, n_animals)
    raw_xy,    # shape (n_frames, 2, n_animals), raw COM x,y per animal
    jumps_dict # dict: ai -> array of jump indices
):
    """
    For each animal:
      1) Fit circle to raw_xy[:, :, ai], get xc,yc,r.
      2) Mask out any smoothed frames where (x,y) is outside circle OR frame in jumps_dict[ai].
      3) Interpolate masked frames by averaging nearest valid neighbors.
    Returns a copy of smoothed with masked values replaced by interpolated values.
    """
    n_frames, _, n_animals = smoothed.shape
    filtered = smoothed.copy().astype(float)

    for ai in range(n_animals):
        x_raw = raw_xy[:, 0, ai]
        y_raw = raw_xy[:, 1, ai]
        xc, yc, r = _fit_circle_xy(x_raw, y_raw)

        # 1) Determine masks
        dist = np.hypot(filtered[:, 0, ai] - xc, filtered[:, 1, ai] - yc)
        outside_circle = dist > r
        jump_mask = np.zeros(n_frames, dtype=bool)
        if ai in jumps_dict:
            jump_mask[jumps_dict[ai]] = True

        mask = outside_circle | jump_mask  # shape (n_frames,)

        # 2) Set masked frames to NaN for all 3 coords
        for coord in range(3):
            arr = filtered[:, coord, ai]
            arr[mask] = np.nan
            filtered[:, coord, ai] = arr

        # 3) Interpolate each coordinate separately
        for coord in range(3):
            arr = filtered[:, coord, ai]
            valid = ~np.isnan(arr)
            valid_idxs = np.where(valid)[0]

            if valid_idxs.size == 0:
                continue  # no valid data to interpolate from

            # For each index i where arr[i] is nan, find neighbors
            nan_idxs = np.where(np.isnan(arr))[0]
            for i in nan_idxs:
                # find previous valid
                prev = valid_idxs[valid_idxs < i]
                nxt  = valid_idxs[valid_idxs > i]
                if prev.size and nxt.size:
                    i_prev = prev[-1]
                    i_nxt  = nxt[0]
                    arr[i] = 0.5 * (arr[i_prev] + arr[i_nxt])
                elif prev.size:
                    arr[i] = arr[prev[-1]]
                elif nxt.size:
                    arr[i] = arr[nxt[0]]
                # else: leave as NaN if no neighbors

            filtered[:, coord, ai] = arr

    return filtered

def generate_social_com_video_smooth(
    com_data,
    base_folder,
    graph_title,
    save_folder,
    cam='Camera1',
    start_frame=0,
    num_frames=60,
    window_sizes=(1, 5, 15),
):
    """
    For each median-filter window size in window_sizes, produce:
      1) A filtered-and-interpolated COM array saved as .npy.
      2) An MP4 per animal where the first num_frames (from start_frame)
         of that filtered COM are overlaid on the video.

    Assumes:
      - find_calib_file, load_cameras, project_to_2d, distortPoints are in scope.
      - scipy.signal.medfilt is available.
      - detect_social_jumps(com_data, save_folder) returns dict {ai: jump_idxs}.
    """
    # 1. Load calibration & camera parameters
    calib = find_calib_file(base_folder)
    cams  = load_cameras(calib)

    # 2. Prepare video reader
    vid_path = os.path.join(base_folder, f'videos/{cam}/0.mp4')
    if not os.path.exists(vid_path):
        raise FileNotFoundError(f"No video at {vid_path}")
    reader = imageio.get_reader(vid_path)
    fps    = reader.get_meta_data().get('fps', 30)

    # 3. Ensure save_folder exists
    os.makedirs(save_folder, exist_ok=True)

    n_frames, _, n_animals = com_data.shape
    meta      = dict(title='dannce_visualization', artist='Matplotlib')

    # 4. Precompute raw 2D XY for circle fitting
    #    raw_xy[frame, coord(0=x,1=y), ai]
    raw_xy = np.zeros((n_frames, 2, n_animals))
    for ai in range(n_animals):
        raw_xy[:, 0, ai] = com_data[:, 0, ai]
        raw_xy[:, 1, ai] = com_data[:, 1, ai]

    # 5. Detect jumps once (uses raw COM in 3D)
    jumps_dict = detect_social_jumps(com_data, save_folder)

    # 6. Loop over smoothing windows
    for w in window_sizes:
        # 6a. Apply median filter if w > 1
        if w > 1:
            smoothed = np.empty_like(com_data)
            for ai in range(n_animals):
                smoothed[:, :, ai] = medfilt(
                    com_data[:, :, ai], kernel_size=(w, 1)
                )
        else:
            smoothed = com_data.astype(float).copy()

        # 6b. Filter out points outside circle & jumps, then interpolate
        filtered = _filter_and_interpolate(smoothed, raw_xy, jumps_dict)

        # 6c. Save filtered array
        npy_out = os.path.join(save_folder, f"com_filtered_win{w:02d}.npy")
        np.save(npy_out, filtered)
        print(f"Saved filtered COM (w={w}) to:\n  {npy_out}")

        # 6d. Project filtered COM and write videos
        for ai in range(n_animals):
            # project selected frames to 2D
            seg_3d = filtered[start_frame : start_frame + num_frames, :, ai]
            xy = project_to_2d(
                seg_3d,
                cams[cam]['K'],
                cams[cam]['r'],
                cams[cam]['t']
            )[:, :2]
            xy = distortPoints(
                xy,
                cams[cam]['K'],
                np.squeeze(cams[cam]['RDistort']),
                np.squeeze(cams[cam]['TDistort'])
            )
            # 关键：先做 .T，再 reshape
            proj2d = xy.T.reshape(num_frames, -1, 2)

            # prepare writer
            out_fname = f"vis_{graph_title}_win{w:02d}_animal{ai+1}.mp4"
            out_path  = os.path.join(save_folder, out_fname)
            writer    = FFMpegWriter(fps=fps, metadata=meta)

            fig = plt.figure(figsize=(6, 6))
            with writer.saving(fig, out_path, dpi=200):
                for fi in range(num_frames):
                    plt.clf()
                    frame_idx = start_frame + fi
                    frame_img = reader.get_data(frame_idx)
                    pts2d     = proj2d[fi]  # shape: (1,2)  或 (n_points,2) 如果有多点
                    plt.imshow(frame_img)
                    plt.scatter(
                        pts2d[:, 0], pts2d[:, 1],
                        s=15, marker='.', alpha=0.6
                    )
                    plt.axis('off')
                    writer.grab_frame()
            plt.close(fig)


def run_social_com_smooths(
    base_folder,
    cam='Camera1',
    start_frame=0,
    num_frames=60,
    window_sizes=(3, 5, 9, 15, 31, 61),
    com_folder_name='COM/predict00'
):
    """
    Wrapper that:
      1) Loads multi-animal COM from base_folder/{com_folder_name}/com3d0.mat.
      2) Calls generate_social_com_video_smooth() to produce:
         - Filtered COM .npy files per window size.
         - Videos per animal per window size.

    Parameters
    ----------
    base_folder : str
        Path to the session folder containing:
          - "{base_folder}/{com_folder_name}/com3d0.mat"
          - "{base_folder}/videos/{cam}/0.mp4"
    cam : str, default 'Camera1'
        Subfolder under 'videos/' where "0.mp4" is stored.
    start_frame : int, default 0
        Index of the first frame to visualize.
    num_frames : int, default 60
        Number of frames (starting from start_frame) to overlay.
    window_sizes : iterable of int, default (3,5,9,15,31,61)
        List of median-filter window sizes (in frames) to test.
    com_folder_name : str, default 'COM/predict00'
        Relative path under base_folder where "com3d0.mat" lives.
    """
    # 1. Paths and checks
    session_name = os.path.basename(os.path.normpath(base_folder))
    graph_title  = f"COM_{session_name}"
    com_dir      = os.path.join(base_folder, com_folder_name)
    com_mat      = os.path.join(com_dir, "com3d0.mat")
    if not os.path.exists(com_mat):
        raise FileNotFoundError(f"No COM file found at {com_mat}")

    vis_folder = os.path.join(com_dir, "vis")
    os.makedirs(vis_folder, exist_ok=True)

    # 2. Load raw COM and reshape to (frames, 3, n_animals)
    raw = sio.loadmat(com_mat).get('com')
    if raw is None:
        raise KeyError(f"'com' variable not found in {com_mat}")
    if raw.ndim == 3 and raw.shape[1] == 3:
        com_data = raw.copy()
    elif raw.ndim == 2 and raw.shape[1] % 3 == 0:
        n_animals = raw.shape[1] // 3
        com_data  = raw.reshape(-1, 3, n_animals)
    else:
        raise ValueError(f"Unexpected COM shape {raw.shape}")

    # 3. Delegate to smoothing + visualization
    generate_social_com_video_smooth(
        com_data=com_data,
        base_folder=base_folder,
        graph_title=graph_title,
        save_folder=vis_folder,
        cam=cam,
        start_frame=start_frame,
        num_frames=num_frames,
        window_sizes=window_sizes
    )
