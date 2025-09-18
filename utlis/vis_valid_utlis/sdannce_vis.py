# -*- coding: utf-8 -*-
"""
Minimal refactor:
- No module-level mutable globals.
- User-editable defaults via VizConfig (frozen dataclass).
- visualize_clip(...) and visualize_frames(...) accept either a VizConfig
  or dynamic overrides via **kwargs (e.g., base_path="...", cammm=2, ...).
- Everything else (projection, skeleton, COM handling) stays the same.
"""

import os, sys
sys.path.append(os.path.abspath('../..'))
from dataclasses import dataclass, replace
from typing import Optional, List, Dict, Any

import numpy as np
import scipy.io as sio
import imageio
import tqdm

from utlis import connectivity
from utlis.projection import *      # load_cameras, project_to_2d, distortPoints

import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


############################  USER-EDITABLE DEFAULTS (now a dataclass)  ############################
@dataclass(frozen=True)
class VizConfig:
    base_path: str = "/data/big_rim/rsync_dcc_sum/Oct3V1/2024_10_31/2social_mini_20240819V1r1_femalebleach_11_48"
    cammm: int = 3
    pred_folder: str = "SDANNCE/predict00"
    pred_filename: str = "save_data_AVG.mat"   # e.g. save_data_AVG.mat or smoothed_prediction_AVG0.mat
    com_filename: str = "com3d_used.mat"       # usually alongside predictions

    start_frame: int = 1000                    # only used by visualize_clip(...)
    n_frames: int = 100                        # only used by visualize_clip(...)
    animal: str = "mouse20"                    # lookup in connectivity dicts

    video_fps: int = 20                        # your original used 20
    save_subdir: str = "vis"                   # under pred_folder

    # Zoom options (affects both functions)
    enable_zoom: bool = False                  # False -> no zoom
    zoom_margin: int = 450                     # pixels around mean of visible points
    zoom_include_both: bool = True             # center using both animals if 2 present

    # Tail handling in viewport centering (your original "drop tail mid/end")
    drop_tail_for_view: bool = True

    # Optional naming/extras
    out_name: Optional[str] = None
    write_pngs: bool = False                   # only used by visualize_frames(...)

# Default instance (read-only)
DEFAULT_CFG = VizConfig()

# Helper to create a tweaked copy quickly
def cfg(**overrides: Any) -> VizConfig:
    """Return a VizConfig derived from DEFAULT_CFG with fields overridden."""
    return replace(DEFAULT_CFG, **overrides)
###################################################################################################


def find_calib_file(base_folder: str) -> Optional[str]:
    for file_name in os.listdir(base_folder):
        if file_name.endswith('label3d_dannce.mat'):
            return os.path.join(base_folder, file_name)
    return None


def _trim_tail(xy: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Drop tail(mid/end) like your original: keep 0-5 and 8-21."""
    if xy is None:
        return None
    return np.r_[xy[0:6, :], xy[8:, :]]


def _adjust_viewport(kpts_2d_a1: np.ndarray,
                     kpts_2d_a2: Optional[np.ndarray] = None,
                     margin: int = 70) -> None:
    """
    Center around the mean of visible points from one or both animals.
    kpts_2d_a*: (22,2) arrays (NaN allowed).
    """
    stack = kpts_2d_a1 if kpts_2d_a2 is None else np.vstack([kpts_2d_a1, kpts_2d_a2])
    valid = ~np.isnan(stack).any(axis=1)
    pts = stack[valid] if valid.any() else stack
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    plt.xlim([cx - margin, cx + margin])
    plt.ylim([cy + margin, cy - margin])  # image origin top-left


def _apply_overrides(base: VizConfig,
                     overrides: Dict[str, Any]) -> VizConfig:
    """Type-safe override helper with unknown-key detection."""
    if not overrides:
        return base
    allowed = set(VizConfig.__annotations__.keys())
    unknown = set(overrides) - allowed
    if unknown:
        raise TypeError(f"Unknown options: {sorted(unknown)}")
    return replace(base, **overrides)


def _prepare_io(c: VizConfig):
    cam = f"Camera{c.cammm}"
    video_path = os.path.join(c.base_path, f"videos/{cam}/0.mp4")
    label3d_path = find_calib_file(c.base_path)
    pred_path = os.path.join(c.base_path, c.pred_folder, c.pred_filename)
    com_path  = os.path.join(c.base_path, c.pred_folder, c.com_filename)
    save_path = os.path.join(c.base_path, c.pred_folder, c.save_subdir)
    os.makedirs(save_path, exist_ok=True)
    return cam, video_path, label3d_path, pred_path, com_path, save_path


def _load_and_project_preds(pred_path: str,
                            cameras: dict,
                            cam: str,
                            start_frame: int,
                            n_frames: int):
    """
    Returns: pred_2d[cam] as (F, 22, 2) or (F, 2, 22, 2) depending on #animals.
    """
    pred_raw = sio.loadmat(pred_path)['pred'][start_frame: start_frame + n_frames]
    pred_raw = np.squeeze(pred_raw)  # (F,3,22) or (F,2,3,22)

    pred_2d = {}
    camK, camr, camt = cameras[cam]["K"], cameras[cam]["r"], cameras[cam]["t"]
    Rdist, Tdist = np.squeeze(cameras[cam]["RDistort"]), np.squeeze(cameras[cam]["TDistort"])

    if pred_raw.ndim == 3:  # (F, 3, 22) -> single animal
        pose_3d = np.transpose(pred_raw, (0, 2, 1))      # (F,22,3)
        pts = pose_3d.reshape(-1, 3)                      # (F*22,3)
        projpts = project_to_2d(pts, camK, camr, camt)[:, :2]
        projpts = distortPoints(projpts, camK, Rdist, Tdist).T
        projpts = projpts.reshape(n_frames, 22, 2)        # (F,22,2)
        pred_2d[cam] = projpts
    else:  # (F, 2, 3, 22) -> two animals
        pose_3d = np.transpose(pred_raw, (0, 1, 3, 2))    # (F,2,22,3)
        pts = pose_3d.reshape(-1, 3)                      # (F*2*22,3)
        projpts = project_to_2d(pts, camK, camr, camt)[:, :2]
        projpts = distortPoints(projpts, camK, Rdist, Tdist).T
        projpts = projpts.reshape(n_frames, 2, 22, 2)     # (F,2,22,2)
        pred_2d[cam] = projpts

    return pred_2d


def _load_and_project_com(com_path: str,
                          cameras: dict,
                          cam: str,
                          start_frame: int,
                          n_frames: int):
    """
    Robust to COM shapes: (F,3,2) | (F,2,3) | (F,3) for single animal.
    Returns pred_2d_com[cam] of shape (F, A, 2), where A ∈ {1,2}.
    """
    com_data = sio.loadmat(com_path)
    pts_com_raw = np.asarray(com_data['com'][start_frame: start_frame + n_frames])  # e.g. (F,3,2)

    camK, camr, camt = cameras[cam]["K"], cameras[cam]["r"], cameras[cam]["t"]
    Rdist, Tdist = np.squeeze(cameras[cam]["RDistort"]), np.squeeze(cameras[cam]["TDistort"])

    if pts_com_raw.ndim == 3:
        if pts_com_raw.shape[1] == 3 and pts_com_raw.shape[2] == 2:      # (F,3,2) -> (F,2,3)
            pts_F_A_C = np.transpose(pts_com_raw, (0, 2, 1))
        elif pts_com_raw.shape[1] == 2 and pts_com_raw.shape[2] == 3:    # (F,2,3)
            pts_F_A_C = pts_com_raw
        else:
            raise ValueError(f"Unexpected COM shape {pts_com_raw.shape}; expected (F,3,2) or (F,2,3).")
    elif pts_com_raw.ndim == 2 and pts_com_raw.shape[1] == 3:            # (F,3) -> (F,1,3)
        pts_F_A_C = pts_com_raw[:, None, :]
    else:
        raise ValueError(f"Unexpected COM shape {pts_com_raw.shape}.")

    n_animals = pts_F_A_C.shape[1]
    proj_com = project_to_2d(pts_F_A_C.reshape(-1, 3), camK, camr, camt)[:, :2]
    proj_com = distortPoints(proj_com, camK, Rdist, Tdist).T
    proj_com = proj_com.reshape(n_frames, n_animals, 2)   # (F,A,2)
    return {cam: proj_com}


def _draw_frame(img: np.ndarray,
                k_a1: np.ndarray,
                k_a2: Optional[np.ndarray],
                k_com: np.ndarray,
                COLOR,
                CONNECTIVITY,
                enable_zoom: bool,
                zoom_margin: int,
                drop_tail_for_view: bool,
                include_both: bool,
                title_text: Optional[str]) -> None:
    plt.clf()
    plt.imshow(img)

    # viewport (optional)
    if enable_zoom:
        a1 = _trim_tail(k_a1) if drop_tail_for_view else k_a1
        a2 = None
        if include_both and k_a2 is not None:
            a2 = _trim_tail(k_a2) if drop_tail_for_view else k_a2
        _adjust_viewport(a1, a2, margin=zoom_margin)

    # COM points
    if k_com.ndim == 2:  # (A,2)
        plt.scatter(k_com[:, 0], k_com[:, 1], marker='.', color='red', linewidths=2, alpha=0.5)

    # keypoints
    plt.scatter(k_a1[:, 0], k_a1[:, 1], marker='.', color='white', linewidths=2, alpha=0.8)
    if k_a2 is not None:
        plt.scatter(k_a2[:, 0], k_a2[:, 1], marker='.', color='cyan', linewidths=2, alpha=0.8)

    # skeleton lines
    for color, (index_from, index_to) in zip(COLOR, CONNECTIVITY):
        xs, ys = [np.array([k_a1[index_from, j], k_a1[index_to, j]]) for j in range(2)]
        plt.plot(xs, ys, c=color, lw=2)
        if k_a2 is not None:
            xs2, ys2 = [np.array([k_a2[index_from, j], k_a2[index_to, j]]) for j in range(2)]
            plt.plot(xs2, ys2, c=color, lw=2, alpha=0.9)

    if title_text:
        plt.title(title_text)
    plt.axis("off")


def visualize_clip(config: Optional[VizConfig] = None, **overrides: Any) -> None:
    """
    Continuous clip renderer (same output behavior as your original).
    You can:
      - pass a VizConfig, or
      - just override fields dynamically: visualize_clip(base_path="...", start_frame=1200, enable_zoom=True)
    """
    c = _apply_overrides(config or DEFAULT_CFG, overrides)

    cam, video_path, label3d_path, pred_path, com_path, save_path = _prepare_io(c)
    COLOR = connectivity.COLOR_DICT[c.animal]
    CONNECTIVITY = connectivity.CONNECTIVITY_DICT[c.animal]
    cameras = load_cameras(label3d_path)

    pred_2d = _load_and_project_preds(pred_path, cameras, cam, c.start_frame, c.n_frames)
    pred_2d_com = _load_and_project_com(com_path, cameras, cam, c.start_frame, c.n_frames)

    vids = imageio.get_reader(video_path)
    fig = plt.figure()
    plt.rcParams['figure.figsize'] = (6, 6)

    vid_title = f"combined_cam{c.cammm}_{c.n_frames}_after{c.start_frame}"
    vid_name = (c.out_name if c.out_name else vid_title) + ".mp4"
    writer = FFMpegWriter(fps=c.video_fps, metadata=dict(title='combined_visualization', artist='Matplotlib'))

    with writer.saving(fig, os.path.join(save_path, "vis_" + vid_name), dpi=300):
        for i in tqdm.tqdm(range(c.n_frames)):
            f_abs = c.start_frame + i
            img = vids.get_data(f_abs)

            k = pred_2d[cam][i]  # (22,2) or (2,22,2)
            if k.ndim == 2:
                k_a1, k_a2 = k, None
            else:
                k_a1, k_a2 = k[0], k[1]

            k_com = pred_2d_com[cam][i]  # (A,2)

            _draw_frame(
                img, k_a1, k_a2, k_com,
                COLOR, CONNECTIVITY,
                enable_zoom=c.enable_zoom, zoom_margin=c.zoom_margin,
                drop_tail_for_view=c.drop_tail_for_view, include_both=c.zoom_include_both,
                title_text=vid_title
            )
            writer.grab_frame()


def visualize_frames(frame_list: List[int],
                     config: Optional[VizConfig] = None,
                     **overrides: Any) -> None:
    """
    Render only the specified absolute frames (e.g., an 'incident' list).
    Saves an MP4 (and optionally PNGs) as controlled by config/write_pngs.
    """
    if not frame_list:
        raise ValueError("frame_list is empty.")
    frame_list = list(sorted(frame_list))

    c = _apply_overrides(config or DEFAULT_CFG, overrides)

    cam, video_path, label3d_path, pred_path, com_path, save_path = _prepare_io(c)
    COLOR = connectivity.COLOR_DICT[c.animal]
    CONNECTIVITY = connectivity.CONNECTIVITY_DICT[c.animal]
    cameras = load_cameras(label3d_path)

    fmin, fmax = frame_list[0], frame_list[-1]
    n_frames = fmax - fmin + 1

    pred_2d = _load_and_project_preds(pred_path, cameras, cam, fmin, n_frames)
    pred_2d_com = _load_and_project_com(com_path, cameras, cam, fmin, n_frames)

    vids = imageio.get_reader(video_path)
    fig = plt.figure()
    plt.rcParams['figure.figsize'] = (6, 6)

    tag = f"custom_frames_{frame_list[0]}_{frame_list[-1]}_{len(frame_list)}f"
    vid_name = (c.out_name if c.out_name else f"combined_cam{c.cammm}_{tag}") + ".mp4"
    writer = FFMpegWriter(fps=c.video_fps, metadata=dict(title='combined_visualization', artist='Matplotlib'))

    with writer.saving(fig, os.path.join(save_path, "vis_" + vid_name), dpi=300):
        for f_abs in tqdm.tqdm(frame_list):
            i = f_abs - fmin
            img = vids.get_data(f_abs)

            k = pred_2d[cam][i]  # (22,2) or (2,22,2)
            if k.ndim == 2:
                k_a1, k_a2 = k, None
            else:
                k_a1, k_a2 = k[0], k[1]

            k_com = pred_2d_com[cam][i]  # (A,2)

            _draw_frame(
                img, k_a1, k_a2, k_com,
                COLOR, CONNECTIVITY,
                enable_zoom=c.enable_zoom, zoom_margin=c.zoom_margin,
                drop_tail_for_view=c.drop_tail_for_view, include_both=c.zoom_include_both,
                title_text=f"cam{c.cammm} frame {f_abs}"
            )
            writer.grab_frame()

            if c.write_pngs:
                png_name = os.path.join(save_path, f"vis_frame_{f_abs}.png")
                plt.savefig(png_name, dpi=300, bbox_inches="tight")
