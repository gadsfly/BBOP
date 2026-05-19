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
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
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
    pred_folder: str = "SDANNCE/predict01"
    pred_filename: str = "save_data_AVG.mat"   # e.g. save_data_AVG.mat or smoothed_prediction_AVG0.mat
    com_filename: str = "com3d_used.mat"       # usually alongside predictions

    start_frame: int = 1000                    # only used by visualize_clip(...)
    n_frames: int = 100                        # only used by visualize_clip(...)
    animal: str = "mouse20"                    # lookup in connectivity dicts

    video_fps: int = 30                       # your original used 20
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


# below works. but it is max-min frames and plot, not plotting the frames from list after alignment...
# def visualize_frames(frame_list: List[int],
#                      config: Optional[VizConfig] = None,
#                      **overrides: Any) -> None:
#     """
#     Render only the specified absolute frames (e.g., an 'incident' list).
#     Saves an MP4 (and optionally PNGs) as controlled by config/write_pngs.
#     """
#     if not frame_list:
#         raise ValueError("frame_list is empty.")
#     frame_list = list(sorted(frame_list))

#     c = _apply_overrides(config or DEFAULT_CFG, overrides)

#     cam, video_path, label3d_path, pred_path, com_path, save_path = _prepare_io(c)
#     COLOR = connectivity.COLOR_DICT[c.animal]
#     CONNECTIVITY = connectivity.CONNECTIVITY_DICT[c.animal]
#     cameras = load_cameras(label3d_path)

#     fmin, fmax = frame_list[0], frame_list[-1]
#     n_frames = fmax - fmin + 1

#     pred_2d = _load_and_project_preds(pred_path, cameras, cam, fmin, n_frames)
#     pred_2d_com = _load_and_project_com(com_path, cameras, cam, fmin, n_frames)

#     vids = imageio.get_reader(video_path)
#     fig = plt.figure()
#     plt.rcParams['figure.figsize'] = (6, 6)

#     tag = f"custom_frames_{frame_list[0]}_{frame_list[-1]}_{len(frame_list)}f"
#     vid_name = (c.out_name if c.out_name else f"combined_cam{c.cammm}_{tag}") + ".mp4"
#     writer = FFMpegWriter(fps=c.video_fps, metadata=dict(title='combined_visualization', artist='Matplotlib'))

#     with writer.saving(fig, os.path.join(save_path, "vis_" + vid_name), dpi=300):
#         for f_abs in tqdm.tqdm(frame_list):
#             i = f_abs - fmin
#             img = vids.get_data(f_abs)

#             k = pred_2d[cam][i]  # (22,2) or (2,22,2)
#             if k.ndim == 2:
#                 k_a1, k_a2 = k, None
#             else:
#                 k_a1, k_a2 = k[0], k[1]

#             k_com = pred_2d_com[cam][i]  # (A,2)

#             _draw_frame(
#                 img, k_a1, k_a2, k_com,
#                 COLOR, CONNECTIVITY,
#                 enable_zoom=c.enable_zoom, zoom_margin=c.zoom_margin,
#                 drop_tail_for_view=c.drop_tail_for_view, include_both=c.zoom_include_both,
#                 title_text=f"cam{c.cammm} frame {f_abs}"
#             )
#             writer.grab_frame()

#             if c.write_pngs:
#                 png_name = os.path.join(save_path, f"vis_frame_{f_abs}.png")
#                 plt.savefig(png_name, dpi=300, bbox_inches="tight")




# -----------------------------
# 1) Sparse-index loader: preds
# -----------------------------
def _load_and_project_preds_frames(pred_path: str,
                                   cameras: Dict[str, dict],
                                   cam: str,
                                   frames: List[int]):
    """
    Returns: pred_2d[cam] as (F, 22, 2) or (F, 2, 22, 2) depending on #animals,
    where F = len(frames). Uses explicit frame indices (non-contiguous OK).
    """
    pred_all = sio.loadmat(pred_path)['pred']                  # (T, 3, 22) or (T, 2, 3, 22)
    T = pred_all.shape[0]

    # bounds check (fail fast with the first few offending indices)
    bad = [f for f in frames if f < 0 or f >= T]
    if bad:
        raise IndexError(f"pred: frame indices out of bounds (T={T}): {bad[:10]}...")

    # keep leading axis even when F == 1
    pred_raw = np.take(pred_all, frames, axis=0)               # shape starts with (F, ...)
    F = pred_raw.shape[0]

    pred_2d = {}
    camK, camr, camt = cameras[cam]["K"], cameras[cam]["r"], cameras[cam]["t"]
    Rdist = np.squeeze(cameras[cam]["RDistort"])
    Tdist = np.squeeze(cameras[cam]["TDistort"])

    # single vs two animals by ndim AFTER selection
    # single: (F, 3, 22), two: (F, 2, 3, 22)
    if pred_raw.ndim == 3:  # (F, 3, 22) -> single animal
        pose_3d = np.transpose(pred_raw, (0, 2, 1))           # (F,22,3)
        pts = pose_3d.reshape(-1, 3)                           # (F*22,3)
        projpts = project_to_2d(pts, camK, camr, camt)[:, :2]
        projpts = distortPoints(projpts, camK, Rdist, Tdist).T
        projpts = projpts.reshape(F, 22, 2)                    # (F,22,2)
        pred_2d[cam] = projpts
    else:  # assume (F, 2, 3, 22) -> two animals
        pose_3d = np.transpose(pred_raw, (0, 1, 3, 2))         # (F,2,22,3)
        pts = pose_3d.reshape(-1, 3)                           # (F*2*22,3)
        projpts = project_to_2d(pts, camK, camr, camt)[:, :2]
        projpts = distortPoints(projpts, camK, Rdist, Tdist).T
        projpts = projpts.reshape(F, 2, 22, 2)                 # (F,2,22,2)
        pred_2d[cam] = projpts

    return pred_2d


# ---------------------------
# 2) Sparse-index loader: COM
# ---------------------------
def _load_and_project_com_frames(com_path: str,
                                 cameras: Dict[str, dict],
                                 cam: str,
                                 frames: List[int]):
    """
    Robust to COM shapes: (T,3,2) | (T,2,3) | (T,3) for single animal.
    Returns {cam: (F, A, 2)} with A in {1,2}, F=len(frames).
    """
    com_all = sio.loadmat(com_path)['com']     # e.g., (T,3,2) or (T,2,3) or (T,3)
    T = com_all.shape[0]

    bad = [f for f in frames if f < 0 or f >= T]
    if bad:
        raise IndexError(f"com: frame indices out of bounds (T={T}): {bad[:10]}...")

    pts_com_raw = np.take(com_all, frames, axis=0)             # keep (F, ...)
    F = pts_com_raw.shape[0]

    # normalize to (F, A, 3)
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

    camK, camr, camt = cameras[cam]["K"], cameras[cam]["r"], cameras[cam]["t"]
    Rdist = np.squeeze(cameras[cam]["RDistort"])
    Tdist = np.squeeze(cameras[cam]["TDistort"])

    proj = project_to_2d(pts_F_A_C.reshape(-1, 3), camK, camr, camt)[:, :2]
    proj = distortPoints(proj, camK, Rdist, Tdist).T
    proj = proj.reshape(F, n_animals, 2)      # (F,A,2)

    return {cam: proj}

#############################################3333FPS helper functions #############################################
def infer_frame_list_and_fps(frames_df: pd.DataFrame,
                             default_fps: float = 30.0
                            ) -> Tuple[List[int], float]:
    """
    Picks the frame index column and computes a steady fps from timestamp_ms_mini.
    Returns (frame_list, fps).
    """
    # 1) pick the frame column (your rule)
    if "camera_frame_sixcam" in frames_df.columns:
        frame_col = "camera_frame_sixcam"
    elif "mapped_sixcam_frame_indices" in frames_df.columns:
        frame_col = "mapped_sixcam_frame_indices"
    else:
        raise KeyError("No frame index column found.")

    frame_list = frames_df[frame_col].astype(int).tolist()

    # 2) compute fps from timestamps (simple & robust)
    if "timestamp_ms_mini" in frames_df.columns:
        ts = frames_df["timestamp_ms_mini"].to_numpy(dtype=float)
        # keep finite & strictly increasing gaps
        dts = np.diff(ts)
        dts = dts[np.isfinite(dts) & (dts > 0)]
        if dts.size:
            median_dt_ms = float(np.median(dts))
            # guardrails
            if median_dt_ms > 0:
                fps = 1000.0 / median_dt_ms
            else:
                fps = default_fps
        else:
            fps = default_fps
    else:
        fps = default_fps

    # Optional: round for nicer filenames/UI
    fps = float(np.round(fps, 3))
    return frame_list, fps
# -----------------------------------------
# 3) visualize_frames using sparse loaders
# -----------------------------------------
def visualize_frames(frame_list: List[int],
                     config: Optional['VizConfig'] = None,
                     **overrides: Any) -> None:
    """
    Render only the specified absolute frames (e.g., an 'incident' list).
    Saves an MP4 (and optionally PNGs) as controlled by config/write_pngs.
    """
    if not frame_list:
        raise ValueError("frame_list is empty.")

    # Keep user-provided order; allow duplicates
    frames = list(frame_list)

    c = _apply_overrides(config or DEFAULT_CFG, overrides)

    cam, video_path, label3d_path, pred_path, com_path, save_path = _prepare_io(c)
    COLOR = connectivity.COLOR_DICT[c.animal]
    CONNECTIVITY = connectivity.CONNECTIVITY_DICT[c.animal]
    cameras = load_cameras(label3d_path)

    # new sparse loaders
    pred_2d = _load_and_project_preds_frames(pred_path, cameras, cam, frames)
    pred_2d_com = _load_and_project_com_frames(com_path, cameras, cam, frames)

    vids = imageio.get_reader(video_path)
    fig = plt.figure()
    plt.rcParams['figure.figsize'] = (6, 6)

    tag = f"custom_frames_{frames[0]}_{frames[-1]}_{len(frames)}f"
    vid_name = (c.out_name if c.out_name else f"combined_cam{c.cammm}_{tag}") + ".mp4"
    writer = FFMpegWriter(fps=c.video_fps, metadata=dict(title='combined_visualization', artist='Matplotlib'))

    out_mp4 = os.path.join(save_path, "vis_" + vid_name)
    with writer.saving(fig, out_mp4, dpi=300):
        for j, f_abs in enumerate(tqdm.tqdm(frames)):
            # random-access the exact requested frame
            img = vids.get_data(f_abs)

            k = pred_2d[cam][j]  # (22,2) or (2,22,2)
            if k.ndim == 2:
                k_a1, k_a2 = k, None
            else:
                k_a1, k_a2 = k[0], k[1]

            k_com = pred_2d_com[cam][j]  # (A,2)

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



import os, numpy as np, matplotlib.pyplot as plt, matplotlib.ticker as mticker
from matplotlib.animation import FFMpegWriter
from typing import Dict, Optional, List, Any
import imageio, tqdm

# --- Helpers: light + local, no external deps beyond your existing module functions ---

def _detect_frame_col(frames) -> Optional[str]:
    for cand in ["camera_frame_sixcam", "mapped_sixcam_frame_indices"]:
        if cand in frames.columns:
            return cand
    return None  # fall back to index

def _abs_frame_at(frames, idx: int, frame_col: Optional[str]) -> int:
    if frame_col:
        return int(frames.iloc[idx][frame_col])
    # assume frames index is absolute six-cam frame
    return int(frames.index[idx])

def _percentile_ylim(y: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> Optional[tuple]:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return None
    a, b = np.percentile(y, [lo, hi])
    if not np.isfinite(a) or not np.isfinite(b) or a == b:
        return None
    pad = 0.05 * max(1e-6, abs(b - a))
    return (a - pad, b + pad)

def _maybe(frames, name: str):
    return frames[name].to_numpy() if name in frames.columns else None

def _build_tracks(frames, s: int, e: int) -> Dict[str, np.ndarray]:
    # Core candidates; only include those present
    candidates = [
        ("dist_mm", "Distance (mm)"),
        ("dD_dt", "Radial speed (mm/s)"),
        ("snout_dist_mm", "Snout–snout (mm)"),
        ("com1_z", "COM1 z"),
        ("com2_z", "COM2 z"),
        ("heading_diff_deg", "Heading Δ (deg)"),
    ]
    out = {}
    for key, _ in candidates:
        arr = _maybe(frames, key)
        if arr is not None:
            out[key] = arr[s:e]
    return out

def _axis_label(key: str) -> str:
    return {
        "dist_mm": "Distance (mm)",
        "dD_dt": "Radial speed (mm/s)",
        "snout_dist_mm": "Snout–snout (mm)",
        "com1_z": "COM1 z",
        "com2_z": "COM2 z",
        "heading_diff_deg": "Heading Δ (deg)",
    }.get(key, key)

def _prepare_pred_overlays(c, label3d_path, pred_path, com_path, cameras, cam, abs_frames: List[int]):
    # Project only the frames we need (contiguous event)
    pred_2d = _load_and_project_preds_frames(pred_path, cameras, cam, abs_frames)
    pred_2d_com = _load_and_project_com_frames(com_path, cameras, cam, abs_frames)
    return pred_2d, pred_2d_com

# --- Main: event-clip renderer with live cursor on strip-charts ---

def visualize_event_clip(
    ev: Dict[str, Any],
    frames,                         # your res["frames"] DataFrame
    config: Optional['VizConfig'] = None,
    contact_mm: float = 50.0,
    frame_col: Optional[str] = None,
    **overrides: Any
) -> str:
    """
    Render a contiguous event [start_idx, end_idx_exclusive) as:
      Left: video with 2D overlays
      Right: stacked strip-charts with moving cursor

    Returns: path to the written MP4.
    """
    c = _apply_overrides(config or DEFAULT_CFG, overrides)
    s = int(ev["start_idx"])
    e = int(ev["end_idx_exclusive"])
    if e <= s:
        raise ValueError(f"Bad event bounds: start={s}, end={e}")

    # Resolve absolute frames for the contiguous span
    frame_col = frame_col or _detect_frame_col(frames)
    rel_indices = list(range(s, e))
    abs_frames = [ _abs_frame_at(frames, i, frame_col) for i in rel_indices ]
    nF = len(abs_frames)

    # Prepare IO & model assets
    cam, video_path, label3d_path, pred_path, com_path, save_path = _prepare_io(c)
    COLOR = connectivity.COLOR_DICT.get(c.animal, connectivity.COLOR_DICT[next(iter(connectivity.COLOR_DICT))])
    CONNECTIVITY = connectivity.CONNECTIVITY_DICT.get(c.animal, connectivity.CONNECTIVITY_DICT[next(iter(connectivity.CONNECTIVITY_DICT))])
    cameras = load_cameras(label3d_path)

    # Overlays (project only needed frames)
    pred_2d, pred_2d_com = _prepare_pred_overlays(c, label3d_path, pred_path, com_path, cameras, cam, abs_frames)

    # Build tracks from DataFrame
    tracks = _build_tracks(frames, s, e)
    # keep a consistent order, limited to a small stack
    preferred_order = ["dist_mm", "dD_dt", "snout_dist_mm", "com1_z", "com2_z", "heading_diff_deg"]
    keys = [k for k in preferred_order if k in tracks][:3]  # default to top-3 present
    if not keys:
        # Always try to show something; fall back to dist-like info if possible
        raise ValueError("No known signals found in frames for plotting (dist_mm/dD_dt/... not present).")

    # Time axis in frames; seconds helper
    x = np.arange(nF)  # event-local frame index 0..nF-1
    fps = float(c.video_fps)
    secs = x / fps

    # Figure/layout
    # Use a simple 2-col GridSpec: left big, right narrow with N stacked axes
    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    gs = fig.add_gridspec(nrows=max(len(keys), 1), ncols=2, width_ratios=[2.4, 1.0], wspace=0.05, hspace=0.15)

    # Left: video axes spans all rows
    ax_vid = fig.add_subplot(gs[:, 0])
    ax_vid.set_axis_off()

    # Right: stacked strip charts
    axes = []
    lines = []
    cursors = []
    thresh_handles = []
    y_text_handle = None

    for r, key in enumerate(keys):
        ax = fig.add_subplot(gs[r, 1], sharex=axes[0] if axes else None)
        y = tracks[key]
        ln, = ax.plot(x, y, lw=1.1)
        axes.append(ax)
        lines.append(ln)
        # Cursor
        cur = ax.axvline(0, ls='--', lw=0.8)
        cursors.append(cur)

        # Y-limits (robust)
        ylim = _percentile_ylim(y)
        if ylim: ax.set_ylim(*ylim)

        # Labels: only bottom axis gets x label
        if r < len(keys) - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("frame")
        ax.set_ylabel(_axis_label(key), fontsize=8)

        # Optional: contact threshold on dist track
        if key == "dist_mm":
            th = ax.axhline(float(contact_mm), ls=":", lw=0.9)
            thresh_handles.append(th)

    # Top secondary x-axis (seconds) on the top strip only
    top_ax = axes[0] if axes else None
    if top_ax is not None:
        secax = top_ax.secondary_xaxis('top', functions=(lambda f: f / fps, lambda s: s * fps))
        secax.set_xlabel("time (s)")
        secax.xaxis.set_major_locator(mticker.MaxNLocator(5))

    # Event annotations on dist axis (contact, trough) if available
    if "contact_idx" in ev and "dist_mm" in keys:
        # event-local index:
        contact_rel = int(ev["contact_idx"]) - s
        if 0 <= contact_rel < nF:
            yval = tracks["dist_mm"][contact_rel]
            axes[keys.index("dist_mm")].scatter([contact_rel], [yval], marker="x", s=28, zorder=4)

    # Live numeric overlay on video (small, monospaced)
    info_text = ax_vid.text(
        0.01, 0.99, "", transform=ax_vid.transAxes, va="top", ha="left", fontsize=8, family="monospace",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6)
    )

    # Video reader + writer
    vids = imageio.get_reader(video_path)
    tag = f"ev_{s}_{e}_cam{c.cammm}"
    vid_name = (c.out_name if c.out_name else f"eventclip_{tag}") + ".mp4"
    out_mp4 = os.path.join(save_path, "vis_" + vid_name)

    writer = FFMpegWriter(fps=c.video_fps, metadata=dict(title='event_visualization', artist='Matplotlib'))

    # --- Main encode loop ---
    with writer.saving(fig, out_mp4, dpi=300):
        for j in tqdm.tqdm(range(nF), desc="Rendering event"):
            f_abs = abs_frames[j]

            # Frame image
            img = vids.get_data(f_abs)
            ax_vid.imshow(img)
            ax_vid.set_axis_off()

            # 2D overlay (projected keypoints & COM). Fail-soft if missing
            try:
                k = pred_2d[cam][j]
                if k.ndim == 2:
                    k_a1, k_a2 = k, None
                else:
                    k_a1, k_a2 = k[0], k[1]
                k_com = pred_2d_com[cam][j]  # (A,2)
                _draw_frame(
                    img, k_a1, k_a2, k_com,
                    COLOR, CONNECTIVITY,
                    enable_zoom=c.enable_zoom, zoom_margin=c.zoom_margin,
                    drop_tail_for_view=c.drop_tail_for_view, include_both=c.zoom_include_both,
                    title_text=f"cam{c.cammm} abs {f_abs}"
                )
            except Exception:
                # Keep going even if overlay failed
                pass

            # Move cursors + update live text
            for cur in cursors:
                cur.set_xdata([j, j])

            # Compose small info line from available tracks at j
            vals = []
            for kname in keys:
                y = tracks[kname]
                val = y[j] if np.isfinite(y[j]) else np.nan
                if kname == "dist_mm":
                    vals.append(f"D={val:.1f}mm")
                elif kname == "dD_dt":
                    vals.append(f"dD/dt={val:.2f}")
                elif kname == "snout_dist_mm":
                    vals.append(f"S={val:.1f}mm")
                elif kname.endswith("_z"):
                    vals.append(f"{kname}={val:.2f}")
                else:
                    vals.append(f"{kname}={val:.2f}")
            info_text.set_text(f"t={j:04d} ({j/fps:.2f}s) | " + "  ".join(vals))

            writer.grab_frame()
            ax_vid.cla()  # clear only the image/overlays for next frame

    plt.close(fig)
    return out_mp4
