"""
3D Keypoint Social Interaction Analysis Module

Author: Mir Qi  
Date: October 2024
"""

from typing import Optional, Union, Tuple, Dict, List, Any, Sequence
import os
import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.stats import zscore
from contextlib import contextmanager

try:
    import imageio
    import tqdm
    from matplotlib.animation import FFMpegWriter
except ImportError:
    pass

# ==============================================================================
# CONSTANTS
# ==============================================================================

MOUSE22_EDGES = [
    (0,1),(1,2),(0,2),
    (0,3),(1,3),(2,3),
    (3,4),(4,5),(5,6),(6,7),
    (8,9),(9,10),(10,11),(11,3),
    (12,13),(13,14),(14,15),(15,3),
    (16,17),(17,18),(18,4),
    (19,20),(20,21),(21,4),
]

KP = {
    'EarL':1, 'EarR':2, 'Snout':3, 'SpineF':4, 'SpineM':5,
    'Tail(base)':6, 'Tail(mid)':7, 'Tail(end)':8,
    'ForepawL':9, 'WristL':10, 'ElbowL':11, 'ShoulderL':12,
    'ForepawR':13, 'WristR':14, 'ElbowR':15, 'ShoulderR':16,
    'HindpawL':17, 'AnkleL':18, 'KneeL':19, 
    'HindpawR':20, 'AnkleR':21, 'KneeR':22
}


# ==============================================================================
# Cell 4
# ==============================================================================

def distance3d(x1, y1, z1, x2, y2, z2):
    x1, y1, z1, x2, y2, z2 = map(np.asarray, (x1, y1, z1, x2, y2, z2))
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def compute_com_distance(
    df: pd.DataFrame,
    p1: str = "com1",
    p2: str = "com2",
    smooth_window: Optional[int] = None,
    dist_smooth_window: Optional[int] = None,
    return_components: bool = False,   # <-- added
):
    cols = [f"{p1}_x", f"{p1}_y", f"{p1}_z", f"{p2}_x", f"{p2}_y", f"{p2}_z"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # optional smoothing on positions first
    pos = (df[cols].rolling(int(smooth_window), center=True, min_periods=1).mean()
           if smooth_window and smooth_window > 1 else df[cols])

    x1, y1, z1 = pos[f"{p1}_x"], pos[f"{p1}_y"], pos[f"{p1}_z"]
    x2, y2, z2 = pos[f"{p2}_x"], pos[f"{p2}_y"], pos[f"{p2}_z"]

    dx = (x2 - x1).to_numpy()
    dy = (y2 - y1).to_numpy()
    dz = (z2 - z1).to_numpy()
    d  = distance3d(x1, y1, z1, x2, y2, z2)

    if return_components:
        out = pd.DataFrame(
            {"dx": dx, "dy": dy, "dz": dz, "dist_mm": d},
            index=df.index
        )
        if dist_smooth_window and dist_smooth_window > 1:
            out["dist_mm"] = out["dist_mm"].rolling(
                int(dist_smooth_window), center=True, min_periods=1
            ).mean()
        return out

    dist_s = pd.Series(d, index=df.index, name="dist_mm")
    if dist_smooth_window and dist_smooth_window > 1:
        dist_s = dist_s.rolling(int(dist_smooth_window), center=True, min_periods=1).mean()
    return dist_s

def _kp_xy_row(df, idx, animal="a1"):
    """
    Return (22,2) array for x,y at a single frame index.
    Row i (0-based) corresponds to kp{i+1}.
    """
    r = df.loc[idx]
    xy = np.empty((22, 2), dtype=float)
    for k in range(1, 23):  # 1..22
        xy[k-1, 0] = r.get(f"kp{k}_x_{animal}", np.nan)
        xy[k-1, 1] = r.get(f"kp{k}_y_{animal}", np.nan)
    return xy

def _pair_dist(a_xy, b_xy, i_a, i_b):
    """
    Euclidean distance between point i_a (0-based) of a_xy and i_b of b_xy.
    """
    dx = b_xy[i_b,0] - a_xy[i_a,0]
    dy = b_xy[i_b,1] - a_xy[i_a,1]
    return float((dx*dx + dy*dy) ** 0.5)

def _time_label(idx_name, idx_value):
    if idx_name and ("ms" in str(idx_name).lower()):
        return f"{idx_value/1000.0:.3f}s"
    return f"frame {int(idx_value)}"

def plot_skeleton_frames(
    df,
    rows,
    n=10,
    animal1="a1",
    animal2="a2",
    pairs=(("Snout","Snout"), ("Snout","Tail(base)"), ("Snout","Tail(end)")),
    threshold_mm=None,      # if given, pair lines under threshold are highlighted thicker
    invert_y=False          # set True if your y-axis is image-style
):
    idx_sel = list(rows)[:int(n)]
    if not idx_sel:
        raise ValueError("No rows to plot.")

    # precompute global xy limits (both animals, selected frames)
    xs, ys = [], []
    for idx in idx_sel:
        a1 = _kp_xy_row(df, idx, animal=animal1)
        a2 = _kp_xy_row(df, idx, animal=animal2)
        xs.extend([a1[:,0].min(), a1[:,0].max(), a2[:,0].min(), a2[:,0].max()])
        ys.extend([a1[:,1].min(), a1[:,1].max(), a2[:,1].min(), a2[:,1].max()])
    xmin, xmax = np.nanmin(xs), np.nanmax(xs)
    ymin, ymax = np.nanmin(ys), np.nanmax(ys)
    pad_x = 0.05 * max(1.0, xmax - xmin)
    pad_y = 0.05 * max(1.0, ymax - ymin)

    # grid
    cols = min(5, max(1, len(idx_sel)))
    rows_grid = int(np.ceil(len(idx_sel) / cols))
    fig, axes = plt.subplots(rows_grid, cols, figsize=(4*cols, 4*rows_grid), squeeze=False)

    # prepare pair indices (0-based)
    def _to_kp0(x):
        return (KP[x]-1) if isinstance(x, str) else (int(x)-1)

    pair_idx = [(_to_kp0(a), _to_kp0(b)) for (a,b) in pairs]

    # draw
    for k, idx in enumerate(idx_sel):
        r = k // cols
        c = k % cols
        ax = axes[r][c]
        a1 = _kp_xy_row(df, idx, animal=animal1)
        a2 = _kp_xy_row(df, idx, animal=animal2)

        # skeleton lines
        for i, j in MOUSE22_EDGES:
            if not (np.any(np.isnan(a1[[i,j]])) or np.any(np.isnan(a2[[i,j]]))):
                ax.plot([a1[i,0], a1[j,0]], [a1[i,1], a1[j,1]], linewidth=1)      # animal1
                ax.plot([a2[i,0], a2[j,0]], [a2[i,1], a2[j,1]], linewidth=1)      # animal2

        # keypoints (small)
        ax.scatter(a1[:,0], a1[:,1], s=8)
        ax.scatter(a2[:,0], a2[:,1], s=8)

        # COM if present
        if f"com1_x" in df.columns and f"com1_y" in df.columns:
            ax.scatter(df.loc[idx, "com1_x"], df.loc[idx, "com1_y"], s=30, marker='x')
        if f"com2_x" in df.columns and f"com2_y" in df.columns:
            ax.scatter(df.loc[idx, "com2_x"], df.loc[idx, "com2_y"], s=30, marker='x')

        # pair lines (across animals)
        title_bits = []
        for ia, ib in pair_idx:
            if (not np.any(np.isnan(a1[ia])) and not np.any(np.isnan(a2[ib]))):
                d = _pair_dist(a1, a2, ia, ib)
                lw = 2.5 if (threshold_mm is not None and d <= float(threshold_mm)) else 1.0
                ax.plot([a1[ia,0], a2[ib,0]], [a1[ia,1], a2[ib,1]], linewidth=lw)
                title_bits.append(f"d{k}-{ia+1}->{ib+1}:{d:.1f}mm")
        # time/idx in title
        ax.set_title(_time_label(df.index.name, idx))

        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        if invert_y:
            ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])

    # hide any unused axes
    total = rows_grid * cols
    for k in range(len(idx_sel), total):
        r = k // cols
        c = k % cols
        axes[r][c].axis('off')

    plt.tight_layout()
    plt.show()


# ==============================================================================
# Cell 5
# ==============================================================================

def _time_seconds(df: pd.DataFrame, time_col: Optional[str], fps: Optional[float]) -> Tuple[np.ndarray, str]:
    """
    Returns (t_sec, units_str). If time_col provided and name contains 'ms', converts to seconds.
    If no time_col and fps is given, builds t from frame index. Else uses frame units.
    """
    if time_col and (time_col in df.columns):
        t = df[time_col].to_numpy()
        if "ms" in time_col.lower():
            return t / 1000.0, "s"
        return t.astype(float), "s"
    if fps and fps > 0:
        n = len(df)
        return np.arange(n, dtype=float) / float(fps), "s"
    return np.arange(len(df), dtype=float), "frames"

def compute_motion_direction(
    df: pd.DataFrame,
    prefix: str = "com1",
    time_col: Optional[str] = "timestamp_ms_mini",
    fps: Optional[float] = None,
    pos_smooth: int = 1,
    vel_smooth: int = 1,
    fill_na: bool = False, #True,
    eps: float = 1e-9
) -> pd.DataFrame:
    """
    Computes velocity (vx,vy,vz), speed, and unit direction (ux,uy,uz).
    Units are mm/s if time axis is seconds; else mm/frame.
    """
    need = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns for '{prefix}': {miss}")

    pos = df[need].copy()
    if pos_smooth and pos_smooth > 1:
        pos = pos.rolling(window=int(pos_smooth), center=True, min_periods=1).mean()
    if fill_na:
        pos = pos.interpolate(method="linear", limit_direction="both")

    t_sec, t_units = _time_seconds(df, time_col, fps)

    vx = np.gradient(pos[f"{prefix}_x"].to_numpy(), t_sec)
    vy = np.gradient(pos[f"{prefix}_y"].to_numpy(), t_sec)
    vz = np.gradient(pos[f"{prefix}_z"].to_numpy(), t_sec)

    vel = pd.DataFrame({f"{prefix}_vx": vx, f"{prefix}_vy": vy, f"{prefix}_vz": vz}, index=df.index)
    if vel_smooth and vel_smooth > 1:
        vel = vel.rolling(window=int(vel_smooth), center=True, min_periods=1).mean()

    speed = np.sqrt(vel[f"{prefix}_vx"]**2 + vel[f"{prefix}_vy"]**2 + vel[f"{prefix}_vz"]**2)
    ux = vel[f"{prefix}_vx"] / (speed + eps)
    uy = vel[f"{prefix}_vy"] / (speed + eps)
    uz = vel[f"{prefix}_vz"] / (speed + eps)

    out = pd.concat([
        vel,
        pd.Series(speed, index=df.index, name=f"{prefix}_speed"),
        pd.Series(ux, index=df.index, name=f"{prefix}_ux"),
        pd.Series(uy, index=df.index, name=f"{prefix}_uy"),
        pd.Series(uz, index=df.index, name=f"{prefix}_uz"),
    ], axis=1)
    out.attrs["units"] = "mm/s" if t_units == "s" else "mm/frame"
    return out

def _boolean_runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run-length encoding of a boolean mask.
    Returns (starts, ends) as index positions where mask is True. Interval is [start, end).
    """
    if mask.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    m = mask.astype(np.int8)
    edges = np.diff(np.pad(m, (1, 1), constant_values=0))
    starts = np.flatnonzero(edges == 1)
    ends   = np.flatnonzero(edges == -1)
    return starts, ends

def detect_approaches(
    df: pd.DataFrame,
    p1: str = "com1",
    p2: str = "com2",
    time_col: Optional[str] = "timestamp_ms_mini",
    fps: Optional[float] = None,
    pos_smooth: int = 3,
    vel_smooth: int = 3,
    radial_thresh: float = 20.0,
    speed_min: float = 5.0,
    dist_min: Optional[float] = None,
    dist_max: Optional[float] = None,
    min_samples: int = 5,
    return_intervals: bool = True,
    eps: float = 1e-9
) -> Dict[str, Union[pd.DataFrame, Dict[str, List[dict]]]]:

    # 1) Distances & relative vector
    comps = compute_com_distance(df, p1, p2, smooth_window=pos_smooth, return_components=True)
    dx, dy, dz = comps["dx"].to_numpy(), comps["dy"].to_numpy(), comps["dz"].to_numpy()
    dist = comps["dist_mm"].to_numpy()
    u12 = np.column_stack([dx, dy, dz])
    denom = np.maximum(dist, eps)[:, None]
    u12 = u12 / denom

    # 2) Velocities & speeds
    m1 = compute_motion_direction(df, prefix=p1, time_col=time_col, fps=fps,
                                  pos_smooth=pos_smooth, vel_smooth=vel_smooth)
    m2 = compute_motion_direction(df, prefix=p2, time_col=time_col, fps=fps,
                                  pos_smooth=pos_smooth, vel_smooth=vel_smooth)

    v1 = np.column_stack([m1[f"{p1}_vx"].to_numpy(),
                          m1[f"{p1}_vy"].to_numpy(),
                          m1[f"{p1}_vz"].to_numpy()])
    v2 = np.column_stack([m2[f"{p2}_vx"].to_numpy(),
                          m2[f"{p2}_vy"].to_numpy(),
                          m2[f"{p2}_vz"].to_numpy()])
    speed1 = m1[f"{p1}_speed"].to_numpy()
    speed2 = m2[f"{p2}_speed"].to_numpy()

    # 3) Radial-toward components
    radial1 = (v1 * u12).sum(axis=1)
    radial2 = (v2 * (-u12)).sum(axis=1)

    # 4) Distance rate dD/dt
    t_sec, t_units = _time_seconds(df, time_col, fps)
    dD_dt = np.gradient(dist, t_sec)

    # 5) Flags with optional distance gating
    in_band = np.ones_like(dist, dtype=bool)
    if dist_min is not None:
        in_band &= (dist >= float(dist_min))
    if dist_max is not None:
        in_band &= (dist <= float(dist_max))

    approach1 = (radial1 >= radial_thresh) & (speed1 >= speed_min) & in_band
    approach2 = (radial2 >= radial_thresh) & (speed2 >= speed_min) & in_band
    mutual    = approach1 & approach2

    # 6) Min-duration cleanup (frames-level)
    def _minlen_filter(mask: np.ndarray, k: int) -> np.ndarray:
        if k <= 1:
            return mask
        s, e = _boolean_runs(mask)
        keep = np.ones_like(mask, dtype=bool)
        for i in range(len(s)):
            if (e[i] - s[i]) < k:
                keep[s[i]:e[i]] = False
        return mask & keep

    if min_samples and min_samples > 1:
        approach1 = _minlen_filter(approach1, int(min_samples))
        approach2 = _minlen_filter(approach2, int(min_samples))
        mutual    = _minlen_filter(mutual,    int(min_samples))

    # 6.5) Pass-through alignment columns if present
    passthrough_cols = []
    for c in ("camera_frame_sixcam", "mapped_sixcam_frame_indices"):
        if c in df.columns:
            passthrough_cols.append(c)

    # 7) Assemble frames
    data = {
        "dist_mm": dist,
        "dD_dt": dD_dt,
        "radial1": radial1,
        "radial2": radial2,
        "speed1": speed1,
        "speed2": speed2,
        "approach1": approach1,
        "approach2": approach2,
        "mutual": mutual,
    }
    # attach passthroughs under original names
    for c in passthrough_cols:
        data[c] = df[c].to_numpy()

    frames = pd.DataFrame(data, index=df.index)
    frames.attrs["velocity_units"] = "mm/s" if t_units == "s" else "mm/frame"

    intervals: Dict[str, List[dict]] = {}
    if return_intervals:
        def _summarize(mask: np.ndarray, label: str) -> List[dict]:
            s, e = _boolean_runs(mask)
            if s.size == 0:
                return []
            out: List[dict] = []
            for i in range(len(s)):
                sl = slice(s[i], e[i])
                t0 = float(t_sec[s[i]])
                t1 = float(t_sec[e[i]-1])
                out.append({
                    "label": label,
                    "start_idx": int(s[i]),
                    "end_idx_exclusive": int(e[i]),
                    "start_time_s": t0 if t_units == "s" else None,
                    "end_time_s":   t1 if t_units == "s" else None,
                    "duration_s":   (t1 - t0) if t_units == "s" else None,
                    "min_dist_mm": float(np.nanmin(dist[sl])),
                    "median_dist_mm": float(np.nanmedian(dist[sl])),
                    "max_radial_mm_per_s": float(np.nanmax(radial1[sl] if label=="approach1" else radial2[sl])),
                    "median_speed_mm_per_s": float(np.nanmedian(speed1[sl] if label=="approach1" else speed2[sl])),
                })
            return out

        intervals = {
            "approach1": _summarize(approach1, "approach1"),
            "approach2": _summarize(approach2, "approach2"),
            "mutual":    _summarize(mutual,    "mutual"),
        }

    return {"frames": frames, "intervals": intervals}

def _kp_idx(k):
    return KP[k] if isinstance(k, str) else int(k)

def get_proximity_rows_by_com(df, threshold_mm=260.0, p1="com1", p2="com2", **kwargs):
    """
    Return df.index where CoM distance <= threshold_mm.
    kwargs are passed to compute_com_distance (e.g., smooth_window=3).
    """
    d = compute_com_distance(df, p1=p1, p2=p2, **kwargs)  # already in mm
    return d.index[d <= float(threshold_mm)]

def point_distance(df, kp_a1, kp_a2, a1="a1", a2="a2"):
    """
    Euclidean distance (mm) between kp_a1 of a1 and kp_a2 of a2.
    kp_* can be name ('Snout') or index (3).
    Expects columns like: kp3_x_a1, kp3_y_a1, kp3_z_a1, kp3_x_a2, ...
    Works for 2D or 3D depending on what columns exist.
    """
    i1 = _kp_idx(kp_a1)
    i2 = _kp_idx(kp_a2)

    # auto-detect axes present
    axes = [ax for ax in ("x","y","z")
            if f"kp{i1}_{ax}_{a1}" in df.columns and f"kp{i2}_{ax}_{a2}" in df.columns]
    if not axes:
        raise ValueError("No matching kp columns found (check names/animals).")

    # simple sqrt of squared diffs
    s = 0.0
    for ax in axes:
        a = df[f"kp{i1}_{ax}_{a1}"]
        b = df[f"kp{i2}_{ax}_{a2}"]
        s = s + (b - a)**2
    return (s**0.5).rename(f"kp{i1}_{a1}__to__kp{i2}_{a2}_mm")

def snout_to(df, others=("Snout","Tail(base)", 'SpineM'), rows=None, a1="a1", a2="a2"):
    """
    Return a DataFrame of snout(a1) to each item in `others` on selected rows.
    """
    idx = df.index if rows is None else rows
    out = {}
    for o in others:
        dist = point_distance(df.loc[idx], "Snout", o, a1=a1, a2=a2)
        # short label
        o_lab = (o if isinstance(o,str) else f"kp{int(o)}").replace("(", "").replace(")", "").replace(" ", "").lower()
        out[f"snout_{a1}__to__{o_lab}_{a2}_mm"] = dist
    return pd.DataFrame(out, index=idx)


# ==============================================================================
# Cell 8
# ==============================================================================

def _kp_xyz_row(df, idx, animal="a1", n_kp=22):
    """(n_kp,3) xyz for one frame; z can fall back to 0 if missing."""
    r = df.loc[idx]
    xyz = np.empty((n_kp, 3), dtype=float)
    for k in range(1, n_kp+1):
        xyz[k-1, 0] = r.get(f"kp{k}_x_{animal}", np.nan)
        xyz[k-1, 1] = r.get(f"kp{k}_y_{animal}", np.nan)
        z = r.get(f"kp{k}_z_{animal}", 0.0)
        xyz[k-1, 2] = z if np.isfinite(z) else 0.0
    return xyz

def _axes_equal_3d(ax, P):
    """Cube-like aspect for 3D."""
    mask = np.isfinite(P).all(axis=1)
    if not mask.any():
        return
    x, y, z = P[mask,0], P[mask,1], P[mask,2]
    cx, cy, cz = x.mean(), y.mean(), z.mean()
    r = max(x.ptp(), y.ptp(), z.ptp(), 1.0) * 0.55
    ax.set_xlim(cx-r, cx+r); ax.set_ylim(cy-r, cy+r); ax.set_zlim(cz-r, cz+r)

def plot_skeleton_frame_3d_const_colors(
    df,
    idx,
    COLOR,
    CONNECTIVITY,
    animal1="a1",
    animal2="a2",
    elev=24, azim=-58,
    invert_y=False,
    dpi=300,
    kp_size=14, lw=2.0
):
    a1 = _kp_xyz_row(df, idx, animal=animal1)
    a2 = _kp_xyz_row(df, idx, animal=animal2)

    fig = plt.figure(figsize=(6.2, 6.2), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # draw colored edges (same palette for both animals)
    for color, (i, j) in zip(COLOR, CONNECTIVITY):
        if np.isfinite(a1[[i,j]]).all():
            ax.plot([a1[i,0], a1[j,0]], [a1[i,1], a1[j,1]], [a1[i,2], a1[j,2]], c=color, lw=lw)
        if np.isfinite(a2[[i,j]]).all():
            ax.plot([a2[i,0], a2[j,0]], [a2[i,1], a2[j,1]], [a2[i,2], a2[j,2]], c=color, lw=lw, alpha=0.9)

    # keypoints (white for a1, cyan for a2)
    ax.scatter(a1[:,0], a1[:,1], a1[:,2], s=kp_size, color='white', depthshade=False)
    ax.scatter(a2[:,0], a2[:,1], a2[:,2], s=kp_size, color='cyan',  depthshade=False, alpha=0.9)

    # COM dots (red) if present; fall back z=0 if no z column
    def _plot_com(prefix):
        xk, yk, zk = f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"
        if xk in df.columns and yk in df.columns:
            cx, cy = df.loc[idx, xk], df.loc[idx, yk]
            cz = df.loc[idx, zk] if zk in df.columns else 0.0
            if np.isfinite([cx, cy, cz]).all():
                ax.scatter([cx], [cy], [cz], s=60, marker='.', color='red', alpha=0.6)
    _plot_com("com1"); _plot_com("com2")

    both = np.vstack([a1, a2])
    _axes_equal_3d(ax, both)
    if invert_y:
        ax.invert_yaxis()

    ax.set_xlabel("X (mm)", labelpad=6)
    ax.set_ylabel("Y (mm)", labelpad=6)
    ax.set_zlabel("Z (mm)", labelpad=6)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.set_box_aspect([1,1,1])
    ax.set_title(_time_label(df.index.name, idx), pad=10)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# Cell 11
# ==============================================================================

def _time_label(idx_name, idx_value):
    """Readable label for time/frame index."""
    if idx_name and ("ms" in str(idx_name).lower()):
        return f"{float(idx_value)/1000.0:.3f}s"
    try:
        return f"frame {int(idx_value)}"
    except Exception:
        return f"{idx_value}"

def _col(animal: str, kp_id: int, axis: str) -> str:
    """Build column name like 'kp3_x_a1'."""
    return f"kp{int(kp_id)}_{axis}_{animal}"

def _distance_series(df: pd.DataFrame,
                     kp_a1: str,
                     kp_a2: str,
                     animal1: str = "a1",
                     animal2: str = "a2",
                     use_z: bool = True,
                     smooth_window: int = 1) -> np.ndarray:
    """
    Vectorized per-frame distance between animal1.kp_a1 and animal2.kp_a2.
    Uses 3D if both z columns exist and use_z=True; otherwise 2D.
    Returns a float numpy array (NaN where invalid).
    """
    i1 = KP[kp_a1]; i2 = KP[kp_a2]

    x1 = df[_col(animal1, i1, "x")].to_numpy(dtype=float, copy=False)
    y1 = df[_col(animal1, i1, "y")].to_numpy(dtype=float, copy=False)
    x2 = df[_col(animal2, i2, "x")].to_numpy(dtype=float, copy=False)
    y2 = df[_col(animal2, i2, "y")].to_numpy(dtype=float, copy=False)

    use3d = False
    if use_z:
        zc1 = _col(animal1, i1, "z")
        zc2 = _col(animal2, i2, "z")
        if (zc1 in df.columns) and (zc2 in df.columns):
            z1 = df[zc1].to_numpy(dtype=float, copy=False)
            z2 = df[zc2].to_numpy(dtype=float, copy=False)
            use3d = True
        else:
            z1 = z2 = None
    else:
        z1 = z2 = None

    # distance with NaN propagation if any coord missing
    dx = x2 - x1
    dy = y2 - y1
    if use3d:
        dz = z2 - z1
        d = np.sqrt(dx*dx + dy*dy + dz*dz, dtype=float)
    else:
        d = np.sqrt(dx*dx + dy*dy, dtype=float)

    # optional light smoothing (centered)
    if smooth_window and smooth_window > 1:
        # pandas rolling handles NaNs gracefully
        d = pd.Series(d, index=df.index, copy=False).rolling(
            int(smooth_window), center=True, min_periods=1
        ).mean().to_numpy()

    return d

def find_incidents_by_pair(df: pd.DataFrame,
                           pair: tuple[str, str],
                           threshold: float,
                           animal1: str = "a1",
                           animal2: str = "a2",
                           use_z: bool = True,
                           smooth_window: int = 1,
                           top_k: int = 12,
                           min_spacing_frames: int = 0,
                           return_details: bool = False):
    """
    Return up to top_k incident indices where the distance between
    (animal1.pair[0]) and (animal2.pair[1]) dips below 'threshold'.

    Strategy:
      1) compute distance series (3D if available) -> d
      2) mask frames under threshold
      3) split into contiguous runs and pick argmin in each run
      4) if too many, keep by smallest distance with spacing constraint

    Parameters
    ----------
    df : DataFrame with columns like 'kp{1..22}_{x,y[,z]}_{a1/a2}'.
    pair : (name_a1, name_a2), e.g., ("Snout", "Tail(base)").
    threshold : float, in your units (mm if your coords are mm).
    animal1, animal2 : column suffixes for animals, default "a1", "a2".
    use_z : if True, uses z when both z columns exist.
    smooth_window : centered moving-average window (1 disables).
    top_k : max incidents to return.
    min_spacing_frames : enforce separation between returned minima.
    return_details : if True, returns a DataFrame with details; else list of index labels.

    Returns
    -------
    list of index labels (default), or a DataFrame with columns:
        ['index', 'distance', 'time_label'] (sorted by time)
    """
    name_a1, name_a2 = pair
    d = _distance_series(df, name_a1, name_a2, animal1, animal2, use_z, smooth_window)

    finite = np.isfinite(d)
    under = finite & (d <= float(threshold))

    if not np.any(under):
        return pd.DataFrame(columns=["index", "distance", "time_label"]) if return_details else []

    # contiguous runs where 'under' is True
    u = under.astype(np.int8)
    diff = np.diff(u, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0] - 1  # inclusive

    # one argmin per run
    mins_pos = []
    mins_val = []
    for s, e in zip(starts, ends):
        seg = d[s:e+1]
        if seg.size == 0 or not np.any(np.isfinite(seg)):
            continue
        j = np.nanargmin(seg)   # within-run argmin
        mins_pos.append(s + j)
        mins_val.append(seg[j])

    if not mins_pos:
        return pd.DataFrame(columns=["index", "distance", "time_label"]) if return_details else []

    mins_pos = np.asarray(mins_pos, dtype=int)
    mins_val = np.asarray(mins_val, dtype=float)

    # spacing + top_k selection: greedy by ascending distance
    order = np.argsort(mins_val)  # smaller distance first
    selected = []
    for idx in order:
        p = mins_pos[idx]
        if min_spacing_frames > 0:
            if any(abs(p - q) <= min_spacing_frames for q in selected):
                continue
        selected.append(p)
        if len(selected) >= int(top_k):
            break

    if not selected:
        return pd.DataFrame(columns=["index", "distance", "time_label"]) if return_details else []

    selected = np.sort(np.asarray(selected, dtype=int))
    sel_vals = d[selected]
    sel_idx_labels = df.index.to_numpy()[selected]

    if return_details:
        idx_name = df.index.name
        out = pd.DataFrame({
            "index": sel_idx_labels,
            "distance": sel_vals,
            "time_label": [_time_label(idx_name, v) for v in sel_idx_labels]
        })
        return out.reset_index(drop=True)
    else:
        return list(sel_idx_labels)

def plot_incidents_3d(df,
                      idx_list,
                      COLOR,
                      CONNECTIVITY,
                      plotter,
                      animal1="a1",
                      animal2="a2",
                      max_plots=None,
                      **plot_kwargs):
    """
    Calls your plotter (e.g., plot_skeleton_frame_3d_const_colors) on each index.
    Parameters:
      plotter: function(df, idx, COLOR, CONNECTIVITY, animal1=..., animal2=..., **kwargs)
    """
    if max_plots is not None:
        idx_list = list(idx_list)[:int(max_plots)]
    for idx in idx_list:
        plotter(df, idx, COLOR=COLOR, CONNECTIVITY=CONNECTIVITY,
                animal1=animal1, animal2=animal2, **plot_kwargs)


# ==============================================================================
# Cell 14
# ==============================================================================

def _time_label(idx_name, idx_value):
    if idx_name and ("ms" in str(idx_name).lower()):
        return f"{float(idx_value)/1000.0:.3f}s"
    try:
        return f"frame {int(idx_value)}"
    except Exception:
        return f"{idx_value}"

def _hide_3d_axes(ax):
    ax.set_axis_off()
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    for setter in [
        lambda: ax.xaxis.set_pane_color((1,1,1,0)),
        lambda: ax.yaxis.set_pane_color((1,1,1,0)),
        lambda: ax.zaxis.set_pane_color((1,1,1,0)),
        lambda: ax.xaxis._axinfo["grid"].update({"linewidth": 0}),
        lambda: ax.yaxis._axinfo["grid"].update({"linewidth": 0}),
        lambda: ax.zaxis._axinfo["grid"].update({"linewidth": 0}),
    ]:
        try: setter()
        except Exception: pass

def _robust_extent(P, pad=0.08, ignore_outlier_pct=99.5):
    """Cube-like extent using robust percentiles; returns (xlim, ylim, zlim)."""
    m = np.isfinite(P).all(axis=1)
    if not m.any():
        return (0,1),(0,1),(0,1)
    Q = P[m]
    lo = 100 - ignore_outlier_pct
    xlo, xhi = np.nanpercentile(Q[:,0], [lo, ignore_outlier_pct])
    ylo, yhi = np.nanpercentile(Q[:,1], [lo, ignore_outlier_pct])
    zlo, zhi = np.nanpercentile(Q[:,2], [lo, ignore_outlier_pct])
    cx, cy, cz = (xlo+xhi)/2, (ylo+yhi)/2, (zlo+zhi)/2
    span = max(xhi-xlo, yhi-ylo, zhi-zlo, 1e-6)
    r = span * (0.5 + pad)
    return (cx-r, cx+r), (cy-r, cy+r), (cz-r, cz+r)

def _global_extent(df, idx_list, animal1, animal2, pad=0.08, ignore_outlier_pct=99.5):
    xs, ys, zs = [], [], []
    for idx in idx_list:
        a1 = _kp_xyz_row(df, idx, animal=animal1)
        a2 = _kp_xyz_row(df, idx, animal=animal2)
        P = np.vstack([a1, a2])
        (xl, xh), (yl, yh), (zl, zh) = _robust_extent(P, pad=pad, ignore_outlier_pct=ignore_outlier_pct)
        xs += [xl, xh]; ys += [yl, yh]; zs += [zl, zh]
    if not xs:
        return (0,1),(0,1),(0,1)
    cx = (min(xs)+max(xs))/2; cy = (min(ys)+max(ys))/2; cz = (min(zs)+max(zs))/2
    span = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs), 1e-6)
    r = span/2
    return (cx-r, cx+r), (cy-r, cy+r), (cz-r, cz+r)

def plot_incidents_3d_grid(
    df,
    idx_list,
    COLOR,
    CONNECTIVITY,
    animal1="a1",
    animal2="a2",
    ncols=5,
    dpi=300,
    elev=22,
    azim=-55,
    invert_y=False,
    kp_size=14,
    lw=2.0,
    figsize=None,
    annotate=False,
    zoom_mode="local",         # "local" (per panel) or "global"
    zoom_pad=0.08,             # padding fraction around tight bbox
    ignore_outlier_pct=99.5,   # robust cropping; lower if you see flares
    a1_color="black",          # white on white looked invisible; change if you prefer
    a2_color="cyan",
    save_path=None             # set a path to write a tightly-cropped image
):
    """
    Render multiple frames as a clean grid; per-panel auto-zoom by default.
    """
    idx_list = list(idx_list)
    if not idx_list:
        raise ValueError("idx_list is empty.")

    n = len(idx_list)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    if figsize is None:
        figsize = (3.2*ncols, 3.2*nrows)

    # Precompute global extent if requested
    global_extent = None
    if zoom_mode == "global":
        global_extent = _global_extent(df, idx_list, animal1, animal2,
                                       pad=zoom_pad, ignore_outlier_pct=ignore_outlier_pct)

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")

    for k, idx in enumerate(idx_list):
        ax = fig.add_subplot(nrows, ncols, k+1, projection="3d")
        a1 = _kp_xyz_row(df, idx, animal=animal1)
        a2 = _kp_xyz_row(df, idx, animal=animal2)

        # Edges
        for color, (i, j) in zip(COLOR, CONNECTIVITY):
            if np.isfinite(a1[[i,j]]).all():
                ax.plot([a1[i,0], a1[j,0]], [a1[i,1], a1[j,1]], [a1[i,2], a1[j,2]], c=color, lw=lw)
            if np.isfinite(a2[[i,j]]).all():
                ax.plot([a2[i,0], a2[j,0]], [a2[i,1], a2[j,1]], [a2[i,2], a2[j,2]], c=color, lw=lw, alpha=0.9)

        # Points (use darker a1 so it doesn't disappear on white)
        ax.scatter(a1[:,0], a1[:,1], a1[:,2], s=kp_size, color=a1_color, depthshade=False, alpha=0.9)
        ax.scatter(a2[:,0], a2[:,1], a2[:,2], s=kp_size, color=a2_color, depthshade=False, alpha=0.9)

        # Limits
        if zoom_mode == "local":
            P = np.vstack([a1, a2])
            xlim, ylim, zlim = _robust_extent(P, pad=zoom_pad, ignore_outlier_pct=ignore_outlier_pct)
        else:
            xlim, ylim, zlim = global_extent

        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        if invert_y: ax.invert_yaxis()
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=elev, azim=azim)

        _hide_3d_axes(ax)

        if annotate:
            ax.text2D(0.03, 0.94, _time_label(df.index.name, idx),
                      transform=ax.transAxes, fontsize=8, ha='left', va='top')

    # Fill any empty slots (clean)
    total = nrows*ncols
    for k in range(n, total):
        ax = fig.add_subplot(nrows, ncols, k+1, projection="3d")
        ax.set_axis_off()

    plt.tight_layout(pad=0.05)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.show()


# ==============================================================================
# Cell 16
# ==============================================================================

def _time_label(idx_name, idx_value):
    if idx_name and ("ms" in str(idx_name).lower()):
        return f"{float(idx_value)/1000.0:.3f}s"
    try: return f"frame {int(idx_value)}"
    except Exception: return f"{idx_value}"

def _poster_axes_3d(ax, show_grid=True, grid_alpha=0.22, grid_lw=0.6):
    ax.grid(show_grid)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    # transparent panes (keep grid visible)
    for setter in [
        lambda: ax.xaxis.set_pane_color((1,1,1,0)),
        lambda: ax.yaxis.set_pane_color((1,1,1,0)),
        lambda: ax.zaxis.set_pane_color((1,1,1,0)),
    ]:
        try: setter()
        except Exception: pass
    # style the grid lines
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            a._axinfo["grid"].update({
                "linewidth": float(grid_lw),
                "linestyle": "-",
                "color": (0,0,0,float(grid_alpha)),
            })
        except Exception:
            pass
    # hide axis lines if present (older mpl)
    for a in ("w_xaxis","w_yaxis","w_zaxis"):
        try: getattr(ax, a).line.set_color((1,1,1,0))
        except Exception: pass

def _robust_extent(P, pad=0.08, ignore_outlier_pct=99.0):
    m = np.isfinite(P).all(axis=1)
    if not m.any(): return (0,1),(0,1),(0,1)
    Q = P[m]
    lo = 100 - ignore_outlier_pct
    xlo, xhi = np.nanpercentile(Q[:,0], [lo, ignore_outlier_pct])
    ylo, yhi = np.nanpercentile(Q[:,1], [lo, ignore_outlier_pct])
    zlo, zhi = np.nanpercentile(Q[:,2], [lo, ignore_outlier_pct])
    cx, cy, cz = (xlo+xhi)/2, (ylo+yhi)/2, (zlo+zhi)/2
    span = max(xhi-xlo, yhi-ylo, zhi-zlo, 1e-6)
    r = span * (0.5 + pad)
    return (cx-r, cx+r), (cy-r, cy+r), (cz-r, cz+r)

def plot_incidents_3d_grid(
    df,
    idx_list,
    COLOR,
    CONNECTIVITY,
    animal1="a1",
    animal2="a2",
    ncols=5,
    dpi=300,
    elev=22,
    azim=-55,
    invert_y=False,
    kp_size=14,
    lw=2.0,
    figsize=None,
    annotate=False,
    zoom_mode="local",        # "local" or "global"
    zoom_pad=0.10,
    ignore_outlier_pct=99.0,
    a1_color="black",
    a2_color="cyan",
    show_grid=True,          # <— keep the grid, hide axes
    grid_alpha=0.22,
    grid_lw=0.6,
    save_path=None
):
    idx_list = list(idx_list)
    if not idx_list: raise ValueError("idx_list is empty.")

    n = len(idx_list)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))
    if figsize is None: figsize = (3.2*ncols, 3.2*nrows)

    # precompute global extent if needed
    global_extent = None
    if zoom_mode == "global":
        xs=ys=zs=None
        # quick global from all frames via robust union of locals
        ext = []
        for idx in idx_list:
            P = np.vstack([_kp_xyz_row(df, idx, animal1), _kp_xyz_row(df, idx, animal2)])
            ext.append(_robust_extent(P, pad=zoom_pad, ignore_outlier_pct=ignore_outlier_pct))
        # merge cube
        xl = min(e[0][0] for e in ext); xh = max(e[0][1] for e in ext)
        yl = min(e[1][0] for e in ext); yh = max(e[1][1] for e in ext)
        zl = min(e[2][0] for e in ext); zh = max(e[2][1] for e in ext)
        cx,cy,cz = (xl+xh)/2,(yl+yh)/2,(zl+zh)/2
        span = max(xh-xl, yh-yl, zh-zl, 1e-6); r = span/2
        global_extent = ((cx-r,cx+r),(cy-r,cy+r),(cz-r,cz+r))

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")

    for k, idx in enumerate(idx_list):
        ax = fig.add_subplot(nrows, ncols, k+1, projection="3d")
        a1 = _kp_xyz_row(df, idx, animal=animal1)
        a2 = _kp_xyz_row(df, idx, animal=animal2)

        # edges
        for color, (i, j) in zip(COLOR, CONNECTIVITY):
            if np.isfinite(a1[[i,j]]).all():
                ax.plot([a1[i,0], a1[j,0]], [a1[i,1], a1[j,1]], [a1[i,2], a1[j,2]], c=color, lw=lw)
            if np.isfinite(a2[[i,j]]).all():
                ax.plot([a2[i,0], a2[j,0]], [a2[i,1], a2[j,1]], [a2[i,2], a2[j,2]], c=color, lw=lw, alpha=0.9)

        # points
        ax.scatter(a1[:,0], a1[:,1], a1[:,2], s=kp_size, color=a1_color, depthshade=False, alpha=0.9)
        ax.scatter(a2[:,0], a2[:,1], a2[:,2], s=kp_size, color=a2_color, depthshade=False, alpha=0.9)

        # limits + view
        if zoom_mode == "local":
            P = np.vstack([a1, a2])
            xlim, ylim, zlim = _robust_extent(P, pad=zoom_pad, ignore_outlier_pct=ignore_outlier_pct)
        else:
            xlim, ylim, zlim = global_extent
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        if invert_y: ax.invert_yaxis()
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=elev, azim=azim)

        # hide axes, keep grid
        _poster_axes_3d(ax, show_grid=show_grid, grid_alpha=grid_alpha, grid_lw=grid_lw)

        if annotate:
            ax.text2D(0.03, 0.94, _time_label(df.index.name, idx),
                      transform=ax.transAxes, fontsize=8, ha='left', va='top')

    # blank any unused cells
    total = nrows*ncols
    for k in range(n, total):
        ax = fig.add_subplot(nrows, ncols, k+1, projection="3d")
        _poster_axes_3d(ax, show_grid=False)

    plt.tight_layout(pad=0.05)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.show()


# ==============================================================================
# Cell 28
# ==============================================================================

def _poster_axes_3d(ax, show_grid=True, grid_alpha=0.22, grid_lw=0.6):
    ax.grid(show_grid)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(''); ax.set_ylabel(''); ax.set_zlabel('')
    for setter in [
        lambda: ax.xaxis.set_pane_color((1,1,1,0)),
        lambda: ax.yaxis.set_pane_color((1,1,1,0)),
        lambda: ax.zaxis.set_pane_color((1,1,1,0)),
    ]:
        try: setter()
        except Exception: pass
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            a._axinfo["grid"].update({"linewidth": float(grid_lw),
                                      "linestyle": "-",
                                      "color": (0,0,0,float(grid_alpha))})
        except Exception: pass

def _robust_extent(P, pad=0.10, ignore_outlier_pct=99.0):
    m = np.isfinite(P).all(axis=1)
    if not m.any(): return (0,1),(0,1),(0,1)
    Q = P[m]
    lo = 100 - ignore_outlier_pct
    xlo, xhi = np.nanpercentile(Q[:,0], [lo, ignore_outlier_pct])
    ylo, yhi = np.nanpercentile(Q[:,1], [lo, ignore_outlier_pct])
    zlo, zhi = np.nanpercentile(Q[:,2], [lo, ignore_outlier_pct])
    cx, cy, cz = (xlo+xhi)/2, (ylo+yhi)/2, (zlo+zhi)/2
    span = max(xhi-xlo, yhi-ylo, zhi-zlo, 1e-6)
    r = span * (0.5 + pad)
    return (cx-r, cx+r), (cy-r, cy+r), (cz-r, cz+r)

def _rotate_z(P, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=float)
    return P @ R.T

def _com(P):
    m = np.isfinite(P).all(axis=1)
    if not m.any(): return np.array([np.nan, np.nan, np.nan])
    return np.nanmean(P[m], axis=0)

def plot_incidents_3d_grid(
    df,
    idx_list,
    COLOR,                  # kept for call-compat; not used
    CONNECTIVITY,
    animal1="a1",
    animal2="a2",
    ncols=5,
    dpi=300,
    elev=22,
    azim=-55,
    invert_y=False,
    kp_size=14,
    lw=2.0,
    figsize=None,
    annotate=False,
    zoom_mode="local",        # "local" or "global"
    zoom_pad=0.10,
    ignore_outlier_pct=99.0,
    a1_color="#111111",       # solid charcoal (A1)
    a2_color="#1f77b4",       # mpl blue (A2)
    edge_alpha=0.9,
    point_alpha=0.9,
    show_grid=True,
    grid_alpha=0.22,
    grid_lw=0.6,
    save_path=None
):
    idx_list = list(idx_list)
    if not idx_list:
        raise ValueError("idx_list is empty.")

    n = len(idx_list)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (3.2*ncols, 3.2*nrows)

    # precompute global extent if needed
    global_extent = None
    if zoom_mode == "global":
        ext = []
        for idx in idx_list:
            P = np.vstack([_kp_xyz_row(df, idx, animal1), _kp_xyz_row(df, idx, animal2)])
            ext.append(_robust_extent(P, pad=zoom_pad, ignore_outlier_pct=ignore_outlier_pct))
        xl = min(e[0][0] for e in ext); xh = max(e[0][1] for e in ext)
        yl = min(e[1][0] for e in ext); yh = max(e[1][1] for e in ext)
        zl = min(e[2][0] for e in ext); zh = max(e[2][1] for e in ext)
        cx, cy, cz = (xl+xh)/2, (yl+yh)/2, (zl+zh)/2
        span = max(xh-xl, yh-yl, zh-zl, 1e-6); r = span/2
        global_extent = ((cx-r, cx+r), (cy-r, cy+r), (cz-r, cz+r))

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")

    for k, idx in enumerate(idx_list):
        ax = fig.add_subplot(nrows, ncols, k+1, projection="3d")
        a1 = _kp_xyz_row(df, idx, animal=animal1)
        a2 = _kp_xyz_row(df, idx, animal=animal2)

        # --- skeleton edges (single color per animal) ---
        for (i, j) in CONNECTIVITY:
            if np.isfinite(a1[[i, j]]).all():
                ax.plot([a1[i,0], a1[j,0]],
                        [a1[i,1], a1[j,1]],
                        [a1[i,2], a1[j,2]],
                        c=a1_color, lw=lw, alpha=edge_alpha)
            if np.isfinite(a2[[i, j]]).all():
                ax.plot([a2[i,0], a2[j,0]],
                        [a2[i,1], a2[j,1]],
                        [a2[i,2], a2[j,2]],
                        c=a2_color, lw=lw, alpha=edge_alpha)

        # --- joints/points (same single color) ---
        ax.scatter(a1[:,0], a1[:,1], a1[:,2], s=kp_size,
                   color=a1_color, depthshade=False, alpha=point_alpha)
        ax.scatter(a2[:,0], a2[:,1], a2[:,2], s=kp_size,
                   color=a2_color, depthshade=False, alpha=point_alpha)

        # limits + view
        if zoom_mode == "local":
            P = np.vstack([a1, a2])
            xlim, ylim, zlim = _robust_extent(P, pad=zoom_pad, ignore_outlier_pct=ignore_outlier_pct)
        else:
            xlim, ylim, zlim = global_extent
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        if invert_y: ax.invert_yaxis()
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)

        _poster_axes_3d(ax, show_grid=show_grid, grid_alpha=grid_alpha, grid_lw=grid_lw)

        if annotate:
            ax.text2D(0.03, 0.94, _time_label(df.index.name, idx),
                      transform=ax.transAxes, fontsize=8, ha='left', va='top')

    # blank any unused cells
    total = nrows * ncols
    for k in range(n, total):
        ax = fig.add_subplot(nrows, ncols, k+1, projection="3d")
        _poster_axes_3d(ax, show_grid=False)

    plt.tight_layout(pad=0.05)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.show()

# ==============================================================================
# Cell 31
# ==============================================================================

def _boolean_runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if mask.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    m = mask.astype(np.int8)
    edges = np.diff(np.pad(m, (1, 1), constant_values=0))
    starts = np.flatnonzero(edges == 1)
    ends   = np.flatnonzero(edges == -1)
    return starts, ends

def find_approach_success(
    frames: pd.DataFrame,
    contact_mm: float = 50.0,     # "touch" threshold
    dD_dt_thresh: float = 0.0,    # <= 0 means distance not increasing
    min_len: int = 10,            # min frames from start of closing to contact
    min_drop_mm: float = 10.0     # require at least this much net distance drop
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Returns:
      mask  : boolean array marking frames that belong to approach_success events
      events: list of dicts with indices and metrics for each event

    Uses only frames['dist_mm'] and frames['dD_dt'].
    """
    dist = frames["dist_mm"].to_numpy()
    closing = frames["dD_dt"].to_numpy() <= float(dD_dt_thresh)
    contact = dist <= float(contact_mm)

    s_close, e_close = _boolean_runs(closing)

    mask = np.zeros(len(frames), dtype=bool)
    events: List[Dict] = []

    for s, e in zip(s_close, e_close):
        # first contact inside this closing run (if any)
        within = np.flatnonzero(contact[s:e])
        if within.size == 0:
            continue
        k = s + int(within[0])  # first contact index

        # length and net drop checks
        length_ok = (k - s + 1) >= int(min_len)
        drop_ok   = (dist[s] - dist[k]) >= float(min_drop_mm)
        if not (length_ok and drop_ok):
            continue

        mask[s:k+1] = True
        events.append({
            "start_idx": int(s),
            "end_idx_exclusive": int(k + 1),
            "contact_idx": int(k),
            "start_dist_mm": float(dist[s]),
            "end_dist_mm": float(dist[k]),
            "drop_mm": float(dist[s] - dist[k]),
            "duration_frames": int(k - s + 1),
        })

    return mask, events


# ==============================================================================
# Cell 33
# ==============================================================================

def visualize_frames_allcams(frame_list: List[int],
                             config: Optional['VizConfig'] = None,
                             cam_indices: Optional[Sequence[Any]] = None,
                             ncols: int = 3,
                             **overrides: Any) -> None:
    """
    Render a mosaic (e.g., 2x3) video of ALL requested cameras for the given absolute frames.
    - Uses camera KEYS (e.g., 'Camera1') for calibration/pred indexing.
    - Uses numeric ids only when building per-cam video paths via _prepare_io(cammm=...).
    - Prevents _draw_frame's plt.clf() from wiping the whole mosaic.
    """

    if not frame_list:
        raise ValueError("frame_list is empty.")
    frames = list(frame_list)

    # ---------- helpers ----------
    def camkey_to_int(k: Any) -> int:
        m = re.search(r'(\d+)', str(k))
        return int(m.group(1)) if m else int(str(k))

    @contextmanager
    def suppress_clf():
        """Temporarily make plt.clf() a no-op so _draw_frame won't clear the mosaic."""
        orig = plt.clf
        plt.clf = lambda *a, **k: None
        try:
            yield
        finally:
            plt.clf = orig

    # ---------- config / shared paths ----------
    c0 = _apply_overrides(config or DEFAULT_CFG, overrides)
    _cam0, _vp0, label3d_path, pred_path, com_path, save_path = _prepare_io(c0)
    os.makedirs(save_path, exist_ok=True)

    # ---------- camera inventory from calibration ----------
    cameras = load_cameras(label3d_path)  # dict like {'Camera1': {...}, 'Camera2': {...}, ...}
    # Build sorted list of (cam_key, cam_int)
    cam_items: List[Tuple[Any, int]] = [(ckey, camkey_to_int(ckey)) for ckey in cameras.keys()]
    cam_items.sort(key=lambda x: x[1])  # sort by numeric id

    # Resolve which cameras to render
    if cam_indices is None:
        cam_keys = [ckey for ckey, _ in cam_items]
    else:
        # Accept ints ('1'), 'Camera1', etc.; normalize to existing keys
        int_to_key = {cint: ckey for ckey, cint in cam_items}
        key_set = set(k for k, _ in cam_items)
        resolved = []
        for c in cam_indices:
            if c in key_set:
                resolved.append(c)
            else:
                cint = camkey_to_int(c)
                if cint in int_to_key:
                    resolved.append(int_to_key[cint])
        # de-dup while preserving order
        seen = set()
        cam_keys = []
        for k in resolved:
            if k not in seen:
                seen.add(k)
                cam_keys.append(k)

    n = len(cam_keys)
    if n == 0:
        raise ValueError("No matching cameras resolved from cam_indices against calibration keys.")

    # ---------- per-cam video paths (keyed by cam_key) ----------
    video_paths: Dict[Any, str] = {}
    for cam_key in cam_keys:
        cam_int = camkey_to_int(cam_key)
        ci = _apply_overrides(c0, dict(cammm=cam_int))
        _c, vp, *_ = _prepare_io(ci)
        video_paths[cam_key] = vp

    # ---------- sparse preds / COM per cam (index using cam_key) ----------
    def _pick_cam_slice(x, cam_key):
        try:
            return x[cam_key]  # dict-of-cam_key
        except Exception:
            return x          # already a list/array for this cam

    pred_2d_all: Dict[Any, Sequence] = {}
    pred_2d_com_all: Dict[Any, Sequence] = {}
    for cam_key in cam_keys:
        p = _load_and_project_preds_frames(pred_path, cameras, cam_key, frames)
        pc = _load_and_project_com_frames(com_path, cameras, cam_key, frames)
        pred_2d_all[cam_key] = _pick_cam_slice(p, cam_key)
        pred_2d_com_all[cam_key] = _pick_cam_slice(pc, cam_key)

    # ---------- readers (keyed by cam_key) ----------
    readers: Dict[Any, Any] = {}
    for cam_key in cam_keys:
        readers[cam_key] = imageio.get_reader(video_paths[cam_key])

    # ---------- style ----------
    COLOR = connectivity.COLOR_DICT[c0.animal]
    CONNECTIVITY = connectivity.CONNECTIVITY_DICT[c0.animal]

    # ---------- figure grid ----------
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis('off')
    # hide extra axes
    for ax in axes[n:]:
        ax.set_visible(False)
    # tight packing
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02, hspace=0.02)

    # ---------- output ----------
    tag = f"allcams_{frames[0]}_{frames[-1]}_{len(frames)}f"
    out_base = (getattr(c0, 'out_name', None) or f"combined_allcams_{tag}") + ".mp4"
    out_mp4 = os.path.join(save_path, "vis_" + out_base)

    writer = FFMpegWriter(fps=c0.video_fps,
                          metadata=dict(title='combined_visualization_allcams', artist='Matplotlib'))

    # ---------- render ----------
    with writer.saving(fig, out_mp4, dpi=250):
        for j, f_abs in enumerate(tqdm.tqdm(frames)):
            for idx, cam_key in enumerate(cam_keys[:len(axes)]):
                ax = axes[idx]
                ax.cla()
                ax.axis('off')

                # image
                try:
                    img = readers[cam_key].get_data(f_abs)
                except Exception:
                    ax.text(0.5, 0.5, f"{cam_key} frame {f_abs}\n(missing)", ha='center', va='center', fontsize=10)
                    continue

                # preds
                k = pred_2d_all[cam_key][j]
                if getattr(k, "ndim", None) == 2:
                    k_a1, k_a2 = k, None
                else:
                    k_a1, k_a2 = k[0], k[1]
                k_com = pred_2d_com_all[cam_key][j]

                # draw into this axes
                plt.sca(ax)
                with suppress_clf():
                    _draw_frame(
                        img, k_a1, k_a2, k_com,
                        COLOR, CONNECTIVITY,
                        enable_zoom=c0.enable_zoom,
                        zoom_margin=c0.zoom_margin,
                        drop_tail_for_view=c0.drop_tail_for_view,
                        include_both=c0.zoom_include_both,
                        title_text=None
                    )

                # robust in-panel label
                ax.text(0.01, 0.03, f"{cam_key}  f={f_abs}", transform=ax.transAxes,
                        fontsize=8, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6))

            writer.grab_frame()
            if getattr(c0, 'write_pngs', False):
                plt.savefig(os.path.join(save_path, f"vis_allcams_frame_{f_abs}.png"),
                            dpi=250, bbox_inches="tight")

    # ---------- cleanup ----------
    for r in readers.values():
        try:
            r.close()
        except Exception:
            pass


# ==============================================================================
# Cell 40
# ==============================================================================

def plot_ca_heatmap_and_distance(
    merged: pd.DataFrame,
    *,
    rec_path: str = None,                   # used only for filename if you decide to save
    exclude_dict: dict = None,              # optional: {session_substring: [roi_indices_to_exclude]}
    exclude_key: str = None,                # optional: explicit key for exclude_dict; else rec_path is used
    shift_time0: bool = True,               # shift time so first shown sample is t=0
    variance_drop_pct: float = 5.0,         # drop lowest N% variance neurons before clustering
    downsample_heatmap: int = 1,            # time-axis decimation for the heatmap only (>=1)
    cmap: str = "RdBu_r",                   # diverging map for z-scored ΔF/F
    distance_smooth_window: int = 1,        # passed to your compute_com_distance(...)
    show_colorbar: bool = False,
    title_prefix: str = None,               # e.g., session name for figure title
    save: bool = False,
    save_dir: str = None,                   # default: f"{rec_path}/MIR_Aligned"
    filename: str = None                    # default: "ca_and_distance.png"
):
    """
    Render a 2-panel figure:
      (A) Clustered ΔF/F heatmap (neurons x time, z-scored, Ward linkage)
      (B) COM distance trace (mm) from compute_com_distance(merged, 'com1', 'com2', ...)

    Assumptions:
      - 'merged' already aligned (rows are synchronized samples)
      - dF/F columns named 'dF_F_roi{int}'
      - time from 'timestamp_ms_mini' col OR index (ms) → seconds
      - distance units are mm
    """
    # ---------- Time vector (seconds) ----------
    if "timestamp_ms_mini" in merged.columns:
        t = merged["timestamp_ms_mini"].to_numpy(dtype=float) / 1000.0
        xlab = "Time (s)"
    elif merged.index.name and ("ms" in str(merged.index.name).lower()):
        t = merged.index.to_numpy(dtype=float) / 1000.0
        xlab = "Time (s)"
    elif "timestamp_ms" in merged.columns:
        t = merged["timestamp_ms"].to_numpy(dtype=float) / 1000.0
        xlab = "Time (s)"
    elif "time_s" in merged.columns:
        t = merged["time_s"].to_numpy(dtype=float)
        xlab = "Time (s)"
    else:
        t = np.arange(len(merged), dtype=float)
        xlab = "Frame"

    if shift_time0 and len(t) > 0:
        t = t - np.nanmin(t)

    # ---------- Distance trace via your helper ----------
    try:
        dist = compute_com_distance(merged, p1="com1", p2="com2", smooth_window=distance_smooth_window)
        dist = np.asarray(dist).reshape(-1)
        # Detect dimensionality from COM columns for label
        has_z = {"com1_z", "com2_z"}.issubset(set(merged.columns))
        dist_title = f"COM1–COM2 distance ({'3D' if has_z else '2D'})"
    except Exception as e:
        dist = None
        dist_title = f"COM1–COM2 distance (unavailable: {e})"

    # ---------- Build ΔF/F matrix (neurons x time) ----------
    roi_cols = [c for c in merged.columns if c.startswith("dF_F_roi")]
    # Apply exclusions (optional)
    if exclude_dict is not None:
        key = exclude_key or rec_path or ""
        excluded = []
        if key in exclude_dict:
            excluded = exclude_dict[key]
        else:
            # substring fallback
            for k, v in exclude_dict.items():
                if k and isinstance(k, str) and k in key:
                    excluded = v
                    break
        exclude_set = {f"dF_F_roi{i}" for i in excluded}
        roi_cols = [c for c in roi_cols if c not in exclude_set]

    # Form matrix
    if len(roi_cols) > 0:
        A = merged[roi_cols].to_numpy(dtype=float, copy=False).T  # (n_neurons x T)
        # Variance filter (drop lowest N%)
        with np.errstate(invalid="ignore"):
            var = np.nanvar(A, axis=1)
        var = np.where(np.isnan(var), 0.0, var)
        if variance_drop_pct and 0 < variance_drop_pct < 100 and A.shape[0] > 1:
            try:
                thresh = np.nanpercentile(var, variance_drop_pct)
                keep = var > thresh
                if not np.any(keep):
                    keep = np.ones_like(var, dtype=bool)
                A = A[keep, :]
                roi_cols = [c for i, c in enumerate(roi_cols) if keep[i]]
            except Exception:
                pass

        # Z-score rows (neurons)
        with np.errstate(invalid="ignore"):
            A = zscore(A, axis=1, nan_policy="omit")
        if np.isnan(A).any():
            A = np.nan_to_num(A, nan=0.0)
    else:
        A = np.empty((0, len(merged)), dtype=float)

    # Optional downsample for heatmap ONLY (time axis)
    step = max(1, int(downsample_heatmap))
    if step > 1 and A.size > 0:
        A_plot = A[:, ::step]
        t_plot = t[::step]
    else:
        A_plot = A
        t_plot = t

    # Cluster neuron order (on the plotted matrix to keep cost bounded)
    if A_plot.shape[0] >= 2 and A_plot.shape[1] >= 2:
        try:
            Z = linkage(A_plot, method="ward")
            order = leaves_list(Z)
            A_plot = A_plot[order, :]
        except Exception:
            # Fallback: order by variance (descending) on the plotted matrix
            v = np.nanvar(A_plot, axis=1)
            order = np.argsort(-np.where(np.isnan(v), -np.inf, v))
            A_plot = A_plot[order, :]
    # else: leave as-is (0 or 1 neuron)

    # ---------- Figure ----------
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2.2, 1.0], hspace=0.08)

    # (A) Heatmap
    ax0 = fig.add_subplot(gs[0, 0])
    if A_plot.size > 0 and A_plot.shape[1] == t_plot.shape[0]:
        hm = ax0.pcolormesh(t_plot, np.arange(A_plot.shape[0] + 1),  # +1 for pcolormesh edges
                            np.vstack([A_plot, A_plot[-1:]]),         # cheap edge pad for shading='auto'
                            cmap=cmap, shading="auto")
        ax0.set_title("Clustered Neuron Activity (z-scored)")
        ax0.tick_params(labelbottom=False)
        ax0.set_ylabel("Neurons (clustered)")
        if show_colorbar:
            cbar = plt.colorbar(hm, ax=ax0, fraction=0.02, pad=0.02)
            cbar.set_label("z-score")
    else:
        ax0.text(0.5, 0.5, "No neuron data", ha="center", va="center", transform=ax0.transAxes)
        # ax0.set_title("No Neuron Activity")
        ax0.tick_params(labelbottom=False)

    # (B) Distance trace
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    if dist is not None and len(dist) == len(t):
        ax1.plot(t, dist, lw=1.2, color="0.15")
    else:
        ax1.text(0.5, 0.5, "Distance unavailable", ha="center", va="center", transform=ax1.transAxes)
    ax1.set_xlabel(xlab)
    ax1.set_ylabel("Distance (mm)")
    ax1.set_title(dist_title)

    if title_prefix:
        fig.suptitle(title_prefix, y=0.995, fontsize=12)

    plt.tight_layout()

    # ---------- Save (optional) ----------
    if save:
        out_dir = save_dir
        if out_dir is None:
            out_dir = (f"{rec_path}/MIR_Aligned") if rec_path else "."
        os.makedirs(out_dir, exist_ok=True)
        if filename is None:
            base = "ca_and_distance"
            if title_prefix:
                base = f"{base}__{title_prefix}"
            filename = f"{base}.png"
        out_path = os.path.join(out_dir, filename)
        fig.savefig(out_path, dpi=300)
        print(f"[OK] Saved: {out_path}")

    plt.show()
    plt.close(fig)


# ==============================================================================
# Cell 42
# ==============================================================================

def _segments_from_mask(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return np.empty((0, 2), dtype=int)
    # rising/falling edges
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(d == 1)
    ends   = np.flatnonzero(d == -1)    # end index is exclusive
    return np.stack([starts, ends], axis=1)  # shape (N,2), [start, end)

def plot_ca_heatmap_and_distance(
    merged: pd.DataFrame,
    *,
    rec_path: str = None,
    exclude_dict: dict = None,
    exclude_key: str = None,
    shift_time0: bool = True,
    variance_drop_pct: float = 5.0,
    downsample_heatmap: int = 1,
    cmap: str = "RdBu_r",
    distance_smooth_window: int = 1,
    show_colorbar: bool = False,
    title_prefix: str = None,
    save: bool = False,
    save_dir: str = None,
    filename: str = None,
    # ---- new options for meetings ----
    meeting_threshold_mm: float = 100.0,
    meeting_min_frames: int = 1,          # drop blips shorter than this
    mark_color: str = "crimson",
    mark_alpha: float = 0.18,
):
    """
    Two panels:
      (A) Clustered ΔF/F heatmap (z-scored)
      (B) COM distance trace with vertical red spans where distance < threshold
    """
    # ----- time base -----
    if "timestamp_ms_mini" in merged.columns:
        t = merged["timestamp_ms_mini"].to_numpy(dtype=float) / 1000.0
        xlab = "Time (s)"
    elif "timestamp_ms" in merged.columns:
        t = merged["timestamp_ms"].to_numpy(dtype=float) / 1000.0
        xlab = "Time (s)"
    elif "time_s" in merged.columns:
        t = merged["time_s"].to_numpy(dtype=float)
        xlab = "Time (s)"
    else:
        t = np.arange(len(merged), dtype=float)
        xlab = "Frame"
    if shift_time0 and len(t) > 0:
        t = t - np.nanmin(t)

    # ----- distance (expects your helper present) -----
    try:
        dist = compute_com_distance(merged, p1="com1", p2="com2",
                                    smooth_window=distance_smooth_window)
        dist = np.asarray(dist, dtype=float).reshape(-1)
        has_z = {"com1_z", "com2_z"}.issubset(set(merged.columns))
        dist_title = f"Inter-animal distance ({'3D' if has_z else '2D'})"
    except Exception as e:
        dist = None
        dist_title = f"Inter-animal distance (unavailable: {e})"

    # ----- ΔF/F matrix -----
    roi_cols = [c for c in merged.columns if c.startswith("dF_F_roi")]
    if exclude_dict:
        key = (exclude_key or rec_path or "")
        exc = []
        if key in exclude_dict:
            exc = exclude_dict[key]
        else:
            for k, v in exclude_dict.items():
                if isinstance(k, str) and k and k in key:
                    exc = v
                    break
        bad = {f"dF_F_roi{i}" for i in exc}
        roi_cols = [c for c in roi_cols if c not in bad]

    if roi_cols:
        A = merged[roi_cols].to_numpy(dtype=float, copy=False).T  # (N,T)
        with np.errstate(invalid="ignore"):
            var = np.nanvar(A, axis=1)
        var = np.where(np.isnan(var), 0.0, var)
        if 0 < variance_drop_pct < 100 and A.shape[0] > 1:
            thr = np.nanpercentile(var, variance_drop_pct)
            keep = np.where(var > thr)[0]
            if keep.size:
                A = A[keep]
        with np.errstate(invalid="ignore"):
            A = zscore(A, axis=1, nan_policy="omit")
        A = np.nan_to_num(A, nan=0.0)
    else:
        A = np.empty((0, len(merged)), float)

    # optional decimation for heatmap only
    step = max(1, int(downsample_heatmap))
    A_plot = A[:, ::step] if (A.size and step > 1) else A
    t_plot = t[::step] if step > 1 else t

    # cluster rows on plotted matrix
    if A_plot.shape[0] >= 2 and A_plot.shape[1] >= 2:
        try:
            Z = linkage(A_plot, method="ward")
            order = leaves_list(Z)
        except Exception:
            order = np.argsort(-np.nanvar(A_plot, axis=1))
        A_plot = A_plot[order]

    # ----- meeting segments from threshold -----
    segs = np.empty((0, 2), int)
    if dist is not None and len(dist) == len(t):
        meet_mask = dist < float(meeting_threshold_mm)
        segs = _segments_from_mask(meet_mask)
        if meeting_min_frames > 1 and segs.size:
            keep = (segs[:, 1] - segs[:, 0]) >= int(meeting_min_frames)
            segs = segs[keep]

    # ----- fig -----
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.08)

    # (A) heatmap
    ax0 = fig.add_subplot(gs[0, 0])
    if A_plot.size and A_plot.shape[1] == t_plot.shape[0]:
        hm = ax0.pcolormesh(
            t_plot, np.arange(A_plot.shape[0] + 1),
            np.vstack([A_plot, A_plot[-1:]]),
            cmap=cmap, shading="auto"
        )
        ax0.set_title("Clustered Neuron Activity (Z-scored)")
        ax0.set_ylabel("Neurons (clustered)")
        ax0.tick_params(labelbottom=False)
        if show_colorbar:
            cbar = plt.colorbar(hm, ax=ax0, fraction=0.02, pad=0.02)
            cbar.set_label("z")
    else:
        ax0.text(0.5, 0.5, "No neuron data", ha="center", va="center",
                 transform=ax0.transAxes)
        ax0.tick_params(labelbottom=False)

    # shade meetings on heatmap
    if segs.size:
        for s, e in segs:
            ax0.axvspan(t[s], t[min(e-1, len(t)-1)], color=mark_color, alpha=mark_alpha, lw=0)

    # (B) distance
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    if dist is not None and len(dist) == len(t):
        ax1.plot(t, dist, lw=1.2, color="0.15", label="COM distance")
        # dashed threshold line
        ax1.axhline(meeting_threshold_mm, ls="--", lw=1.0, color="0.25")
        ax1.text(t[0] if len(t) else 0, meeting_threshold_mm,
                 f"  Threshold = {meeting_threshold_mm:.3f} mm",
                 va="bottom", ha="left", fontsize=9, color="0.25")
        # shade meetings on distance panel
        for s, e in segs:
            ax1.axvspan(t[s], t[min(e-1, len(t)-1)], color=mark_color, alpha=mark_alpha, lw=0)
    else:
        ax1.text(0.5, 0.5, "Distance unavailable", ha="center", va="center",
                 transform=ax1.transAxes)
    ax1.set_xlabel(xlab)
    ax1.set_ylabel("Distance (mm)")
    ax1.set_title(dist_title)

    if title_prefix:
        fig.suptitle(title_prefix, y=0.995, fontsize=12)

    plt.tight_layout()

    if save:
        out_dir = save_dir or ((f"{rec_path}/MIR_Aligned") if rec_path else ".")
        os.makedirs(out_dir, exist_ok=True)
        if filename is None:
            base = "ca_and_distance"
            if title_prefix:
                base = f"{base}__{title_prefix}"
            filename = f"{base}.png"
        fig.savefig(os.path.join(out_dir, filename), dpi=300)

    plt.show()
    plt.close(fig)


# ==============================================================================
# Cell 45
# ==============================================================================

def _time_from_index_or_col(df: pd.DataFrame):
    """
    Return time array (seconds) and x-label.
    Prefers 'timestamp_ms_mini' column; else inspects index name for 'ms';
    else falls back to frames.
    """
    if "timestamp_ms_mini" in df.columns:
        t = df["timestamp_ms_mini"].to_numpy(dtype=float) / 1000.0
        return t, "Time (s)"
    idx = df.index
    if getattr(idx, "name", None) and ("ms" in str(idx.name).lower()):
        return idx.to_numpy(dtype=float) / 1000.0, "Time (s)"
    if "timestamp_ms" in df.columns:
        return df["timestamp_ms"].to_numpy(dtype=float) / 1000.0, "Time (s)"
    if "time_s" in df.columns:
        return df["time_s"].to_numpy(dtype=float), "Time (s)"
    return np.arange(len(df), dtype=float), "Frame"

def _segments_from_mask(mask_1d_bool: Union[pd.Series, np.ndarray]):
    """Return Nx2 array of [start,end] (inclusive) segments where mask is True."""
    m = np.asarray(mask_1d_bool, dtype=bool)
    if m.size == 0:
        return np.empty((0, 2), dtype=int)
    starts = np.flatnonzero(~np.concatenate(([False], m[:-1])) & m)
    ends   = np.flatnonzero(m & ~np.concatenate((m[1:], [False])))
    if starts.size == 0:
        return np.empty((0, 2), dtype=int)
    return np.column_stack((starts, ends))

def _map_frame_segments_to_time(frames: pd.DataFrame, segs_fe: np.ndarray, col="timestamp_ms_mini"):
    """
    Map frame-based segments (row indices in 'frames') to time ranges (ms).
    Returns list of (t_start_ms, t_end_ms) per segment.
    """
    if col not in frames.columns:
        if getattr(frames.index, "name", None) and ("ms" in str(frames.index.name).lower()):
            t_ms = frames.index.to_numpy(dtype=float)
        else:
            t_ms = frames.index.to_numpy(dtype=float)  # fallback: treat index as time-like
    else:
        t_ms = frames[col].to_numpy(dtype=float)
    out = []
    for s, e in segs_fe:
        ts = float(t_ms[s])
        te = float(t_ms[e])
        if te < ts:
            ts, te = te, ts
        out.append((ts, te))
    return out

def _search_window_indices(t_vec_s: np.ndarray, t0_s: float, pre_s: float, post_s: float):
    """Return (i0, i1) such that t in [t0_s - pre_s, t0_s + post_s]."""
    t_start = t0_s - pre_s
    t_end   = t0_s + post_s
    i0 = int(np.searchsorted(t_vec_s, t_start, side="left"))
    i1 = int(np.searchsorted(t_vec_s, t_end, side="right")) - 1
    i0 = max(0, min(i0, len(t_vec_s)-1))
    i1 = max(0, min(i1, len(t_vec_s)-1))
    if i1 < i0:
        i0, i1 = i1, i0
    return i0, i1

def _snout_distance(merged: pd.DataFrame, smooth_window: int = 1):
    """
    Compute snout–snout distance over the full session.
      - Prefer 3D (kp3_z_* present); else 2D fallback.
    Returns (dist_vec, is_3d: bool). If missing, returns (None, False).
    """
    have_3d = all(c in merged.columns for c in [
        "kp3_x_a1","kp3_y_a1","kp3_z_a1","kp3_x_a2","kp3_y_a2","kp3_z_a2"
    ])
    have_2d = all(c in merged.columns for c in [
        "kp3_x_a1","kp3_y_a1","kp3_x_a2","kp3_y_a2"
    ])
    if not have_3d and not have_2d:
        return None, False

    if smooth_window and smooth_window > 1:
        roll = merged.rolling(int(smooth_window), center=True, min_periods=1)
        get = lambda c: roll[c].mean().to_numpy(dtype=float)
    else:
        get = lambda c: merged[c].to_numpy(dtype=float)

    x1 = get("kp3_x_a1"); y1 = get("kp3_y_a1")
    x2 = get("kp3_x_a2"); y2 = get("kp3_y_a2")

    if have_3d:
        z1 = get("kp3_z_a1"); z2 = get("kp3_z_a2")
        d = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        return d, True
    else:
        d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        return d, False

def _prepare_neuron_matrix(merged: pd.DataFrame, variance_drop_pct: float = 5.0):
    """
    Returns (A_z, order_idx, roi_cols) where:
      - A_z: (n_neurons x T) z-scored DF/F
      - order_idx: Ward linkage order (stable across windows)
      - roi_cols: names kept
    """
    roi_cols = [c for c in merged.columns if c.startswith("dF_F_roi")]
    if len(roi_cols) == 0:
        return np.empty((0, len(merged))), np.arange(0), []

    A = merged[roi_cols].to_numpy(dtype=float, copy=False).T  # (N x T)

    # variance filter
    with np.errstate(invalid="ignore"):
        var = np.nanvar(A, axis=1)
    var = np.where(np.isnan(var), 0.0, var)
    if variance_drop_pct and 0 < variance_drop_pct < 100 and A.shape[0] > 2:
        try:
            thr = np.nanpercentile(var, variance_drop_pct)
            keep = var > thr
            if keep.any():
                A = A[keep, :]
                roi_cols = [c for i, c in enumerate(roi_cols) if keep[i]]
        except Exception:
            pass

    # z-score neurons (rows)
    with np.errstate(invalid="ignore"):
        A = zscore(A, axis=1, nan_policy="omit")
    if np.isnan(A).any():
        A = np.nan_to_num(A, nan=0.0)

    # cluster once
    if A.shape[0] >= 2 and A.shape[1] >= 2:
        try:
            Z = linkage(A, method="ward")
            order = leaves_list(Z)
        except Exception:
            vv = np.nanvar(A, axis=1)
            order = np.argsort(-np.where(np.isnan(vv), -np.inf, vv))
    else:
        order = np.arange(A.shape[0])

    return A, order, roi_cols

def plot_incident_windows_3d(
    merged: pd.DataFrame,
    frames: pd.DataFrame,
    mask: Union[pd.Series, np.ndarray],
    *,
    pre_s: float = 2.0,
    post_s: float = 2.0,
    approach_thresh_mm: float = 120.0,
    contact_thresh_mm: float = 50.0,
    snout_contact_mm: float = 30.0,
    smooth_pos_window: int = 1,     # for compute_com_distance (positions)
    smooth_dist_window: int = 1,    # for compute_com_distance (distance)
    snout_smooth_window: int = 1,   # snout smoothing
    heatmap_downsample: int = 1,    # time decimation for heatmap only
    heatmap_cmap: str = "viridis",  # minimal, clean
    heatmap_clim: Optional[Tuple[float, float]] = None,  # None → auto scaling
    variance_drop_pct: float = 5.0, # match your earlier logic
    title_prefix: Optional[str] = None,
    save: bool = False,
    out_dir: Optional[str] = None
):
    """
    Minimal, clean, windowed multi-panel per incident:
      A) ΔF/F heatmap (session-level z-score + order, no colorbar)
      B) COM distance (3D solid; 2D dashed)
      C) Snout–snout distance (3D if available, else 2D)
      D) 3D−2D COM distance (vertical separation)
    Relies on your existing `compute_com_distance(...)`.
    """
    # --- time vectors ---
    t_merged_s, _ = _time_from_index_or_col(merged)
    t_frames_s, _ = _time_from_index_or_col(frames)
    if t_merged_s.size == 0 or t_frames_s.size == 0:
        print("[WARN] Empty time vectors; aborting.")
        return

    # --- full-session COM distances & components ---
    try:
        comp = compute_com_distance(
            merged, p1="com1", p2="com2",
            smooth_window=smooth_pos_window,
            dist_smooth_window=smooth_dist_window,
            return_components=True
        )
    except NameError:
        raise RuntimeError("`compute_com_distance` is not defined. Please ensure your earlier helper is imported.")

    d3d = comp["dist_mm"].to_numpy(dtype=float)
    dxy = np.hypot(comp["dx"].to_numpy(dtype=float), comp["dy"].to_numpy(dtype=float))
    gap = d3d - dxy  # ≥ 0 when there is vertical separation

    # --- snout distance across session ---
    snout_d, snout_is_3d = _snout_distance(merged, smooth_window=snout_smooth_window)
    if snout_d is None:
        snout_d = np.full_like(d3d, np.nan)
        snout_is_3d = False

    # --- neuron matrix + stable order once for the whole session ---
    A_full, order, _ = _prepare_neuron_matrix(merged, variance_drop_pct=variance_drop_pct)
    have_neurons = (A_full.size > 0)

    # --- segment extraction from mask (frames) and mapping to miniscope times ---
    segs_fe = _segments_from_mask(mask)
    if segs_fe.size == 0:
        print("[INFO] No True segments in mask; nothing to plot.")
        return
    seg_time_ms = _map_frame_segments_to_time(frames, segs_fe, col="timestamp_ms_mini")

    t_merged_ms = t_merged_s * 1000.0

    # --- iterate over events ---
    for k, (t0_ms, t1_ms) in enumerate(seg_time_ms, start=1):
        if t1_ms < t0_ms:
            t0_ms, t1_ms = t1_ms, t0_ms

        i0 = int(np.searchsorted(t_merged_ms, t0_ms, side="left"))
        i1 = int(np.searchsorted(t_merged_ms, t1_ms, side="right")) - 1
        i0 = max(0, min(i0, len(t_merged_s)-1))
        i1 = max(0, min(i1, len(t_merged_s)-1))
        if i1 <= i0:
            continue

        # rep index = argmin D3D within the segment
        seg_slice = slice(i0, i1+1)
        seg_d3d = d3d[seg_slice]
        if not np.any(np.isfinite(seg_d3d)):
            continue
        rep_rel = int(np.nanargmin(seg_d3d))
        rep_idx = i0 + rep_rel
        rep_t = float(t_merged_s[rep_idx])

        # window bounds in merged time
        j0, j1 = _search_window_indices(t_merged_s, rep_t, pre_s, post_s)
        w = slice(j0, j1+1)

        # local time axis (rep at 0)
        tloc = t_merged_s[w] - rep_t
        d3d_loc = d3d[w]
        dxy_loc = dxy[w]
        gap_loc = gap[w]
        snout_loc = snout_d[w]

        # heatmap slice (downsample for plotting only)
        if have_neurons:
            A_loc = A_full[order, w]
            if heatmap_downsample and heatmap_downsample > 1 and A_loc.shape[1] > 1:
                step = int(heatmap_downsample)
                A_plot = A_loc[:, ::step]
                t_plot = tloc[::step]
            else:
                A_plot = A_loc
                t_plot = tloc
        else:
            A_plot = np.empty((0, len(tloc)))
            t_plot = tloc

        # ---- figure (minimal style) ----
        fig = plt.figure(figsize=(12, 7.5))
        gs = fig.add_gridspec(nrows=4, ncols=1, height_ratios=[2.0, 1.0, 1.0, 1.0], hspace=0.08)

        def _lighten(ax):
            ax.grid(False)
            for s in ax.spines.values():
                s.set_linewidth(0.6)
                s.set_color("0.5")

        # A) heatmap (no colorbar)
        ax0 = fig.add_subplot(gs[0, 0])
        if A_plot.size > 0 and A_plot.shape[1] == t_plot.shape[0]:
            hm = ax0.pcolormesh(
                t_plot,
                np.arange(A_plot.shape[0] + 1),
                np.vstack([A_plot, A_plot[-1:]]) if A_plot.shape[0] > 0 else np.zeros((1, A_plot.shape[1])),
                cmap=heatmap_cmap,
                shading="auto"
            )
            if heatmap_clim is not None:
                hm.set_clim(*heatmap_clim)
            ax0.set_title("Clustered Neuron Activity")
            ax0.set_ylabel("Neurons")
            ax0.tick_params(labelbottom=False)
        else:
            ax0.text(0.5, 0.5, "No neuron data", ha="center", va="center", transform=ax0.transAxes)
            ax0.set_title("No Neuron Activity")
            ax0.tick_params(labelbottom=False)
        _lighten(ax0)

        # B) COM distance (3D solid, 2D dashed)
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        ax1.plot(tloc, d3d_loc, lw=1.6, color="0.15", label="3D")
        ax1.plot(tloc, dxy_loc, lw=1.0, ls="--", color="0.5", label="2D")
        ax1.axhline(approach_thresh_mm, ls="--", lw=1, color="0.3", alpha=0.6)
        ax1.axhline(contact_thresh_mm,  ls=":",  lw=1, color="0.3", alpha=0.6)
        ax1.set_ylabel("Distance (mm)")
        ax1.set_title("COM distance")
        ax1.legend(loc="upper right", fontsize="small", frameon=False)
        ax1.tick_params(labelbottom=False)
        _lighten(ax1)

        # C) Snout–snout distance
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
        label = "Snout–snout (3D)" if snout_is_3d else "Snout–snout (2D)"
        ax2.plot(tloc, snout_loc, lw=1.4, color="0.2", label=label)
        ax2.axhline(snout_contact_mm, ls="--", lw=1, color="0.3", alpha=0.6)
        ax2.set_ylabel("Distance (mm)")
        ax2.set_title(label)
        ax2.legend(loc="upper right", fontsize="small", frameon=False)
        ax2.tick_params(labelbottom=False)
        _lighten(ax2)

        # D) 3D−2D gap
        ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
        ax3.plot(tloc, gap_loc, lw=1.4, color="0.25")
        ax3.set_xlabel("Time from event rep (s)")
        ax3.set_ylabel("Δ distance (mm)")
        ax3.set_title("3D − 2D (COM)")
        _lighten(ax3)

        # Suptitle (minimal)
        min_d3d = np.nanmin(d3d_loc)
        snout_min = np.nanmin(snout_loc) if np.any(np.isfinite(snout_loc)) else np.nan
        bits = [f"Event {k}", f"min D3D={min_d3d:.1f} mm"]
        if np.isfinite(snout_min):
            bits.append(f"min snout={snout_min:.1f} mm")
        sup = ("{} — ".format(title_prefix) if title_prefix else "") + " | ".join(bits)
        fig.suptitle(sup, y=0.995, fontsize=11)

        plt.tight_layout()

        if save:
            od = out_dir or "."
            os.makedirs(od, exist_ok=True)
            out_path = os.path.join(od, f"incident_{k:03d}_clean.png")
            fig.savefig(out_path, dpi=300)
            print(f"[OK] Saved: {out_path}")

        plt.show()
        plt.close(fig)


# ==============================================================================
# Cell 47
# ==============================================================================

def _time_from_index_or_col(df: pd.DataFrame):
    """Return time array (seconds) and x-label. Prefers 'timestamp_ms_mini'."""
    if "timestamp_ms_mini" in df.columns:
        t = df["timestamp_ms_mini"].to_numpy(dtype=float) / 1000.0
        return t, "Time (s)"
    idx = df.index
    if getattr(idx, "name", None) and ("ms" in str(idx.name).lower()):
        return idx.to_numpy(dtype=float) / 1000.0, "Time (s)"
    if "timestamp_ms" in df.columns:
        return df["timestamp_ms"].to_numpy(dtype=float) / 1000.0, "Time (s)"
    if "time_s" in df.columns:
        return df["time_s"].to_numpy(dtype=float), "Time (s)"
    return np.arange(len(df), dtype=float), "Frame"

def _segments_from_mask(mask_1d_bool: Union[pd.Series, np.ndarray]):
    m = np.asarray(mask_1d_bool, dtype=bool)
    if m.size == 0:
        return np.empty((0, 2), dtype=int)
    starts = np.flatnonzero(~np.concatenate(([False], m[:-1])) & m)
    ends   = np.flatnonzero(m & ~np.concatenate((m[1:], [False])))
    if starts.size == 0:
        return np.empty((0, 2), dtype=int)
    return np.column_stack((starts, ends))

def _map_frame_segments_to_time(frames: pd.DataFrame, segs_fe: np.ndarray, col="timestamp_ms_mini"):
    """Map frame-based segments (row indices) to time ranges (ms)."""
    if col not in frames.columns:
        if getattr(frames.index, "name", None) and ("ms" in str(frames.index.name).lower()):
            t_ms = frames.index.to_numpy(dtype=float)
        else:
            t_ms = frames.index.to_numpy(dtype=float)
    else:
        t_ms = frames[col].to_numpy(dtype=float)
    out = []
    for s, e in segs_fe:
        ts = float(t_ms[s]); te = float(t_ms[e])
        if te < ts:
            ts, te = te, ts
        out.append((ts, te))
    return out

def _search_window_indices(t_vec_s: np.ndarray, t0_s: float, pre_s: float, post_s: float):
    """Return (i0, i1) such that t in [t0_s - pre_s, t0_s + post_s]."""
    t_start = t0_s - pre_s
    t_end   = t0_s + post_s
    i0 = int(np.searchsorted(t_vec_s, t_start, side="left"))
    i1 = int(np.searchsorted(t_vec_s, t_end, side="right")) - 1
    i0 = max(0, min(i0, len(t_vec_s)-1))
    i1 = max(0, min(i1, len(t_vec_s)-1))
    if i1 < i0:
        i0, i1 = i1, i0
    return i0, i1

def _prepare_neuron_matrix(merged: pd.DataFrame, variance_drop_pct: float = 5.0):
    """
    Returns (A_z, order_idx, roi_cols) where:
      - A_z: (n_neurons x T) z-scored ΔF/F
      - order_idx: Ward linkage order (stable across windows)
    """
    roi_cols = [c for c in merged.columns if c.startswith("dF_F_roi")]
    if len(roi_cols) == 0:
        return np.empty((0, len(merged))), np.arange(0), []

    A = merged[roi_cols].to_numpy(dtype=float, copy=False).T  # (N x T)

    with np.errstate(invalid="ignore"):
        var = np.nanvar(A, axis=1)
    var = np.where(np.isnan(var), 0.0, var)
    if variance_drop_pct and 0 < variance_drop_pct < 100 and A.shape[0] > 2:
        try:
            thr = np.nanpercentile(var, variance_drop_pct)
            keep = var > thr
            if keep.any():
                A = A[keep, :]
                roi_cols = [c for i, c in enumerate(roi_cols) if keep[i]]
        except Exception:
            pass

    with np.errstate(invalid="ignore"):
        A = zscore(A, axis=1, nan_policy="omit")
    if np.isnan(A).any():
        A = np.nan_to_num(A, nan=0.0)

    if A.shape[0] >= 2 and A.shape[1] >= 2:
        try:
            Z = linkage(A, method="ward")
            order = leaves_list(Z)
        except Exception:
            vv = np.nanvar(A, axis=1)
            order = np.argsort(-np.where(np.isnan(vv), -np.inf, vv))
    else:
        order = np.arange(A.shape[0])

    return A, order, roi_cols

def _vertical_sep_mm(merged: pd.DataFrame):
    """
    |Δz| between animals (mm). Prefer COM z; fallback to snout z; else NaN.
    Returns (vert_sep_mm, source_str).
    """
    if all(c in merged.columns for c in ["com1_z", "com2_z"]):
        dz = np.abs(merged["com2_z"].to_numpy(dtype=float) - merged["com1_z"].to_numpy(dtype=float))
        return dz, "COM z"
    if all(c in merged.columns for c in ["kp3_z_a1", "kp3_z_a2"]):
        dz = np.abs(merged["kp3_z_a2"].to_numpy(dtype=float) - merged["kp3_z_a1"].to_numpy(dtype=float))
        return dz, "Snout z"
    return np.full(len(merged), np.nan, dtype=float), "z unavailable"

def _pitch_yaw_deg(df, animal="a1"):
    """
    Head orientation from a simple head axis: SpineF (#4) → Snout (#3).
    Pitch: angle of that axis vs horizontal plane (deg).
    Yaw: angle of XY projection (deg).
    """
    n = len(df)
    need = [f"kp4_x_{animal}", f"kp4_y_{animal}", f"kp4_z_{animal}",
            f"kp3_x_{animal}", f"kp3_y_{animal}", f"kp3_z_{animal}"]
    if not all(c in df.columns for c in need):
        return np.full(n, np.nan), np.full(n, np.nan)

    x4 = df[f"kp4_x_{animal}"].to_numpy(float)
    y4 = df[f"kp4_y_{animal}"].to_numpy(float)
    z4 = df[f"kp4_z_{animal}"].to_numpy(float)
    x3 = df[f"kp3_x_{animal}"].to_numpy(float)
    y3 = df[f"kp3_y_{animal}"].to_numpy(float)
    z3 = df[f"kp3_z_{animal}"].to_numpy(float)

    vx, vy, vz = (x3 - x4), (y3 - y4), (z3 - z4)           # SpineF→Snout
    hyp = np.hypot(vx, vy)
    pitch = np.degrees(np.arctan2(vz, hyp))                # +up/-down
    yaw   = np.degrees(np.arctan2(vy, vx))                 # global XY
    return pitch, yaw


def plot_incident_windows_3d_advantage(
    merged: pd.DataFrame,
    frames: pd.DataFrame,
    mask: Union[pd.Series, np.ndarray],
    *,
    pre_s: float = 2.0,
    post_s: float = 2.0,
    approach_thresh_mm: float = 120.0,
    contact_thresh_mm: float = 50.0,
    variance_drop_pct: float = 5.0,
    heatmap_downsample: int = 1,
    heatmap_cmap: str = "viridis",
    heatmap_clim: Optional[Tuple[float, float]] = None,
    title_prefix: Optional[str] = None,
    save: bool = False,
    out_dir: Optional[str] = None,
):
    """
    Panels:
      A) Clustered ΔF/F heatmap (session-level order, z-score)
      B) COM distance: solid=3D, dashed=2D (+ thresholds)
      C) Vertical separation |Δz| (mm)  [3D-only]
      D) Head pitch (mean |pitch|, deg) [3D-only]
      E) 2D apparent-proximity underestimation (%) = 100*(1 - dXY/d3D)
    Window center = min 3D distance within each incident segment.
    """
    # time bases
    t_merged_s, _ = _time_from_index_or_col(merged)
    t_frames_s, _ = _time_from_index_or_col(frames)
    if t_merged_s.size == 0 or t_frames_s.size == 0:
        print("[WARN] Empty time vectors; aborting.")
        return
    t_merged_ms = t_merged_s * 1000.0

    # distances & components (requires your helper)
    try:
        comp = compute_com_distance(
            merged, p1="com1", p2="com2",
            smooth_window=1,
            dist_smooth_window=1,
            return_components=True
        )
    except NameError:
        raise RuntimeError("`compute_com_distance` is not defined. Import it before calling.")

    d3d = comp["dist_mm"].to_numpy(dtype=float)
    dx  = comp["dx"].to_numpy(dtype=float)
    dy  = comp["dy"].to_numpy(dtype=float)
    dxy = np.hypot(dx, dy)

    # 3D-only features
    vert_sep_mm, vert_src = _vertical_sep_mm(merged)
    pitch = _head_pitch_deg(merged)

    # neuron matrix once
    A_full, order, _ = _prepare_neuron_matrix(merged, variance_drop_pct)
    have_neurons = (A_full.size > 0)

    # incident segments in frames → miniscope time
    segs_fe = _segments_from_mask(mask)
    if segs_fe.size == 0:
        print("[INFO] No True segments in mask; nothing to plot.")
        return
    seg_time_ms = _map_frame_segments_to_time(frames, segs_fe, col="timestamp_ms_mini")

    # simple aesthetic helper
    def _lighten(ax):
        ax.grid(False)
        for s in ax.spines.values():
            s.set_linewidth(0.6)
            s.set_color("0.5")

    for k, (t0_ms, t1_ms) in enumerate(seg_time_ms, start=1):
        if t1_ms < t0_ms:
            t0_ms, t1_ms = t1_ms, t0_ms

        i0 = int(np.searchsorted(t_merged_ms, t0_ms, side="left"))
        i1 = int(np.searchsorted(t_merged_ms, t1_ms, side="right")) - 1
        i0 = max(0, min(i0, len(t_merged_s)-1))
        i1 = max(0, min(i1, len(t_merged_s)-1))
        if i1 <= i0:
            continue

        # representative frame = minimal 3D distance within segment
        seg_slice = slice(i0, i1+1)
        seg_d3d = d3d[seg_slice]
        if not np.any(np.isfinite(seg_d3d)):
            continue
        rep_idx = i0 + int(np.nanargmin(seg_d3d))
        rep_t = float(t_merged_s[rep_idx])

        # window bounds
        j0, j1 = _search_window_indices(t_merged_s, rep_t, pre_s, post_s)
        w = slice(j0, j1+1)
        tloc = t_merged_s[w] - rep_t

        # local series
        d3d_loc = d3d[w]
        dxy_loc = dxy[w]
        vert_loc = vert_sep_mm[w]
        pitch_loc = pitch["mean_abs"][w]

        # % underestimation (clip invalid)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_under = 100.0 * (1.0 - (dxy_loc / d3d_loc))
        pct_under = np.where(np.isfinite(pct_under), pct_under, np.nan)

        # heatmap slice (downsample for plotting only)
        if have_neurons:
            A_loc = A_full[order, w]
            if heatmap_downsample and heatmap_downsample > 1 and A_loc.shape[1] > 1:
                step = int(heatmap_downsample)
                A_plot = A_loc[:, ::step]
                t_plot = tloc[::step]
            else:
                A_plot = A_loc
                t_plot = tloc
        else:
            A_plot = np.empty((0, len(tloc)))
            t_plot = tloc

        # ===== Figure (5 stacked rows) — constrained layout =====
        fig = plt.figure(figsize=(12, 9.2), constrained_layout=True)
        # pads: a bit more air between rows; modest outer padding
        fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04, wspace=0.02, hspace=0.14)

        gs = fig.add_gridspec(
            nrows=5, ncols=1,
            height_ratios=[2.0, 1.2, 0.9, 0.9, 1.0]
        )

        # A) ΔF/F heatmap
        ax0 = fig.add_subplot(gs[0, 0])
        if A_plot.size > 0 and A_plot.shape[1] == t_plot.shape[0]:
            hm = ax0.pcolormesh(
                t_plot,
                np.arange(A_plot.shape[0] + 1),
                np.vstack([A_plot, A_plot[-1:]]) if A_plot.shape[0] > 0 else np.zeros((1, A_plot.shape[1])),
                cmap=heatmap_cmap,
                shading="auto"
            )
            if heatmap_clim is not None:
                hm.set_clim(*heatmap_clim)
            ax0.set_title("Clustered Neuron Activity (z-scored)")
            ax0.set_ylabel("Neurons")
            ax0.tick_params(labelbottom=False)
        else:
            ax0.text(0.5, 0.5, "No neuron data", ha="center", va="center", transform=ax0.transAxes)
            ax0.set_title("No Neuron Activity")
            ax0.tick_params(labelbottom=False)
        _lighten(ax0)

        # B) 3D vs 2D COM distance (legend outside to avoid geometry distortion)
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        ax1.plot(tloc, d3d_loc, lw=1.6, color="0.15", label="3D")
        ax1.plot(tloc, dxy_loc, lw=1.0, ls="--", color="0.5", label="2D")
        ax1.axhline(approach_thresh_mm, ls="--", lw=1, color="0.3", alpha=0.6)
        ax1.axhline(contact_thresh_mm,  ls=":",  lw=1, color="0.3", alpha=0.6)
        ax1.set_ylabel("Distance (mm)")
        ax1.set_title("COM distance")
        ax1.tick_params(labelbottom=False)
        _lighten(ax1)
        ax1.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize="small")

        # C) Vertical separation |Δz|
        ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
        ax2.plot(tloc, vert_loc, lw=1.4, color="0.2")
        ax2.set_ylabel("|Δz| (mm)")
        ax2.set_title(f"Vertical separation ({vert_src})")
        ax2.tick_params(labelbottom=False)
        _lighten(ax2)

        # D) Head pitch (mean |pitch|)
        ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
        ax3.plot(tloc, pitch_loc, lw=1.4, color="0.25")
        ax3.set_ylabel("Pitch (deg)")
        ax3.set_title("Head pitch (mean |pitch|)")
        ax3.tick_params(labelbottom=False)
        _lighten(ax3)

        # E) 2D apparent-proximity underestimation (%)
        ax4 = fig.add_subplot(gs[4, 0], sharex=ax0)
        ax4.plot(tloc, pct_under, lw=1.4, color="0.25")
        ax4.axhline(0, lw=1, color="0.5", alpha=0.6)
        ax4.set_xlabel("Time from event rep (s)")
        ax4.set_ylabel("Underest. (%)")
        ax4.set_title("2D apparent-proximity error: 100 × (1 − dXY/d3D)")
        _lighten(ax4)

        # Align all y-labels to the same left margin
        fig.align_ylabels([ax0, ax1, ax2, ax3, ax4])

        # Title (constrained layout accounts for this automatically)
        min_d3d = np.nanmin(d3d_loc)
        sup = (f"{title_prefix} — " if title_prefix else "") + f"Event {k} | min D3D={min_d3d:.1f} mm"
        fig.suptitle(sup, fontsize=11)

        if save:
            od = out_dir or "."
            os.makedirs(od, exist_ok=True)
            out_path = os.path.join(od, f"incident_{k:03d}_adv.png")
            fig.savefig(out_path, dpi=300)
            print(f"[OK] Saved: {out_path}")

        plt.show()
        plt.close(fig)


# ==============================================================================
# Cell 48
# ==============================================================================


# def plot_incident_windows_newset(
#     merged: pd.DataFrame,
#     frames: pd.DataFrame,
#     mask: Union[pd.Series, np.ndarray],
#     *,
#     pre_s: float = 2.0,
#     post_s: float = 2.0,
#     approach_thresh_mm: float = 120.0,
#     contact_thresh_mm: float = 50.0,
#     variance_drop_pct: float = 5.0,
#     heatmap_downsample: int = 1,
#     heatmap_cmap: str = "viridis",
#     title_prefix: Optional[str] = None,
#     save: bool = False,
#     out_dir: Optional[str] = None,
#     legend_outside: bool = True,
#     show_colorbar: bool = True,   # <— NEW
# ):
#     # time
#     t_merged_s, _ = _time_from_index_or_col(merged)
#     t_frames_s, _ = _time_from_index_or_col(frames)
#     if t_merged_s.size == 0 or t_frames_s.size == 0:
#         print("[WARN] empty time vectors"); return
#     t_merged_ms = t_merged_s * 1000.0

#     # metrics
#     try:
#         comp = compute_com_distance(
#             merged, p1="com1", p2="com2",
#             smooth_window=1, dist_smooth_window=1, return_components=True
#         )
#     except NameError:
#         raise RuntimeError("compute_com_distance(...) not found; import it first.")
#     d3d = comp["dist_mm"].to_numpy(float)
#     com_dz = _com_dz_abs(merged)
#     snout_d, snout_is3d = _snout_dist_3d(merged)
#     snout_dz = _snout_dz_abs(merged)

#     # neural + orientation
#     A_full, order, _ = _neuron_matrix(merged, variance_drop_pct)
#     have_neurons = (A_full.size > 0)
#     pitch_a1, yaw_a1 = _pitch_yaw_deg(merged, "a1")
#     pitch_a2, yaw_a2 = _pitch_yaw_deg(merged, "a2")

#     # segments
#     segs_fe = _segments_from_mask(mask)
#     if segs_fe.size == 0:
#         print("[INFO] mask has no True segments"); return
#     seg_time_ms = _map_frame_segments_to_time(frames, segs_fe, col="timestamp_ms_mini")

#     # style helpers
#     def _light(ax):
#         ax.grid(False)
#         for s in ax.spines.values():
#             s.set_linewidth(0.6); s.set_color("0.5")

#     col_dist, col_vert = "C0", "C3"
#     od = out_dir or "."

#     for k,(t0_ms, t1_ms) in enumerate(seg_time_ms, start=1):
#         if t1_ms < t0_ms: t0_ms, t1_ms = t1_ms, t0_ms
#         i0 = int(np.searchsorted(t_merged_ms, t0_ms, side="left"))
#         i1 = int(np.searchsorted(t_merged_ms, t1_ms, side="right")) - 1
#         i0 = max(0, min(i0, len(t_merged_s)-1))
#         i1 = max(0, min(i1, len(t_merged_s)-1))
#         if i1 <= i0: continue

#         seg = slice(i0, i1+1)
#         if not np.any(np.isfinite(d3d[seg])): continue
#         rep_idx = i0 + int(np.nanargmin(d3d[seg]))
#         rep_t = float(t_merged_s[rep_idx])
#         w = _window_bounds(t_merged_s, rep_t, pre_s, post_s)

#         tloc = t_merged_s[w] - rep_t
#         d3d_loc, com_dz_loc = d3d[w], com_dz[w]
#         snout_loc, snout_dz_loc = snout_d[w], snout_dz[w]

#         if have_neurons:
#             A_loc = A_full[order, w]
#             if heatmap_downsample > 1 and A_loc.shape[1] > 1:
#                 step = int(heatmap_downsample)
#                 A_plot = A_loc[:, ::step]; t_plot = tloc[::step]
#             else:
#                 A_plot, t_plot = A_loc, tloc
#         else:
#             A_plot = np.empty((0, len(tloc))); t_plot = tloc

#         # ================= single figure (aligned) =================
#         fig = plt.figure(figsize=(12, 12))
#         ratios = [2.0, 1.2, 0.9, 1.0, 0.9, 0.9, 0.9]
#         gs = fig.add_gridspec(nrows=len(ratios), ncols=1, height_ratios=ratios)
#         fig.subplots_adjust(left=0.10, right=0.86, top=0.95, bottom=0.08, hspace=0.20)

#         axes = [fig.add_subplot(gs[i,0]) for i in range(len(ratios))]
#         ax0, ax1, ax2, ax3, ax4, ax5, ax6 = axes

#         # (1) ΔF/F heatmap
#         if A_plot.size > 0 and A_plot.shape[1] == t_plot.shape[0]:
#             pcm = ax0.pcolormesh(                       # <— keep handle for cbar
#                 t_plot, np.arange(A_plot.shape[0]+1),
#                 np.vstack([A_plot, A_plot[-1:]]) if A_plot.shape[0] else np.zeros((1, A_plot.shape[1])),
#                 cmap=heatmap_cmap, shading="auto"
#             )
#             ax0.set_title("Clustered Neuron Activity (z-scored)", fontsize=18)
#             ax0.set_ylabel("Neurons")

#             if show_colorbar:
#                 # compact, non-intrusive; stays within ax0 box
#                 cb = fig.colorbar(pcm, ax=ax0, fraction=0.025, pad=0.01)
#                 cb.set_label("ΔF/F (z-score)", rotation=90)
#                 cb.outline.set_linewidth(0.6)
#                 cb.ax.tick_params(labelsize=9, width=0.6)
#         else:
#             ax0.text(0.5,0.5,"No neuron data",ha="center",va="center",transform=ax0.transAxes)
#             ax0.set_title("No Neuron Activity", fontsize=18)
#         ax0.tick_params(labelbottom=False); _light(ax0); ax0.axvline(0, lw=2.0, alpha=0.35)

#         # (2) COM 3D
#         l3d, = ax1.plot(tloc, d3d_loc, lw=2.8, color=col_dist, label="COM 3D")
#         ax1.axhline(approach_thresh_mm, ls="--", lw=1.2, color="0.5")
#         ax1.axhline(contact_thresh_mm,  ls=":",  lw=1.2, color="0.5")
#         ax1.set_ylabel("COM dist (mm)"); ax1.set_title("COM distance", fontsize=18)
#         ax1.tick_params(labelbottom=False); _light(ax1); ax1.axvline(0, lw=2.0, alpha=0.35)
#         if legend_outside:
#             ax1.legend(handles=[l3d], loc="center left", bbox_to_anchor=(1.02, 0.5),
#                        frameon=False, fontsize=12)

#         # (3) COM |Δz|
#         ax2.plot(tloc, com_dz_loc, lw=2.4, color=col_vert)
#         ax2.set_ylabel("|Δz| (mm)"); ax2.set_title("COM vertical separation", fontsize=18)
#         ax2.tick_params(labelbottom=False); _light(ax2); ax2.axvline(0, lw=2.0, alpha=0.35)

#         # (4) Snout–snout distance
#         ax3.plot(tloc, snout_loc, lw=2.4, color=col_dist)
#         ax3.set_ylabel("Snout dist (mm)")
#         ax3.set_title("Snout–snout distance" + (" (3D)" if snout_is3d else " (2D)"), fontsize=18)
#         ax3.tick_params(labelbottom=False); _light(ax3); ax3.axvline(0, lw=2.0, alpha=0.35)

#         # (5) Snout |Δz|
#         ax4.plot(tloc, snout_dz_loc, lw=2.4, color=col_vert)
#         ax4.set_ylabel("|Δz| (mm)"); ax4.set_title("Snout vertical separation", fontsize=18)
#         ax4.tick_params(labelbottom=False); _light(ax4); ax4.axvline(0, lw=2.0, alpha=0.35)

#         # (6) Pitch
#         lp1, = ax5.plot(tloc, pitch_a1[w], lw=2.2, label="A pitch")
#         lp2, = ax5.plot(tloc, pitch_a2[w], lw=2.2, label="B pitch")
#         ax5.set_ylabel("Pitch (deg)"); ax5.set_title("Head pitch", fontsize=18)
#         ax5.tick_params(labelbottom=False); _light(ax5); ax5.axvline(0, lw=2.0, alpha=0.35)
#         if legend_outside:
#             ax5.legend(handles=[lp1,lp2], loc="center left", bbox_to_anchor=(1.02, 0.5),
#                        frameon=False, fontsize=12)

#         # (7) Yaw
#         ly1, = ax6.plot(tloc, yaw_a1[w],  lw=2.2, label="A yaw")
#         ly2, = ax6.plot(tloc, yaw_a2[w],  lw=2.2, label="B yaw")
#         ax6.set_xlabel("Time from event (s)"); ax6.set_ylabel("Yaw (deg)")
#         ax6.set_title("Head yaw (global XY)", fontsize=18); _light(ax6)
#         ax6.axvline(0, lw=2.0, alpha=0.35)
#         if legend_outside:
#             ax6.legend(handles=[ly1,ly2], loc="center left", bbox_to_anchor=(1.02, 0.5),
#                        frameon=False, fontsize=12)

#         fig.align_ylabels(axes)

#         if save:
#             os.makedirs(od, exist_ok=True)
#             fig.savefig(os.path.join(od, f"event_{k:03d}_all_panels.png"), dpi=300)
#         plt.show(); plt.close(fig)


# def plot_incident_windows_newset(
#     merged: pd.DataFrame,
#     frames: pd.DataFrame,
#     mask: Union[pd.Series, np.ndarray],
#     *,
#     pre_s: float = 2.0,
#     post_s: float = 2.0,
#     approach_thresh_mm: float = 120.0,
#     contact_thresh_mm: float = 50.0,
#     variance_drop_pct: float = 5.0,
#     heatmap_downsample: int = 1,
#     heatmap_cmap: str = "viridis",
#     title_prefix: Optional[str] = None,
#     save: bool = False,
#     out_dir: Optional[str] = None,
#     legend_outside: bool = True,
#     show_colorbar: bool = True,   # <— NEW
# ):
#     # time
#     t_merged_s, _ = _time_from_index_or_col(merged)
#     t_frames_s, _ = _time_from_index_or_col(frames)
#     if t_merged_s.size == 0 or t_frames_s.size == 0:
#         print("[WARN] empty time vectors"); return
#     t_merged_ms = t_merged_s * 1000.0

#     # metrics
#     try:
#         comp = compute_com_distance(
#             merged, p1="com1", p2="com2",
#             smooth_window=1, dist_smooth_window=1, return_components=True
#         )
#     except NameError:
#         raise RuntimeError("compute_com_distance(...) not found; import it first.")
#     d3d = comp["dist_mm"].to_numpy(float)
#     com_dz = _com_dz_abs(merged)
#     snout_d, snout_is3d = _snout_dist_3d(merged)
#     snout_dz = _snout_dz_abs(merged)

#     # neural + orientation
#     A_full, order, _ = _neuron_matrix(merged, variance_drop_pct)
#     have_neurons = (A_full.size > 0)
#     pitch_a1, yaw_a1 = _pitch_yaw_deg(merged, "a1")
#     pitch_a2, yaw_a2 = _pitch_yaw_deg(merged, "a2")

#     # segments
#     segs_fe = _segments_from_mask(mask)
#     if segs_fe.size == 0:
#         print("[INFO] mask has no True segments"); return
#     seg_time_ms = _map_frame_segments_to_time(frames, segs_fe, col="timestamp_ms_mini")

#     # style helpers
#     def _light(ax):
#         ax.grid(False)
#         for s in ax.spines.values():
#             s.set_linewidth(0.6); s.set_color("0.5")

#     col_dist, col_vert = "C0", "C3"
#     od = out_dir or "."

#     # Collect all valid events first
#     event_data = []
#     for k,(t0_ms, t1_ms) in enumerate(seg_time_ms, start=1):
#         if t1_ms < t0_ms: t0_ms, t1_ms = t1_ms, t0_ms
#         i0 = int(np.searchsorted(t_merged_ms, t0_ms, side="left"))
#         i1 = int(np.searchsorted(t_merged_ms, t1_ms, side="right")) - 1
#         i0 = max(0, min(i0, len(t_merged_s)-1))
#         i1 = max(0, min(i1, len(t_merged_s)-1))
#         if i1 <= i0: continue

#         seg = slice(i0, i1+1)
#         if not np.any(np.isfinite(d3d[seg])): continue
#         rep_idx = i0 + int(np.nanargmin(d3d[seg]))
#         rep_t = float(t_merged_s[rep_idx])
#         w = _window_bounds(t_merged_s, rep_t, pre_s, post_s)

#         tloc = t_merged_s[w] - rep_t
#         d3d_loc, com_dz_loc = d3d[w], com_dz[w]
#         snout_loc, snout_dz_loc = snout_d[w], snout_dz[w]

#         if have_neurons:
#             A_loc = A_full[order, w]
#             if heatmap_downsample > 1 and A_loc.shape[1] > 1:
#                 step = int(heatmap_downsample)
#                 A_plot = A_loc[:, ::step]; t_plot = tloc[::step]
#             else:
#                 A_plot, t_plot = A_loc, tloc
#         else:
#             A_plot = np.empty((0, len(tloc))); t_plot = tloc

#         event_data.append({
#             'k': k,
#             'w': w,
#             'tloc': tloc,
#             'd3d_loc': d3d_loc,
#             'com_dz_loc': com_dz_loc,
#             'snout_loc': snout_loc,
#             'snout_dz_loc': snout_dz_loc,
#             'A_plot': A_plot,
#             't_plot': t_plot
#         })

#     if not event_data:
#         print("[INFO] No valid events to plot"); return

#     n_events = len(event_data)
    
#     # ================= Multi-column figure (all events side-by-side) =================
#     # Each event gets ~4 inches width, making 2s windows look more natural
#     fig_width = min(4 * n_events, 20)  # Cap at 20 inches total
#     fig = plt.figure(figsize=(fig_width, 11))
    
#     ratios = [2.0, 1.2, 0.9, 1.0, 0.9, 0.9, 0.9]
#     n_rows = len(ratios)
    
#     # Create grid: n_rows x n_events
#     gs = fig.add_gridspec(nrows=n_rows, ncols=n_events, height_ratios=ratios,
#                          hspace=0.35, wspace=0.25)
    
#     # Adjust margins
#     fig.subplots_adjust(left=0.08, right=0.96, top=0.94, bottom=0.06)

#     # Plot each event in its own column
#     for col_idx, evt in enumerate(event_data):
#         k = evt['k']
#         tloc = evt['tloc']
#         d3d_loc = evt['d3d_loc']
#         com_dz_loc = evt['com_dz_loc']
#         snout_loc = evt['snout_loc']
#         snout_dz_loc = evt['snout_dz_loc']
#         A_plot = evt['A_plot']
#         t_plot = evt['t_plot']
#         w = evt['w']
        
#         # Create subplots for this column
#         axes = [fig.add_subplot(gs[row, col_idx]) for row in range(n_rows)]
#         ax0, ax1, ax2, ax3, ax4, ax5, ax6 = axes

#         # (1) ΔF/F heatmap
#         if A_plot.size > 0 and A_plot.shape[1] == t_plot.shape[0]:
#             pcm = ax0.pcolormesh(
#                 t_plot, np.arange(A_plot.shape[0]+1),
#                 np.vstack([A_plot, A_plot[-1:]]) if A_plot.shape[0] else np.zeros((1, A_plot.shape[1])),
#                 cmap=heatmap_cmap, shading="auto"
#             )
#             if col_idx == 0:  # Only show title on first column
#                 ax0.set_title(f"Event {k}\nNeuron Activity", fontsize=12, pad=8)
#             else:
#                 ax0.set_title(f"Event {k}", fontsize=12, pad=8)
            
#             if col_idx == 0:  # Only leftmost gets y-label
#                 ax0.set_ylabel("Neurons", fontsize=10)

#             if show_colorbar and col_idx == n_events - 1:  # Only rightmost gets colorbar
#                 cb = fig.colorbar(pcm, ax=ax0, fraction=0.04, pad=0.02)
#                 cb.set_label("ΔF/F", rotation=90, fontsize=9)
#                 cb.outline.set_linewidth(0.6)
#                 cb.ax.tick_params(labelsize=8, width=0.6)
#         else:
#             ax0.text(0.5,0.5,"No data",ha="center",va="center",transform=ax0.transAxes, fontsize=9)
#             ax0.set_title(f"Event {k}", fontsize=12, pad=8)
#         ax0.tick_params(labelbottom=False, labelsize=9); _light(ax0); ax0.axvline(0, lw=1.5, alpha=0.35)

#         # (2) COM 3D
#         ax1.plot(tloc, d3d_loc, lw=2.2, color=col_dist)
#         ax1.axhline(approach_thresh_mm, ls="--", lw=1.0, color="0.5")
#         ax1.axhline(contact_thresh_mm,  ls=":",  lw=1.0, color="0.5")
#         if col_idx == 0:
#             ax1.set_ylabel("COM (mm)", fontsize=10)
#         ax1.tick_params(labelbottom=False, labelsize=9); _light(ax1); ax1.axvline(0, lw=1.5, alpha=0.35)

#         # (3) COM |Δz|
#         ax2.plot(tloc, com_dz_loc, lw=2.0, color=col_vert)
#         if col_idx == 0:
#             ax2.set_ylabel("|Δz| (mm)", fontsize=10)
#         ax2.tick_params(labelbottom=False, labelsize=9); _light(ax2); ax2.axvline(0, lw=1.5, alpha=0.35)

#         # (4) Snout–snout distance
#         ax3.plot(tloc, snout_loc, lw=2.0, color=col_dist)
#         if col_idx == 0:
#             ax3.set_ylabel("Snout (mm)", fontsize=10)
#         ax3.tick_params(labelbottom=False, labelsize=9); _light(ax3); ax3.axvline(0, lw=1.5, alpha=0.35)

#         # (5) Snout |Δz|
#         ax4.plot(tloc, snout_dz_loc, lw=2.0, color=col_vert)
#         if col_idx == 0:
#             ax4.set_ylabel("|Δz| (mm)", fontsize=10)
#         ax4.tick_params(labelbottom=False, labelsize=9); _light(ax4); ax4.axvline(0, lw=1.5, alpha=0.35)

#         # (6) Pitch
#         ax5.plot(tloc, pitch_a1[w], lw=1.8, label="A" if col_idx==0 else "")
#         ax5.plot(tloc, pitch_a2[w], lw=1.8, label="B" if col_idx==0 else "")
#         if col_idx == 0:
#             ax5.set_ylabel("Pitch (°)", fontsize=10)
#             ax5.legend(fontsize=8, frameon=False, loc='upper left')
#         ax5.tick_params(labelbottom=False, labelsize=9); _light(ax5); ax5.axvline(0, lw=1.5, alpha=0.35)

#         # (7) Yaw
#         ax6.plot(tloc, yaw_a1[w],  lw=1.8, label="A" if col_idx==0 else "")
#         ax6.plot(tloc, yaw_a2[w],  lw=1.8, label="B" if col_idx==0 else "")
#         ax6.set_xlabel("Time (s)", fontsize=10)
#         if col_idx == 0:
#             ax6.set_ylabel("Yaw (°)", fontsize=10)
#             ax6.legend(fontsize=8, frameon=False, loc='upper left')
#         ax6.tick_params(labelsize=9); _light(ax6); ax6.axvline(0, lw=1.5, alpha=0.35)

#     if title_prefix:
#         fig.suptitle(title_prefix, fontsize=14, y=0.98)

#     if save:
#         os.makedirs(od, exist_ok=True)
#         fig.savefig(os.path.join(od, "all_events_comparison.png"), dpi=300, bbox_inches='tight')
    
#     plt.show()
#     plt.close(fig)

# def plot_incident_windows_newset(
#     merged: pd.DataFrame,
#     frames: pd.DataFrame,
#     mask: Union[pd.Series, np.ndarray],
#     *,
#     pre_s: float = 2.0,
#     post_s: float = 2.0,
#     approach_thresh_mm: float = 120.0,
#     contact_thresh_mm: float = 50.0,
#     variance_drop_pct: float = 5.0,
#     heatmap_downsample: int = 1,
#     heatmap_cmap: str = "viridis",
#     title_prefix: Optional[str] = None,
#     save: bool = False,
#     out_dir: Optional[str] = None,
#     legend_outside: bool = True,
#     show_colorbar: bool = True,
# ):
#     """Original version: one figure per event."""
#     # time
#     t_merged_s, _ = _time_from_index_or_col(merged)
#     t_frames_s, _ = _time_from_index_or_col(frames)
#     if t_merged_s.size == 0 or t_frames_s.size == 0:
#         print("[WARN] empty time vectors"); return
#     t_merged_ms = t_merged_s * 1000.0

#     # metrics
#     try:
#         comp = compute_com_distance(
#             merged, p1="com1", p2="com2",
#             smooth_window=1, dist_smooth_window=1, return_components=True
#         )
#     except NameError:
#         raise RuntimeError("compute_com_distance(...) not found; import it first.")
#     d3d = comp["dist_mm"].to_numpy(float)
#     com_dz = _com_dz_abs(merged)
#     snout_d, snout_is3d = _snout_dist_3d(merged)
#     snout_dz = _snout_dz_abs(merged)

#     # neural + orientation
#     A_full, order, _ = _neuron_matrix(merged, variance_drop_pct)
#     have_neurons = (A_full.size > 0)
    
#     # Try to get pitch/yaw with error handling
#     try:
#         pitch_a1, yaw_a1 = _pitch_yaw_deg(merged, "a1")
#         pitch_a2, yaw_a2 = _pitch_yaw_deg(merged, "a2")
#         have_orientation = True
#         # Check if data is actually valid
#         if (pitch_a1.size == 0 or np.all(~np.isfinite(pitch_a1)) or 
#             pitch_a2.size == 0 or np.all(~np.isfinite(pitch_a2))):
#             print("[WARN] Pitch/yaw data exists but is all NaN/invalid")
#             have_orientation = False
#     except Exception as e:
#         print(f"[WARN] Could not compute pitch/yaw: {e}")
#         have_orientation = False
#         # Create dummy arrays
#         pitch_a1 = np.zeros(len(merged))
#         pitch_a2 = np.zeros(len(merged))
#         yaw_a1 = np.zeros(len(merged))
#         yaw_a2 = np.zeros(len(merged))

#     # segments
#     segs_fe = _segments_from_mask(mask)
#     if segs_fe.size == 0:
#         print("[INFO] mask has no True segments"); return
#     seg_time_ms = _map_frame_segments_to_time(frames, segs_fe, col="timestamp_ms_mini")

#     # style helpers
#     def _light(ax):
#         ax.grid(False)
#         for s in ax.spines.values():
#             s.set_linewidth(0.6); s.set_color("0.5")

#     col_dist, col_vert = "C0", "C3"
#     od = out_dir or "."

#     for k,(t0_ms, t1_ms) in enumerate(seg_time_ms, start=1):
#         if t1_ms < t0_ms: t0_ms, t1_ms = t1_ms, t0_ms
#         i0 = int(np.searchsorted(t_merged_ms, t0_ms, side="left"))
#         i1 = int(np.searchsorted(t_merged_ms, t1_ms, side="right")) - 1
#         i0 = max(0, min(i0, len(t_merged_s)-1))
#         i1 = max(0, min(i1, len(t_merged_s)-1))
#         if i1 <= i0: continue

#         seg = slice(i0, i1+1)
#         if not np.any(np.isfinite(d3d[seg])): continue
#         rep_idx = i0 + int(np.nanargmin(d3d[seg]))
#         rep_t = float(t_merged_s[rep_idx])
#         w = _window_bounds(t_merged_s, rep_t, pre_s, post_s)

#         tloc = t_merged_s[w] - rep_t
#         d3d_loc, com_dz_loc = d3d[w], com_dz[w]
#         snout_loc, snout_dz_loc = snout_d[w], snout_dz[w]

#         if have_neurons:
#             A_loc = A_full[order, w]
#             if heatmap_downsample > 1 and A_loc.shape[1] > 1:
#                 step = int(heatmap_downsample)
#                 A_plot = A_loc[:, ::step]; t_plot = tloc[::step]
#             else:
#                 A_plot, t_plot = A_loc, tloc
#         else:
#             A_plot = np.empty((0, len(tloc))); t_plot = tloc

#         # ================= single figure (aligned) =================
#         fig = plt.figure(figsize=(12, 12))
#         ratios = [2.0, 1.2, 0.9, 1.0, 0.9, 0.9, 0.9]
#         gs = fig.add_gridspec(nrows=len(ratios), ncols=1, height_ratios=ratios)
#         fig.subplots_adjust(left=0.10, right=0.86, top=0.95, bottom=0.08, hspace=0.35)

#         axes = [fig.add_subplot(gs[i,0]) for i in range(len(ratios))]
#         ax0, ax1, ax2, ax3, ax4, ax5, ax6 = axes

#         # (1) ΔF/F heatmap
#         if A_plot.size > 0 and A_plot.shape[1] == t_plot.shape[0]:
#             pcm = ax0.pcolormesh(
#                 t_plot, np.arange(A_plot.shape[0]+1),
#                 np.vstack([A_plot, A_plot[-1:]]) if A_plot.shape[0] else np.zeros((1, A_plot.shape[1])),
#                 cmap=heatmap_cmap, shading="auto"
#             )
#             ax0.set_title("Clustered Neuron Activity (z-scored)", fontsize=14)
#             ax0.set_ylabel("Neurons", fontsize=11)

#             if show_colorbar:
#                 cb = fig.colorbar(pcm, ax=ax0, fraction=0.025, pad=0.01)
#                 cb.set_label("ΔF/F (z-score)", rotation=90, fontsize=10)
#                 cb.outline.set_linewidth(0.6)
#                 cb.ax.tick_params(labelsize=9, width=0.6)
#         else:
#             ax0.text(0.5,0.5,"No neuron data",ha="center",va="center",transform=ax0.transAxes)
#             ax0.set_title("No Neuron Activity", fontsize=14)
#         ax0.tick_params(labelbottom=False); _light(ax0); ax0.axvline(0, lw=2.0, alpha=0.35)

#         # (2) COM 3D
#         l3d, = ax1.plot(tloc, d3d_loc, lw=2.8, color=col_dist, label="COM 3D")
#         ax1.axhline(approach_thresh_mm, ls="--", lw=1.2, color="0.5")
#         ax1.axhline(contact_thresh_mm,  ls=":",  lw=1.2, color="0.5")
#         ax1.set_ylabel("COM dist (mm)", fontsize=11)
#         ax1.set_title("COM distance", fontsize=14)
#         ax1.tick_params(labelbottom=False); _light(ax1); ax1.axvline(0, lw=2.0, alpha=0.35)
#         if legend_outside:
#             ax1.legend(handles=[l3d], loc="center left", bbox_to_anchor=(1.02, 0.5),
#                        frameon=False, fontsize=11)

#         # (3) COM |Δz|
#         ax2.plot(tloc, com_dz_loc, lw=2.4, color=col_vert)
#         ax2.set_ylabel("|Δz| (mm)", fontsize=11)
#         ax2.set_title("COM vertical separation", fontsize=14)
#         ax2.tick_params(labelbottom=False); _light(ax2); ax2.axvline(0, lw=2.0, alpha=0.35)

#         # (4) Snout–snout distance
#         ax3.plot(tloc, snout_loc, lw=2.4, color=col_dist)
#         ax3.set_ylabel("Snout dist (mm)", fontsize=11)
#         ax3.set_title("Snout–snout distance" + (" (3D)" if snout_is3d else " (2D)"), fontsize=14)
#         ax3.tick_params(labelbottom=False); _light(ax3); ax3.axvline(0, lw=2.0, alpha=0.35)

#         # (5) Snout |Δz|
#         ax4.plot(tloc, snout_dz_loc, lw=2.4, color=col_vert)
#         ax4.set_ylabel("|Δz| (mm)", fontsize=11)
#         ax4.set_title("Snout vertical separation", fontsize=14)
#         ax4.tick_params(labelbottom=False); _light(ax4); ax4.axvline(0, lw=2.0, alpha=0.35)

#         # (6) Pitch
#         if have_orientation:
#             lp1, = ax5.plot(tloc, pitch_a1[w], lw=2.2, label="A pitch")
#             lp2, = ax5.plot(tloc, pitch_a2[w], lw=2.2, label="B pitch")
#         else:
#             ax5.text(0.5,0.5,"No orientation data",ha="center",va="center",
#                     transform=ax5.transAxes, fontsize=10)
#         ax5.set_ylabel("Pitch (deg)", fontsize=11)
#         ax5.set_title("Head pitch", fontsize=14)
#         ax5.tick_params(labelbottom=False); _light(ax5); ax5.axvline(0, lw=2.0, alpha=0.35)
#         if legend_outside and have_orientation:
#             ax5.legend(handles=[lp1,lp2], loc="center left", bbox_to_anchor=(1.02, 0.5),
#                        frameon=False, fontsize=11)

#         # (7) Yaw
#         if have_orientation:
#             ly1, = ax6.plot(tloc, yaw_a1[w],  lw=2.2, label="A yaw")
#             ly2, = ax6.plot(tloc, yaw_a2[w],  lw=2.2, label="B yaw")
#         else:
#             ax6.text(0.5,0.5,"No orientation data",ha="center",va="center",
#                     transform=ax6.transAxes, fontsize=10)
#         ax6.set_xlabel("Time from event (s)", fontsize=11)
#         ax6.set_ylabel("Yaw (deg)", fontsize=11)
#         ax6.set_title("Head yaw (global XY)", fontsize=14)
#         _light(ax6); ax6.axvline(0, lw=2.0, alpha=0.35)
#         if legend_outside and have_orientation:
#             ax6.legend(handles=[ly1,ly2], loc="center left", bbox_to_anchor=(1.02, 0.5),
#                        frameon=False, fontsize=11)

#         fig.align_ylabels(axes)

#         if save:
#             os.makedirs(od, exist_ok=True)
#             fig.savefig(os.path.join(od, f"event_{k:03d}_all_panels.png"), dpi=300)
#         plt.show(); plt.close(fig)


# def plot_incident_windows_newset_comb(
#     merged: pd.DataFrame,
#     frames: pd.DataFrame,
#     mask: Union[pd.Series, np.ndarray],
#     *,
#     pre_s: float = 2.0,
#     post_s: float = 2.0,
#     approach_thresh_mm: float = 120.0,
#     contact_thresh_mm: float = 50.0,
#     variance_drop_pct: float = 5.0,
#     heatmap_downsample: int = 1,
#     heatmap_cmap: str = "viridis",
#     title_prefix: Optional[str] = None,
#     save: bool = False,
#     out_dir: Optional[str] = None,
#     legend_outside: bool = True,
#     show_colorbar: bool = True,
# ):
#     """Combined version: all events side-by-side in one figure."""
#     # time
#     t_merged_s, _ = _time_from_index_or_col(merged)
#     t_frames_s, _ = _time_from_index_or_col(frames)
#     if t_merged_s.size == 0 or t_frames_s.size == 0:
#         print("[WARN] empty time vectors"); return
#     t_merged_ms = t_merged_s * 1000.0

#     # metrics
#     try:
#         comp = compute_com_distance(
#             merged, p1="com1", p2="com2",
#             smooth_window=1, dist_smooth_window=1, return_components=True
#         )
#     except NameError:
#         raise RuntimeError("compute_com_distance(...) not found; import it first.")
#     d3d = comp["dist_mm"].to_numpy(float)
#     com_dz = _com_dz_abs(merged)
#     snout_d, snout_is3d = _snout_dist_3d(merged)
#     snout_dz = _snout_dz_abs(merged)

#     # neural + orientation
#     A_full, order, _ = _neuron_matrix(merged, variance_drop_pct)
#     have_neurons = (A_full.size > 0)
    
#     # Try to get pitch/yaw with error handling
#     try:
#         pitch_a1, yaw_a1 = _pitch_yaw_deg(merged, "a1")
#         pitch_a2, yaw_a2 = _pitch_yaw_deg(merged, "a2")
#         have_orientation = True
#         # Check if data is actually valid
#         if (pitch_a1.size == 0 or np.all(~np.isfinite(pitch_a1)) or 
#             pitch_a2.size == 0 or np.all(~np.isfinite(pitch_a2))):
#             print("[WARN] Pitch/yaw data exists but is all NaN/invalid")
#             have_orientation = False
#     except Exception as e:
#         print(f"[WARN] Could not compute pitch/yaw: {e}")
#         have_orientation = False
#         # Create dummy arrays
#         pitch_a1 = np.zeros(len(merged))
#         pitch_a2 = np.zeros(len(merged))
#         yaw_a1 = np.zeros(len(merged))
#         yaw_a2 = np.zeros(len(merged))

#     # segments
#     segs_fe = _segments_from_mask(mask)
#     if segs_fe.size == 0:
#         print("[INFO] mask has no True segments"); return
#     seg_time_ms = _map_frame_segments_to_time(frames, segs_fe, col="timestamp_ms_mini")

#     # style helpers
#     def _light(ax):
#         ax.grid(False)
#         for s in ax.spines.values():
#             s.set_linewidth(0.6); s.set_color("0.5")

#     col_dist, col_vert = "C0", "C3"
#     od = out_dir or "."

#     # Collect all valid events first
#     event_data = []
#     for k,(t0_ms, t1_ms) in enumerate(seg_time_ms, start=1):
#         if t1_ms < t0_ms: t0_ms, t1_ms = t1_ms, t0_ms
#         i0 = int(np.searchsorted(t_merged_ms, t0_ms, side="left"))
#         i1 = int(np.searchsorted(t_merged_ms, t1_ms, side="right")) - 1
#         i0 = max(0, min(i0, len(t_merged_s)-1))
#         i1 = max(0, min(i1, len(t_merged_s)-1))
#         if i1 <= i0: continue

#         seg = slice(i0, i1+1)
#         if not np.any(np.isfinite(d3d[seg])): continue
#         rep_idx = i0 + int(np.nanargmin(d3d[seg]))
#         rep_t = float(t_merged_s[rep_idx])
#         w = _window_bounds(t_merged_s, rep_t, pre_s, post_s)

#         tloc = t_merged_s[w] - rep_t
#         d3d_loc, com_dz_loc = d3d[w], com_dz[w]
#         snout_loc, snout_dz_loc = snout_d[w], snout_dz[w]

#         if have_neurons:
#             A_loc = A_full[order, w]
#             if heatmap_downsample > 1 and A_loc.shape[1] > 1:
#                 step = int(heatmap_downsample)
#                 A_plot = A_loc[:, ::step]; t_plot = tloc[::step]
#             else:
#                 A_plot, t_plot = A_loc, tloc
#         else:
#             A_plot = np.empty((0, len(tloc))); t_plot = tloc

#         event_data.append({
#             'k': k,
#             'w': w,
#             'tloc': tloc,
#             'd3d_loc': d3d_loc,
#             'com_dz_loc': com_dz_loc,
#             'snout_loc': snout_loc,
#             'snout_dz_loc': snout_dz_loc,
#             'A_plot': A_plot,
#             't_plot': t_plot
#         })

#     if not event_data:
#         print("[INFO] No valid events to plot"); return

#     n_events = len(event_data)
    
#     # ================= Multi-column figure (all events side-by-side) =================
#     # Each event gets ~4 inches width, making 2s windows look more natural
#     fig_width = min(4 * n_events, 20)  # Cap at 20 inches total
#     fig = plt.figure(figsize=(fig_width, 11))
    
#     ratios = [2.0, 1.2, 0.9, 1.0, 0.9, 0.9, 0.9]
#     n_rows = len(ratios)
    
#     # Create grid: n_rows x n_events
#     gs = fig.add_gridspec(nrows=n_rows, ncols=n_events, height_ratios=ratios,
#                          hspace=0.35, wspace=0.25)
    
#     # Adjust margins
#     fig.subplots_adjust(left=0.08, right=0.96, top=0.94, bottom=0.06)

#     # Plot each event in its own column
#     for col_idx, evt in enumerate(event_data):
#         k = evt['k']
#         tloc = evt['tloc']
#         d3d_loc = evt['d3d_loc']
#         com_dz_loc = evt['com_dz_loc']
#         snout_loc = evt['snout_loc']
#         snout_dz_loc = evt['snout_dz_loc']
#         A_plot = evt['A_plot']
#         t_plot = evt['t_plot']
#         w = evt['w']
        
#         # Create subplots for this column
#         axes = [fig.add_subplot(gs[row, col_idx]) for row in range(n_rows)]
#         ax0, ax1, ax2, ax3, ax4, ax5, ax6 = axes

#         # (1) ΔF/F heatmap
#         if A_plot.size > 0 and A_plot.shape[1] == t_plot.shape[0]:
#             pcm = ax0.pcolormesh(
#                 t_plot, np.arange(A_plot.shape[0]+1),
#                 np.vstack([A_plot, A_plot[-1:]]) if A_plot.shape[0] else np.zeros((1, A_plot.shape[1])),
#                 cmap=heatmap_cmap, shading="auto"
#             )
#             if col_idx == 0:
#                 ax0.set_title(f"Event {k}\nNeuron Activity", fontsize=12, pad=8)
#             else:
#                 ax0.set_title(f"Event {k}", fontsize=12, pad=8)
            
#             if col_idx == 0:
#                 ax0.set_ylabel("Neurons", fontsize=10)

#             if show_colorbar and col_idx == n_events - 1:
#                 cb = fig.colorbar(pcm, ax=ax0, fraction=0.04, pad=0.02)
#                 cb.set_label("ΔF/F", rotation=90, fontsize=9)
#                 cb.outline.set_linewidth(0.6)
#                 cb.ax.tick_params(labelsize=8, width=0.6)
#         else:
#             ax0.text(0.5,0.5,"No data",ha="center",va="center",transform=ax0.transAxes, fontsize=9)
#             ax0.set_title(f"Event {k}", fontsize=12, pad=8)
#         ax0.tick_params(labelbottom=False, labelsize=9); _light(ax0); ax0.axvline(0, lw=1.5, alpha=0.35)

#         # (2) COM 3D
#         ax1.plot(tloc, d3d_loc, lw=2.2, color=col_dist)
#         ax1.axhline(approach_thresh_mm, ls="--", lw=1.0, color="0.5")
#         ax1.axhline(contact_thresh_mm,  ls=":",  lw=1.0, color="0.5")
#         if col_idx == 0:
#             ax1.set_ylabel("COM (mm)", fontsize=10)
#         ax1.tick_params(labelbottom=False, labelsize=9); _light(ax1); ax1.axvline(0, lw=1.5, alpha=0.35)

#         # (3) COM |Δz|
#         ax2.plot(tloc, com_dz_loc, lw=2.0, color=col_vert)
#         if col_idx == 0:
#             ax2.set_ylabel("|Δz| (mm)", fontsize=10)
#         ax2.tick_params(labelbottom=False, labelsize=9); _light(ax2); ax2.axvline(0, lw=1.5, alpha=0.35)

#         # (4) Snout–snout distance
#         ax3.plot(tloc, snout_loc, lw=2.0, color=col_dist)
#         if col_idx == 0:
#             ax3.set_ylabel("Snout (mm)", fontsize=10)
#         ax3.tick_params(labelbottom=False, labelsize=9); _light(ax3); ax3.axvline(0, lw=1.5, alpha=0.35)

#         # (5) Snout |Δz|
#         ax4.plot(tloc, snout_dz_loc, lw=2.0, color=col_vert)
#         if col_idx == 0:
#             ax4.set_ylabel("|Δz| (mm)", fontsize=10)
#         ax4.tick_params(labelbottom=False, labelsize=9); _light(ax4); ax4.axvline(0, lw=1.5, alpha=0.35)

#         # (6) Pitch
#         if have_orientation:
#             ax5.plot(tloc, pitch_a1[w], lw=1.8, label="A" if col_idx==0 else "")
#             ax5.plot(tloc, pitch_a2[w], lw=1.8, label="B" if col_idx==0 else "")
#             if col_idx == 0:
#                 ax5.legend(fontsize=8, frameon=False, loc='upper left')
#         else:
#             if col_idx == 0:
#                 ax5.text(0.5,0.5,"No data",ha="center",va="center",
#                         transform=ax5.transAxes, fontsize=9)
#         if col_idx == 0:
#             ax5.set_ylabel("Pitch (°)", fontsize=10)
#         ax5.tick_params(labelbottom=False, labelsize=9); _light(ax5); ax5.axvline(0, lw=1.5, alpha=0.35)

#         # (7) Yaw
#         if have_orientation:
#             ax6.plot(tloc, yaw_a1[w],  lw=1.8, label="A" if col_idx==0 else "")
#             ax6.plot(tloc, yaw_a2[w],  lw=1.8, label="B" if col_idx==0 else "")
#             if col_idx == 0:
#                 ax6.legend(fontsize=8, frameon=False, loc='upper left')
#         else:
#             if col_idx == 0:
#                 ax6.text(0.5,0.5,"No data",ha="center",va="center",
#                         transform=ax6.transAxes, fontsize=9)
#         ax6.set_xlabel("Time (s)", fontsize=10)
#         if col_idx == 0:
#             ax6.set_ylabel("Yaw (°)", fontsize=10)
#         ax6.tick_params(labelsize=9); _light(ax6); ax6.axvline(0, lw=1.5, alpha=0.35)

#     if title_prefix:
#         fig.suptitle(title_prefix, fontsize=14, y=0.98)

#     if save:
#         os.makedirs(od, exist_ok=True)
#         fig.savefig(os.path.join(od, "all_events_comparison.png"), dpi=300, bbox_inches='tight')
    
#     plt.show()
#     plt.close(fig)


def plot_incident_windows_newset(
    merged: pd.DataFrame,
    frames: pd.DataFrame,
    mask: Union[pd.Series, np.ndarray],
    *,
    pre_s: float = 2.0,
    post_s: float = 2.0,
    approach_thresh_mm: float = 120.0,
    contact_thresh_mm: float = 50.0,
    variance_drop_pct: float = 5.0,
    heatmap_downsample: int = 1,
    heatmap_cmap: str = "viridis",
    title_prefix: Optional[str] = None,
    save: bool = False,
    out_dir: Optional[str] = None,
    legend_outside: bool = True,
    show_colorbar: bool = True,
    show_orientation_legends: bool = False,  # NEW: control A/B legends
):
    """
    Original version: one figure per event.
    Shows 7 panels vertically for each event.
    """
    # time
    t_merged_s, _ = _time_from_index_or_col(merged)
    t_frames_s, _ = _time_from_index_or_col(frames)
    if t_merged_s.size == 0 or t_frames_s.size == 0:
        print("[WARN] empty time vectors"); return
    t_merged_ms = t_merged_s * 1000.0

    # metrics
    try:
        comp = compute_com_distance(
            merged, p1="com1", p2="com2",
            smooth_window=1, dist_smooth_window=1, return_components=True
        )
    except NameError:
        raise RuntimeError("compute_com_distance(...) not found; import it first.")
    d3d = comp["dist_mm"].to_numpy(float)
    com_dz = _com_dz_abs(merged)
    snout_d, snout_is3d = _snout_dist_3d(merged)
    snout_dz = _snout_dz_abs(merged)

    # neural + orientation
    A_full, order, _ = _neuron_matrix(merged, variance_drop_pct)
    have_neurons = (A_full.size > 0)
    pitch_a1, yaw_a1 = _pitch_yaw_deg(merged, "a1")
    pitch_a2, yaw_a2 = _pitch_yaw_deg(merged, "a2")

    # segments
    segs_fe = _segments_from_mask(mask)
    if segs_fe.size == 0:
        print("[INFO] mask has no True segments"); return
    seg_time_ms = _map_frame_segments_to_time(frames, segs_fe, col="timestamp_ms_mini")

    # style helpers
    def _light(ax):
        ax.grid(False)
        for s in ax.spines.values():
            s.set_linewidth(0.6); s.set_color("0.5")

    col_dist, col_vert = "C0", "C3"
    od = out_dir or "."

    for k,(t0_ms, t1_ms) in enumerate(seg_time_ms, start=1):
        if t1_ms < t0_ms: t0_ms, t1_ms = t1_ms, t0_ms
        i0 = int(np.searchsorted(t_merged_ms, t0_ms, side="left"))
        i1 = int(np.searchsorted(t_merged_ms, t1_ms, side="right")) - 1
        i0 = max(0, min(i0, len(t_merged_s)-1))
        i1 = max(0, min(i1, len(t_merged_s)-1))
        if i1 <= i0: continue

        seg = slice(i0, i1+1)
        if not np.any(np.isfinite(d3d[seg])): continue
        rep_idx = i0 + int(np.nanargmin(d3d[seg]))
        rep_t = float(t_merged_s[rep_idx])
        w = _window_bounds(t_merged_s, rep_t, pre_s, post_s)

        tloc = t_merged_s[w] - rep_t
        d3d_loc, com_dz_loc = d3d[w], com_dz[w]
        snout_loc, snout_dz_loc = snout_d[w], snout_dz[w]

        if have_neurons:
            A_loc = A_full[order, w]
            if heatmap_downsample > 1 and A_loc.shape[1] > 1:
                step = int(heatmap_downsample)
                A_plot = A_loc[:, ::step]; t_plot = tloc[::step]
            else:
                A_plot, t_plot = A_loc, tloc
        else:
            A_plot = np.empty((0, len(tloc))); t_plot = tloc

        # ================= single figure (aligned) =================
        fig = plt.figure(figsize=(12, 12))
        ratios = [2.0, 1.2, 0.9, 1.0, 0.9, 0.9, 0.9]
        gs = fig.add_gridspec(nrows=len(ratios), ncols=1, height_ratios=ratios)
        fig.subplots_adjust(left=0.10, right=0.86, top=0.95, bottom=0.08, hspace=0.35)

        axes = [fig.add_subplot(gs[i,0]) for i in range(len(ratios))]
        ax0, ax1, ax2, ax3, ax4, ax5, ax6 = axes

        # (1) ΔF/F heatmap
        if A_plot.size > 0 and A_plot.shape[1] == t_plot.shape[0]:
            pcm = ax0.pcolormesh(
                t_plot, np.arange(A_plot.shape[0]+1),
                np.vstack([A_plot, A_plot[-1:]]) if A_plot.shape[0] else np.zeros((1, A_plot.shape[1])),
                cmap=heatmap_cmap, shading="auto"
            )
            ax0.set_title("Clustered Neuron Activity (z-scored)", fontsize=14)
            ax0.set_ylabel("Neurons", fontsize=11)

            if show_colorbar:
                cb = fig.colorbar(pcm, ax=ax0, fraction=0.025, pad=0.01)
                cb.set_label("ΔF/F (z-score)", rotation=90, fontsize=10)
                cb.outline.set_linewidth(0.6)
                cb.ax.tick_params(labelsize=9, width=0.6)
        else:
            ax0.text(0.5,0.5,"No neuron data",ha="center",va="center",transform=ax0.transAxes)
            ax0.set_title("No Neuron Activity", fontsize=14)
        ax0.tick_params(labelbottom=False); _light(ax0); ax0.axvline(0, lw=2.0, alpha=0.35)

        # (2) COM 3D
        l3d, = ax1.plot(tloc, d3d_loc, lw=2.8, color=col_dist, label="COM 3D")
        ax1.axhline(approach_thresh_mm, ls="--", lw=1.2, color="0.5")
        ax1.axhline(contact_thresh_mm,  ls=":",  lw=1.2, color="0.5")
        ax1.set_ylabel("COM dist (mm)", fontsize=11)
        ax1.set_title("Inter-animal distance", fontsize=14)
        ax1.tick_params(labelbottom=False); _light(ax1); ax1.axvline(0, lw=2.0, alpha=0.35)
        if legend_outside:
            ax1.legend(handles=[l3d], loc="center left", bbox_to_anchor=(1.02, 0.5),
                       frameon=False, fontsize=11)

        # (3) COM |Δz|
        ax2.plot(tloc, com_dz_loc, lw=2.4, color=col_vert)
        ax2.set_ylabel("|Δz| (mm)", fontsize=11)
        ax2.set_title("COM vertical separation", fontsize=14)
        ax2.tick_params(labelbottom=False); _light(ax2); ax2.axvline(0, lw=2.0, alpha=0.35)

        # (4) Snout–snout distance
        ax3.plot(tloc, snout_loc, lw=2.4, color=col_dist)
        ax3.set_ylabel("Snout dist (mm)", fontsize=11)
        ax3.set_title("Snout–snout distance" + (" (3D)" if snout_is3d else " (2D)"), fontsize=14)
        ax3.tick_params(labelbottom=False); _light(ax3); ax3.axvline(0, lw=2.0, alpha=0.35)

        # (5) Snout |Δz|
        ax4.plot(tloc, snout_dz_loc, lw=2.4, color=col_vert)
        ax4.set_ylabel("|Δz| (mm)", fontsize=11)
        ax4.set_title("Snout vertical separation", fontsize=14)
        ax4.tick_params(labelbottom=False); _light(ax4); ax4.axvline(0, lw=2.0, alpha=0.35)

        # (6) Pitch
        lp1, = ax5.plot(tloc, pitch_a1[w], lw=2.2, label="A pitch")
        lp2, = ax5.plot(tloc, pitch_a2[w], lw=2.2, label="B pitch")
        ax5.set_ylabel("Pitch (deg)", fontsize=11)
        ax5.set_title("Head pitch", fontsize=14)
        ax5.tick_params(labelbottom=False); _light(ax5); ax5.axvline(0, lw=2.0, alpha=0.35)
        if legend_outside and show_orientation_legends:
            ax5.legend(handles=[lp1,lp2], loc="center left", bbox_to_anchor=(1.02, 0.5),
                       frameon=False, fontsize=11)

        # (7) Yaw
        ly1, = ax6.plot(tloc, yaw_a1[w],  lw=2.2, label="A yaw")
        ly2, = ax6.plot(tloc, yaw_a2[w],  lw=2.2, label="B yaw")
        ax6.set_xlabel("Time from event (s)", fontsize=11)
        ax6.set_ylabel("Yaw (deg)", fontsize=11)
        ax6.set_title("Head yaw (global XY)", fontsize=14)
        _light(ax6); ax6.axvline(0, lw=2.0, alpha=0.35)
        if legend_outside and show_orientation_legends:
            ax6.legend(handles=[ly1,ly2], loc="center left", bbox_to_anchor=(1.02, 0.5),
                       frameon=False, fontsize=11)

        fig.align_ylabels(axes)

        if save:
            os.makedirs(od, exist_ok=True)
            fig.savefig(os.path.join(od, f"event_{k:03d}_all_panels.png"), dpi=300)
        plt.show(); plt.close(fig)


def plot_incident_windows_newset_comb(
    merged: pd.DataFrame,
    frames: pd.DataFrame,
    mask: Union[pd.Series, np.ndarray],
    *,
    pre_s: float = 2.0,
    post_s: float = 2.0,
    approach_thresh_mm: float = 120.0,
    contact_thresh_mm: float = 50.0,
    variance_drop_pct: float = 5.0,
    heatmap_downsample: int = 1,
    heatmap_cmap: str = "viridis",
    title_prefix: Optional[str] = None,
    save: bool = False,
    out_dir: Optional[str] = None,
    legend_outside: bool = True,
    show_colorbar: bool = True,
    show_orientation_legends: bool = False,  # NEW: control A/B legends
    event_indices: Optional[list] = None,  # NEW: select which events to plot
):
    """
    Combined version: all events side-by-side in one figure.
    Better for comparing events and makes 2-second windows look natural.
    
    Args:
        event_indices: List of event numbers to plot (e.g., [1,2,3]). 
                      None = plot all events.
    """
    # time
    t_merged_s, _ = _time_from_index_or_col(merged)
    t_frames_s, _ = _time_from_index_or_col(frames)
    if t_merged_s.size == 0 or t_frames_s.size == 0:
        print("[WARN] empty time vectors"); return
    t_merged_ms = t_merged_s * 1000.0

    # metrics
    try:
        comp = compute_com_distance(
            merged, p1="com1", p2="com2",
            smooth_window=1, dist_smooth_window=1, return_components=True
        )
    except NameError:
        raise RuntimeError("compute_com_distance(...) not found; import it first.")
    d3d = comp["dist_mm"].to_numpy(float)
    com_dz = _com_dz_abs(merged)
    snout_d, snout_is3d = _snout_dist_3d(merged)
    snout_dz = _snout_dz_abs(merged)

    # neural + orientation
    A_full, order, _ = _neuron_matrix(merged, variance_drop_pct)
    have_neurons = (A_full.size > 0)
    pitch_a1, yaw_a1 = _pitch_yaw_deg(merged, "a1")
    pitch_a2, yaw_a2 = _pitch_yaw_deg(merged, "a2")

    # segments
    segs_fe = _segments_from_mask(mask)
    if segs_fe.size == 0:
        print("[INFO] mask has no True segments"); return
    seg_time_ms = _map_frame_segments_to_time(frames, segs_fe, col="timestamp_ms_mini")

    # style helpers
    def _light(ax):
        ax.grid(False)
        for s in ax.spines.values():
            s.set_linewidth(0.6); s.set_color("0.5")

    col_dist, col_vert = "C0", "C3"
    od = out_dir or "."

    # Collect all valid events first
    event_data = []
    for k,(t0_ms, t1_ms) in enumerate(seg_time_ms, start=1):
        if t1_ms < t0_ms: t0_ms, t1_ms = t1_ms, t0_ms
        i0 = int(np.searchsorted(t_merged_ms, t0_ms, side="left"))
        i1 = int(np.searchsorted(t_merged_ms, t1_ms, side="right")) - 1
        i0 = max(0, min(i0, len(t_merged_s)-1))
        i1 = max(0, min(i1, len(t_merged_s)-1))
        if i1 <= i0: continue

        seg = slice(i0, i1+1)
        if not np.any(np.isfinite(d3d[seg])): continue
        rep_idx = i0 + int(np.nanargmin(d3d[seg]))
        rep_t = float(t_merged_s[rep_idx])
        w = _window_bounds(t_merged_s, rep_t, pre_s, post_s)

        tloc = t_merged_s[w] - rep_t
        d3d_loc, com_dz_loc = d3d[w], com_dz[w]
        snout_loc, snout_dz_loc = snout_d[w], snout_dz[w]

        if have_neurons:
            A_loc = A_full[order, w]
            if heatmap_downsample > 1 and A_loc.shape[1] > 1:
                step = int(heatmap_downsample)
                A_plot = A_loc[:, ::step]; t_plot = tloc[::step]
            else:
                A_plot, t_plot = A_loc, tloc
        else:
            A_plot = np.empty((0, len(tloc))); t_plot = tloc

        event_data.append({
            'k': k,
            'w': w,
            'tloc': tloc,
            'd3d_loc': d3d_loc,
            'com_dz_loc': com_dz_loc,
            'snout_loc': snout_loc,
            'snout_dz_loc': snout_dz_loc,
            'A_plot': A_plot,
            't_plot': t_plot
        })

    if not event_data:
        print("[INFO] No valid events to plot"); return

    # ===== FILTER EVENTS if requested =====
    if event_indices is not None:
        event_data = [evt for evt in event_data if evt['k'] in event_indices]
        if not event_data:
            print(f"[INFO] No events found with indices {event_indices}"); return

    n_events = len(event_data)
    
    # ===== COMPUTE GLOBAL COLOR LIMITS for uniform heatmaps =====
    if have_neurons:
        all_heatmap_data = [evt['A_plot'] for evt in event_data if evt['A_plot'].size > 0]
        if all_heatmap_data:
            global_vmin = min(arr.min() for arr in all_heatmap_data)
            global_vmax = max(arr.max() for arr in all_heatmap_data)
            # Make symmetric around zero for z-scored data
            global_vlim = max(abs(global_vmin), abs(global_vmax))
            global_vmin, global_vmax = -global_vlim, global_vlim
        else:
            global_vmin, global_vmax = -2, 2  # default
    else:
        global_vmin, global_vmax = -2, 2
    
    # ================= Multi-column figure (all events side-by-side) =================
    # Each event gets ~4 inches width, making 2s windows look more natural
    fig_width = min(4 * n_events, 20)  # Cap at 20 inches total
    fig = plt.figure(figsize=(fig_width, 11))
    
    ratios = [2.0, 1.2, 0.9, 1.0, 0.9, 0.9, 0.9]
    n_rows = len(ratios)
    
    # Create grid: n_rows x n_events
    gs = fig.add_gridspec(nrows=n_rows, ncols=n_events, height_ratios=ratios,
                         hspace=0.35, wspace=0.25)
    
    # Adjust margins
    fig.subplots_adjust(left=0.08, right=0.96, top=0.94, bottom=0.06)

    # Plot each event in its own column
    for col_idx, evt in enumerate(event_data):
        k = evt['k']
        tloc = evt['tloc']
        d3d_loc = evt['d3d_loc']
        com_dz_loc = evt['com_dz_loc']
        snout_loc = evt['snout_loc']
        snout_dz_loc = evt['snout_dz_loc']
        A_plot = evt['A_plot']
        t_plot = evt['t_plot']
        w = evt['w']
        
        # Create subplots for this column
        axes = [fig.add_subplot(gs[row, col_idx]) for row in range(n_rows)]
        ax0, ax1, ax2, ax3, ax4, ax5, ax6 = axes

        # (1) ΔF/F heatmap with UNIFORM colors
        if A_plot.size > 0 and A_plot.shape[1] == t_plot.shape[0]:
            pcm = ax0.pcolormesh(
                t_plot, np.arange(A_plot.shape[0]+1),
                np.vstack([A_plot, A_plot[-1:]]) if A_plot.shape[0] else np.zeros((1, A_plot.shape[1])),
                cmap=heatmap_cmap, shading="auto",
                vmin=global_vmin, vmax=global_vmax  # <-- UNIFORM COLOR SCALE
            )
            if col_idx == 0:
                ax0.set_title(f"Event {k}", fontsize=12, pad=8) #\nNeuron Activity
            else:
                ax0.set_title(f"Event {k}", fontsize=12, pad=8)
            
            if col_idx == 0:
                ax0.set_ylabel("Neurons", fontsize=10)

            # Show colorbar on last event only
            if show_colorbar and col_idx == n_events - 1:
                cb = fig.colorbar(pcm, ax=ax0, fraction=0.04, pad=0.02)
                cb.set_label("ΔF/F", rotation=90, fontsize=9)
                cb.outline.set_linewidth(0.6)
                cb.ax.tick_params(labelsize=8, width=0.6)
        else:
            ax0.text(0.5,0.5,"No data",ha="center",va="center",transform=ax0.transAxes, fontsize=9)
            ax0.set_title(f"Event {k}", fontsize=12, pad=8)
        ax0.tick_params(labelbottom=False, labelsize=9); _light(ax0); ax0.axvline(0, lw=1.5, alpha=0.35)

        # (2) COM 3D
        ax1.plot(tloc, d3d_loc, lw=2.2, color=col_dist)
        ax1.axhline(approach_thresh_mm, ls="--", lw=1.0, color="0.5")
        ax1.axhline(contact_thresh_mm,  ls=":",  lw=1.0, color="0.5")
        if col_idx == 0:
            ax1.set_ylabel("Distance (mm)", fontsize=10)
        ax1.tick_params(labelbottom=False, labelsize=9); _light(ax1); ax1.axvline(0, lw=1.5, alpha=0.35)

        # (3) COM |Δz|
        ax2.plot(tloc, com_dz_loc, lw=2.0, color=col_vert)
        if col_idx == 0:
            ax2.set_ylabel("|Δz| (mm)", fontsize=10)
        ax2.tick_params(labelbottom=False, labelsize=9); _light(ax2); ax2.axvline(0, lw=1.5, alpha=0.35)

        # (4) Snout–snout distance
        ax3.plot(tloc, snout_loc, lw=2.0, color=col_dist)
        if col_idx == 0:
            ax3.set_ylabel("Snout (mm)", fontsize=10)
        ax3.tick_params(labelbottom=False, labelsize=9); _light(ax3); ax3.axvline(0, lw=1.5, alpha=0.35)

        # (5) Snout |Δz|
        ax4.plot(tloc, snout_dz_loc, lw=2.0, color=col_vert)
        if col_idx == 0:
            ax4.set_ylabel("|Δz| (mm)", fontsize=10)
        ax4.tick_params(labelbottom=False, labelsize=9); _light(ax4); ax4.axvline(0, lw=1.5, alpha=0.35)

        # (6) Pitch - no legend unless requested
        ax5.plot(tloc, pitch_a1[w], lw=1.8, label="A" if col_idx==0 else "")
        ax5.plot(tloc, pitch_a2[w], lw=1.8, label="B" if col_idx==0 else "")
        if col_idx == 0:
            ax5.set_ylabel("Pitch (°)", fontsize=10)
            if show_orientation_legends:
                ax5.legend(fontsize=8, frameon=False, loc='upper left')
        ax5.tick_params(labelbottom=False, labelsize=9); _light(ax5); ax5.axvline(0, lw=1.5, alpha=0.35)

        # (7) Yaw - no legend unless requested
        ax6.plot(tloc, yaw_a1[w],  lw=1.8, label="A" if col_idx==0 else "")
        ax6.plot(tloc, yaw_a2[w],  lw=1.8, label="B" if col_idx==0 else "")
        ax6.set_xlabel("Time (s)", fontsize=10)
        if col_idx == 0:
            ax6.set_ylabel("Yaw (°)", fontsize=10)
            if show_orientation_legends:
                ax6.legend(fontsize=8, frameon=False, loc='upper left')
        ax6.tick_params(labelsize=9); _light(ax6); ax6.axvline(0, lw=1.5, alpha=0.35)

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14, y=0.98)

    if save:
        os.makedirs(od, exist_ok=True)
        fig.savefig(os.path.join(od, "all_events_comparison.png"), dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(fig)


# ==============================================================================
# Additional helper functions for plot_incident_windows_newset
# ==============================================================================

def _window_bounds(t_vec_s, t0_s, pre_s, post_s):
    """Return slice for time window [t0-pre, t0+post]."""
    i0, i1 = _search_window_indices(t_vec_s, t0_s, pre_s, post_s)
    return slice(i0, i1+1)

def _com_dz_abs(df):
    """Return |dz| between com1 and com2."""
    if "com1_z" in df.columns and "com2_z" in df.columns:
        return np.abs(df["com2_z"].to_numpy() - df["com1_z"].to_numpy())
    return np.zeros(len(df))

def _snout_dist_3d(df):
    """Return snout-snout distance (3D if possible, else 2D)."""
    d, is3d = _snout_distance(df, smooth_window=1)
    if d is None:
        return np.zeros(len(df)), False
    return d, is3d

def _snout_dz_abs(df):
    """Return |dz| between snout keypoints."""
    if "kp3_z_a1" in df.columns and "kp3_z_a2" in df.columns:
        return np.abs(df["kp3_z_a2"].to_numpy() - df["kp3_z_a1"].to_numpy())
    return np.zeros(len(df))

def _neuron_matrix(df, variance_drop_pct=5.0):
    """Build clustered z-scored neural matrix."""
    roi_cols = [c for c in df.columns if c.startswith("dF_F_roi")]
    if not roi_cols:
        return np.empty((0, len(df))), np.arange(0), []
    
    A = df[roi_cols].to_numpy(float).T
    var = np.nanvar(A, axis=1)
    if 0 < variance_drop_pct < 100 and A.shape[0] > 1:
        thresh = np.nanpercentile(var, variance_drop_pct)
        keep = var > thresh
        if keep.any():
            A = A[keep]
            roi_cols = [c for i,c in enumerate(roi_cols) if keep[i]]
    
    A = zscore(A, axis=1, nan_policy='omit')
    A = np.nan_to_num(A, 0.0)
    
    if A.shape[0] >= 2:
        try:
            Z = linkage(A, method='ward')
            order = leaves_list(Z)
        except:
            order = np.argsort(-np.nanvar(A, axis=1))
    else:
        order = np.arange(A.shape[0])
    
    return A, order, roi_cols

def _egocentric_bearing_elev(df, animal="a1"):
    """Compute bearing and elevation from animal's head frame."""
    # Simplified - returns NaN arrays if columns missing
    n = len(df)
    return np.full(n, np.nan), np.full(n, np.nan)

def _pitch_yaw_deg(df, animal="a1"):
    """
    Head orientation from a simple head axis: SpineF (#4) → Snout (#3).
    Pitch: angle of that axis vs horizontal plane (deg).
    Yaw: angle of XY projection (deg).
    """
    n = len(df)
    need = [f"kp4_x_{animal}", f"kp4_y_{animal}", f"kp4_z_{animal}",
            f"kp3_x_{animal}", f"kp3_y_{animal}", f"kp3_z_{animal}"]
    if not all(c in df.columns for c in need):
        return np.full(n, np.nan), np.full(n, np.nan)

    x4 = df[f"kp4_x_{animal}"].to_numpy(float)
    y4 = df[f"kp4_y_{animal}"].to_numpy(float)
    z4 = df[f"kp4_z_{animal}"].to_numpy(float)
    x3 = df[f"kp3_x_{animal}"].to_numpy(float)
    y3 = df[f"kp3_y_{animal}"].to_numpy(float)
    z3 = df[f"kp3_z_{animal}"].to_numpy(float)

    vx, vy, vz = (x3 - x4), (y3 - y4), (z3 - z4)           # SpineF→Snout
    hyp = np.hypot(vx, vy)
    pitch = np.degrees(np.arctan2(vz, hyp))                # +up/-down
    yaw   = np.degrees(np.arctan2(vy, vx))                 # global XY
    return pitch, yaw