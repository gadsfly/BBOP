"""
3D Keypoint Social Interaction Analysis Module

Author: Mir Qi
Date: October 2024

This module provides functions for analyzing 3D keypoint data from multi-animal
behavioral recordings, including:
- Distance calculations between animals and body parts
- Motion direction and velocity analysis  
- Approach detection and social interaction events
- Visualization of skeletons, trajectories, and heatmaps
"""

from typing import Optional, Union, Tuple, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================================
# CONSTANTS & CONFIGURATIONS
# ==============================================================================

# Mouse skeleton connectivity (22 keypoints, 0-indexed)
MOUSE22_EDGES = [
    (0,1),(1,2),(0,2),                          # Head triangle
    (0,3),(1,3),(2,3),                          # Head to spine
    (3,4),(4,5),(5,6),(6,7),                    # Spine to tail
    (8,9),(9,10),(10,11),(11,3),                # Left forelimb
    (12,13),(13,14),(14,15),(15,3),             # Right forelimb
    (16,17),(17,18),(18,4),                     # Left hindlimb
    (19,20),(20,21),(21,4),                     # Right hindlimb
]

# Keypoint name to index mapping (1-based indexing)
KP = {
    'EarL':1, 'EarR':2, 'Snout':3, 'SpineF':4, 'SpineM':5,
    'Tail(base)':6, 'Tail(mid)':7, 'Tail(end)':8,
    'ForepawL':9, 'WristL':10, 'ElbowL':11, 'ShoulderL':12,
    'ForepawR':13, 'WristR':14, 'ElbowR':15, 'ShoulderR':16,
    'HindpawL':17, 'AnkleL':18, 'KneeL':19, 
    'HindpawR':20, 'AnkleR':21, 'KneeR':22
}


# ==============================================================================
# BASIC DISTANCE & GEOMETRY
# ==============================================================================

def distance3d(x1, y1, z1, x2, y2, z2):
    """
    Compute 3D Euclidean distance between two points.
    
    Parameters
    ----------
    x1, y1, z1 : array-like
        Coordinates of first point(s)
    x2, y2, z2 : array-like
        Coordinates of second point(s)
    
    Returns
    -------
    np.ndarray
        Euclidean distances in 3D space
    """
    x1, y1, z1, x2, y2, z2 = map(np.asarray, (x1, y1, z1, x2, y2, z2))
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


def compute_com_distance(
    df: pd.DataFrame,
    p1: str = "com1",
    p2: str = "com2",
    smooth_window: Optional[int] = None,
    dist_smooth_window: Optional[int] = None,
    return_components: bool = False,
):
    """
    Compute distance between two center-of-mass (COM) positions.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing COM columns (e.g., com1_x, com1_y, com1_z)
    p1, p2 : str
        Prefix for the two COM entities
    smooth_window : int, optional
        Window size for smoothing positions before distance calculation
    dist_smooth_window : int, optional
        Additional smoothing applied to the distance itself
    return_components : bool
        If True, return DataFrame with dx, dy, dz, dist_mm columns
    
    Returns
    -------
    pd.Series or pd.DataFrame
        Distance in mm, or components if return_components=True
    """
    cols = [f"{p1}_x", f"{p1}_y", f"{p1}_z", f"{p2}_x", f"{p2}_y", f"{p2}_z"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Optional position smoothing
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


# ==============================================================================
# MOTION & VELOCITY ANALYSIS
# ==============================================================================

def _time_seconds(df: pd.DataFrame, time_col: Optional[str], fps: Optional[float]) -> Tuple[np.ndarray, str]:
    """
    Convert time column to seconds or build time array from FPS.
    
    Returns
    -------
    tuple
        (time_array_in_seconds, units_string)
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
    fill_na: bool = False,
    eps: float = 1e-9
) -> pd.DataFrame:
    """
    Compute velocity, speed, and unit direction vectors from position data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain {prefix}_x, {prefix}_y, {prefix}_z columns
    prefix : str
        Column prefix for the entity (e.g., "com1", "com2")
    time_col : str, optional
        Column name containing timestamps
    fps : float, optional
        Frames per second (if no time column provided)
    pos_smooth : int
        Smoothing window for positions before differentiation
    vel_smooth : int
        Smoothing window for velocities after differentiation
    fill_na : bool
        Whether to interpolate NaN values
    eps : float
        Small value to avoid division by zero
    
    Returns
    -------
    pd.DataFrame
        Contains vx, vy, vz, speed, ux, uy, uz columns
        Units are mm/s if time in seconds, else mm/frame
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
    vel[f"{prefix}_speed"] = speed

    # Unit direction vectors
    speed_nz = np.maximum(speed.to_numpy(), eps)
    vel[f"{prefix}_ux"] = vel[f"{prefix}_vx"] / speed_nz
    vel[f"{prefix}_uy"] = vel[f"{prefix}_vy"] / speed_nz
    vel[f"{prefix}_uz"] = vel[f"{prefix}_vz"] / speed_nz

    vel.attrs["velocity_units"] = "mm/s" if t_units == "s" else "mm/frame"
    return vel


# ==============================================================================
# APPROACH DETECTION
# ==============================================================================

def _boolean_runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find start and end indices of True runs in a boolean array.
    
    Returns
    -------
    tuple
        (start_indices, end_indices_exclusive)
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return starts, ends


def detect_approaches(
    df: pd.DataFrame,
    p1: str = "com1",
    p2: str = "com2",
    time_col: Optional[str] = "timestamp_ms_mini",
    fps: Optional[float] = None,
    pos_smooth: int = 5,
    vel_smooth: int = 5,
    radial_thresh: float = 20.0,
    speed_min: float = 5.0,
    dist_min: Optional[float] = None,
    dist_max: Optional[float] = 300.0,
    min_samples: Optional[int] = 15,
    return_intervals: bool = True,
    eps: float = 1e-9
) -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Detect approach behaviors between two animals based on radial velocity.
    
    An approach is defined as movement toward the other animal with sufficient
    speed, within a specified distance range, sustained for minimum duration.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain COM positions for both animals
    p1, p2 : str
        Prefixes for the two animals (e.g., "com1", "com2")
    time_col : str, optional
        Column with timestamps
    fps : float, optional
        Frames per second if no time column
    pos_smooth : int
        Position smoothing window
    vel_smooth : int
        Velocity smoothing window
    radial_thresh : float
        Minimum radial velocity toward other (mm/s) to count as approach
    speed_min : float
        Minimum total speed (mm/s) required
    dist_min : float, optional
        Minimum inter-animal distance to consider (avoids collision artifacts)
    dist_max : float, optional
        Maximum distance for interaction zone
    min_samples : int, optional
        Minimum consecutive frames for valid approach event
    return_intervals : bool
        If True, return interval summaries in addition to per-frame data
    eps : float
        Small value for numerical stability
    
    Returns
    -------
    dict
        'frames': pd.DataFrame with per-frame metrics and boolean flags
        'intervals': dict of approach event summaries (if return_intervals=True)
    """
    # 1) Distance components & unit vector between animals
    comps = compute_com_distance(df, p1=p1, p2=p2, smooth_window=pos_smooth, return_components=True)
    dx, dy, dz = comps["dx"].to_numpy(), comps["dy"].to_numpy(), comps["dz"].to_numpy()
    dist = comps["dist_mm"].to_numpy()
    u12 = np.column_stack([dx, dy, dz])
    denom = np.maximum(dist, eps)[:, None]
    u12 = u12 / denom

    # 2) Compute velocities & speeds for both animals
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

    # 3) Radial velocity components (toward the other animal)
    radial1 = (v1 * u12).sum(axis=1)     # Animal 1 toward animal 2
    radial2 = (v2 * (-u12)).sum(axis=1)  # Animal 2 toward animal 1

    # 4) Distance rate of change
    t_sec, t_units = _time_seconds(df, time_col, fps)
    dD_dt = np.gradient(dist, t_sec)

    # 5) Define approach flags with distance gating
    in_band = np.ones_like(dist, dtype=bool)
    if dist_min is not None:
        in_band &= (dist >= float(dist_min))
    if dist_max is not None:
        in_band &= (dist <= float(dist_max))

    approach1 = (radial1 >= radial_thresh) & (speed1 >= speed_min) & in_band
    approach2 = (radial2 >= radial_thresh) & (speed2 >= speed_min) & in_band
    mutual = approach1 & approach2

    # 6) Minimum duration filter
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

    # 7) Assemble per-frame results
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
    
    # Pass through alignment columns if present
    for c in ("camera_frame_sixcam", "mapped_sixcam_frame_indices"):
        if c in df.columns:
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


# ==============================================================================
# KEYPOINT UTILITIES
# ==============================================================================

def _kp_idx(k):
    """Convert keypoint name or number to integer index."""
    return KP[k] if isinstance(k, str) else int(k)


def get_proximity_rows_by_com(df, threshold_mm=260.0, p1="com1", p2="com2", **kwargs):
    """
    Return DataFrame index where COM distance <= threshold_mm.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with COM position data
    threshold_mm : float
        Maximum distance for proximity
    p1, p2 : str
        Animal prefixes
    **kwargs
        Additional arguments passed to compute_com_distance
    
    Returns
    -------
    pd.Index
        Indices where animals are in proximity
    """
    d = compute_com_distance(df, p1=p1, p2=p2, **kwargs)
    return d.index[d <= float(threshold_mm)]


def point_distance(df, kp_a1, kp_a2, a1="a1", a2="a2"):
    """
    Compute distance between any two keypoints across animals.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with keypoint columns
    kp_a1 : str or int
        Keypoint name (e.g., "Snout") or index for animal 1
    kp_a2 : str or int
        Keypoint name or index for animal 2
    a1, a2 : str
        Animal identifiers in column names
    
    Returns
    -------
    pd.Series
        Euclidean distance in mm between the two keypoints
    """
    i1 = _kp_idx(kp_a1)
    i2 = _kp_idx(kp_a2)

    # Auto-detect available axes
    axes = [ax for ax in ("x","y","z")
            if f"kp{i1}_{ax}_{a1}" in df.columns and f"kp{i2}_{ax}_{a2}" in df.columns]
    if not axes:
        raise ValueError("No matching kp columns found (check names/animals).")

    # Compute squared differences and take square root
    s = 0.0
    for ax in axes:
        a = df[f"kp{i1}_{ax}_{a1}"]
        b = df[f"kp{i2}_{ax}_{a2}"]
        s = s + (b - a)**2
    return (s**0.5).rename(f"kp{i1}_{a1}__to__kp{i2}_{a2}_mm")


def snout_to(df, others=("Snout","Tail(base)", 'SpineM'), rows=None, a1="a1", a2="a2"):
    """
    Compute distances from animal1's snout to multiple points on animal2.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with keypoint data
    others : tuple of str or int
        Keypoint names/indices on animal2 to measure to
    rows : pd.Index, optional
        Subset of rows to compute (e.g., proximity frames)
    a1, a2 : str
        Animal identifiers
    
    Returns
    -------
    pd.DataFrame
        One column per target keypoint with distances
    """
    idx = df.index if rows is None else rows
    out = {}
    for o in others:
        dist = point_distance(df.loc[idx], "Snout", o, a1=a1, a2=a2)
        o_lab = (o if isinstance(o,str) else f"kp{int(o)}").replace("(", "").replace(")", "").replace(" ", "").lower()
        out[f"snout_{a1}__to__{o_lab}_{a2}_mm"] = dist
    return pd.DataFrame(out, index=idx)


# ==============================================================================
# VISUALIZATION HELPERS
# ==============================================================================

def _kp_xy_row(df, idx, animal="a1"):
    """
    Extract (22, 2) array of x,y coordinates for all keypoints at a single frame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with keypoint columns
    idx : index value
        Frame index to extract
    animal : str
        Animal identifier ("a1", "a2", etc.)
    
    Returns
    -------
    np.ndarray
        Shape (22, 2) with x,y coordinates; row i (0-based) = keypoint i+1
    """
    r = df.loc[idx]
    xy = np.empty((22, 2), dtype=float)
    for k in range(1, 23):
        xy[k-1, 0] = r.get(f"kp{k}_x_{animal}", np.nan)
        xy[k-1, 1] = r.get(f"kp{k}_y_{animal}", np.nan)
    return xy


def _kp_xyz_row(df, idx, animal="a1"):
    """
    Extract (22, 3) array of x,y,z coordinates for all keypoints at a single frame.
    
    Returns
    -------
    np.ndarray
        Shape (22, 3) with x,y,z coordinates
    """
    r = df.loc[idx]
    xyz = np.empty((22, 3), dtype=float)
    for k in range(1, 23):
        xyz[k-1, 0] = r.get(f"kp{k}_x_{animal}", np.nan)
        xyz[k-1, 1] = r.get(f"kp{k}_y_{animal}", np.nan)
        xyz[k-1, 2] = r.get(f"kp{k}_z_{animal}", np.nan)
    return xyz


def _pair_dist(a_xy, b_xy, i_a, i_b):
    """
    Euclidean distance between point i_a of animal A and point i_b of animal B.
    
    Parameters
    ----------
    a_xy, b_xy : np.ndarray
        (22, 2) or (22, 3) arrays of keypoint coordinates
    i_a, i_b : int
        0-based keypoint indices
    
    Returns
    -------
    float
        Euclidean distance
    """
    diff = b_xy[i_b] - a_xy[i_a]
    return float(np.sqrt(np.sum(diff**2)))


def _time_label(idx_name, idx_value):
    """
    Format frame index as time label.
    
    Returns
    -------
    str
        Formatted label (e.g., "1.234s" or "frame 100")
    """
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
    threshold_mm=None,
    invert_y=False
):
    """
    Plot 2D skeleton poses for selected frames in a grid layout.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with keypoint data
    rows : array-like
        Frame indices to plot
    n : int
        Maximum number of frames to show
    animal1, animal2 : str
        Animal identifiers
    pairs : tuple of tuples
        Keypoint pairs to connect across animals (e.g., for distance visualization)
    threshold_mm : float, optional
        If provided, highlight pairs below this distance with thicker lines
    invert_y : bool
        Whether to invert y-axis (for image-style coordinates)
    """
    idx_sel = list(rows)[:int(n)]
    if not idx_sel:
        raise ValueError("No rows to plot.")

    # Compute global limits for consistent scaling
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

    # Create grid
    cols = min(5, max(1, len(idx_sel)))
    rows_grid = int(np.ceil(len(idx_sel) / cols))
    fig, axes = plt.subplots(rows_grid, cols, figsize=(4*cols, 4*rows_grid), squeeze=False)

    # Convert pair names to indices
    def _to_kp0(x):
        return (KP[x]-1) if isinstance(x, str) else (int(x)-1)
    pair_idx = [(_to_kp0(a), _to_kp0(b)) for (a,b) in pairs]

    # Plot each frame
    for k, idx in enumerate(idx_sel):
        r = k // cols
        c = k % cols
        ax = axes[r][c]
        a1 = _kp_xy_row(df, idx, animal=animal1)
        a2 = _kp_xy_row(df, idx, animal=animal2)

        # Draw skeleton edges
        for i, j in MOUSE22_EDGES:
            if not (np.any(np.isnan(a1[[i,j]])) or np.any(np.isnan(a2[[i,j]]))):
                ax.plot([a1[i,0], a1[j,0]], [a1[i,1], a1[j,1]], 'b-', linewidth=1, alpha=0.7)
                ax.plot([a2[i,0], a2[j,0]], [a2[i,1], a2[j,1]], 'r-', linewidth=1, alpha=0.7)

        # Draw keypoints
        ax.scatter(a1[:,0], a1[:,1], s=12, c='blue', alpha=0.7)
        ax.scatter(a2[:,0], a2[:,1], s=12, c='red', alpha=0.7)

        # Draw COM if present
        if f"com1_x" in df.columns and f"com1_y" in df.columns:
            ax.scatter(df.loc[idx, "com1_x"], df.loc[idx, "com1_y"], s=40, marker='x', c='blue', linewidths=2)
        if f"com2_x" in df.columns and f"com2_y" in df.columns:
            ax.scatter(df.loc[idx, "com2_x"], df.loc[idx, "com2_y"], s=40, marker='x', c='red', linewidths=2)

        # Draw cross-animal pair connections
        for ia, ib in pair_idx:
            if (not np.any(np.isnan(a1[ia])) and not np.any(np.isnan(a2[ib]))):
                d = _pair_dist(a1, a2, ia, ib)
                lw = 2.5 if (threshold_mm is not None and d <= float(threshold_mm)) else 1.0
                ax.plot([a1[ia,0], a2[ib,0]], [a1[ia,1], a2[ib,1]], 'g--', linewidth=lw, alpha=0.5)

        ax.set_title(_time_label(df.index.name, idx), fontsize=10)
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        if invert_y:
            ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    total = rows_grid * cols
    for k in range(len(idx_sel), total):
        r = k // cols
        c = k % cols
        axes[r][c].axis('off')

    plt.tight_layout()
    return fig