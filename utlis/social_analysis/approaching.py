from typing import Optional, List, Dict, Tuple, Union
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


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

################ socail definition  



def _boolean_runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if mask.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    m = mask.astype(np.int8)
    edges = np.diff(np.pad(m, (1, 1), constant_values=0))
    starts = np.flatnonzero(edges == 1)
    ends   = np.flatnonzero(edges == -1)
    return starts, ends

#this is great, but it only counts until the threshold, i would like to include the things even smaller than threshold too.
# def find_approach_success(
#     frames: pd.DataFrame,
#     contact_mm: float = 50.0,     # "touch" threshold
#     dD_dt_thresh: float = 0.0,    # <= 0 means distance not increasing
#     min_len: int = 10,            # min frames from start of closing to contact
#     min_drop_mm: float = 10.0     # require at least this much net distance drop
# ) -> Tuple[np.ndarray, List[Dict]]:
#     """
#     Returns:
#       mask  : boolean array marking frames that belong to approach_success events
#       events: list of dicts with indices and metrics for each event

#     Uses only frames['dist_mm'] and frames['dD_dt'].
#     """
#     dist = frames["dist_mm"].to_numpy()
#     closing = frames["dD_dt"].to_numpy() <= float(dD_dt_thresh)
#     contact = dist <= float(contact_mm)

#     s_close, e_close = _boolean_runs(closing)

#     mask = np.zeros(len(frames), dtype=bool)
#     events: List[Dict] = []

#     for s, e in zip(s_close, e_close):
#         # first contact inside this closing run (if any)
#         within = np.flatnonzero(contact[s:e])
#         if within.size == 0:
#             continue
#         k = s + int(within[0])  # first contact index

#         # length and net drop checks
#         length_ok = (k - s + 1) >= int(min_len)
#         drop_ok   = (dist[s] - dist[k]) >= float(min_drop_mm)
#         if not (length_ok and drop_ok):
#             continue

#         mask[s:k+1] = True
#         events.append({
#             "start_idx": int(s),
#             "end_idx_exclusive": int(k + 1),
#             "contact_idx": int(k),
#             "start_dist_mm": float(dist[s]),
#             "end_dist_mm": float(dist[k]),
#             "drop_mm": float(dist[s] - dist[k]),
#             "duration_frames": int(k - s + 1),
#         })

#     return mask, events


def find_approach_success(
    frames: pd.DataFrame,
    contact_mm: float = 50.0,        # threshold to define first contact
    dD_dt_thresh: float = 0.0,       # closing if dD_dt <= this
    min_len: int = 10,               # frames from run start to first contact
    min_drop_mm: float = 10.0,       # net drop (start -> min before increase)
    sep_consec: int = 2,             # require this many consecutive "increasing" frames
    dD_dt_sep_thresh: float = 0.0    # "increasing" if dD_dt > this
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Mark approach_success events that:
      - start at the beginning of a closing run (dD_dt <= dD_dt_thresh),
      - hit first contact (dist <= contact_mm),
      - END at the first sustained increase of distance after contact
        (dD_dt > dD_dt_sep_thresh for `sep_consec` consecutive frames).
      - If no increase occurs inside the closing run, end at the run end.

    Returns:
      mask   : boolean array for frames inside any event
      events : list of dicts with indices and metrics
    """
    dist = frames["dist_mm"].to_numpy()
    dD_dt = frames["dD_dt"].to_numpy()

    closing = dD_dt <= float(dD_dt_thresh)
    contact = dist   <= float(contact_mm)
    increasing = dD_dt > float(dD_dt_sep_thresh)

    s_close, e_close = _boolean_runs(closing)

    n = len(frames)
    mask = np.zeros(n, dtype=bool)
    events: List[Dict] = []

    for s, e in zip(s_close, e_close):
        # First contact within this closing run
        within = np.flatnonzero(contact[s:e])
        if within.size == 0:
            continue
        k = s + int(within[0])  # first contact index

        # Find first sustained increase after contact: sep_consec consecutive 'increasing'
        r = e  # default to the end of this closing run
        if k < e - 1:
            # Build a rolling window of length sep_consec over 'increasing'
            # We want the first index u in [k, e - sep_consec + 1) where all True
            inc_seg = increasing[k:e].astype(np.uint8)
            if sep_consec <= inc_seg.size:
                # convolution to detect consecutive True of length sep_consec
                window_sum = np.convolve(inc_seg, np.ones(sep_consec, dtype=np.uint8), mode="valid")
                found = np.flatnonzero(window_sum == sep_consec)
                if found.size > 0:
                    u_rel = int(found[0])         # relative to k
                    r = k + u_rel                 # event ends *at* the first increase
        # Sanity guard
        r = max(k + 1, min(r, e))  # ensure at least one frame after k and within [s,e]

        # Checks
        length_ok = (k - s + 1) >= int(min_len)
        min_dist_segment = float(np.min(dist[s:r]))
        drop_ok = (float(dist[s]) - min_dist_segment) >= float(min_drop_mm)
        if not (length_ok and drop_ok):
            continue

        # Mark and collect metrics
        mask[s:r] = True
        bottom_idx = int(s + int(np.argmin(dist[s:r])))

        events.append({
            "start_idx": int(s),
            "end_idx_exclusive": int(r),
            "contact_idx": int(k),
            "bottom_idx": int(bottom_idx),
            "start_dist_mm": float(dist[s]),
            "contact_dist_mm": float(dist[k]),
            "min_dist_mm": float(min_dist_segment),
            "end_dist_mm": float(dist[r-1]),
            "drop_mm": float(dist[s] - min_dist_segment),
            "duration_frames_to_contact": int(k - s + 1),
            "duration_frames_total": int(r - s),
            "increasing_guard": {
                "sep_consec": int(sep_consec),
                "dD_dt_sep_thresh": float(dD_dt_sep_thresh),
            },
        })

    return mask, events


#####################################below is vis on point to point distances #####################################

# zero-based skeleton edges (your list)
MOUSE22_EDGES = [
    (0,1),(1,2),(0,2),
    (0,3),(1,3),(2,3),
    (3,4),(4,5),(5,6),(6,7),
    (8,9),(9,10),(10,11),(11,3),
    (12,13),(13,14),(14,15),(15,3),
    (16,17),(17,18),(18,4),
    (19,20),(20,21),(21,4),
]

# simple name→index (1-based) for convenience
KP = {
    'EarL':1,'EarR':2,'Snout':3,'SpineF':4,'SpineM':5,
    'Tail(base)':6,'Tail(mid)':7,'Tail(end)':8,
    'ForepawL':9,'WristL':10,'ElbowL':11,'ShoulderL':12,
    'ForepawR':13,'WristR':14,'ElbowR':15,'ShoulderR':16,
    'HindpawL':17,'AnkleL':18,'KneeL':19,'HindpawR':20,'AnkleR':21,'KneeR':22
}

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
