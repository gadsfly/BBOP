#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --------------------------------------------------------------------------
# Your existing loader (unchanged)
# --------------------------------------------------------------------------
def load_session_data(rec_path):
    """
    Load a single session's HDF5 file by dynamically searching the MIR_Aligned folder.

    Selection logic:
      1) If exactly one file matches aligned_predictions_with_ca_and_dF_F*.h5, use it.
      2) Otherwise, pick the file with the longest stem (most extra info).
    """
    rec_path = Path(rec_path)
    aligned_dir = rec_path / "MIR_Aligned"

    h5_paths = list(aligned_dir.glob("aligned_predictions_with_ca_and_dF_F*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files found in {aligned_dir}")

    if len(h5_paths) == 1:
        hdf5_file_path = h5_paths[0]
    else:
        hdf5_file_path = max(h5_paths, key=lambda p: len(p.stem))

    print("Using:", hdf5_file_path)
    df = pd.read_hdf(hdf5_file_path, key='df')
    df['session_id']     = rec_path.name
    df['recording_date'] = rec_path.parent.name
    df['experiment']     = rec_path.parent.parent.name
    df['session_path']   = str(rec_path)
    df['file_path']      = str(hdf5_file_path)
    return df

# --------------------------------------------------------------------------
# Robust circle fit on hull points (hull-based least squares)
# --------------------------------------------------------------------------
def fit_circle(x, y):
    pts = np.column_stack((x, y))
    hull = ConvexHull(pts)
    hx, hy = pts[hull.vertices].T

    def resid(p, xx, yy):
        return np.hypot(xx - p[0], yy - p[1]) - p[2]

    p0 = [hx.mean(), hy.mean(), np.mean(np.hypot(hx - hx.mean(), hy - hy.mean()))]
    res = least_squares(resid, p0, args=(hx, hy))
    return res.x  # (xc, yc, r)

# --------------------------------------------------------------------------
# Plane fit via SVD
# --------------------------------------------------------------------------
def fit_plane(points):
    C = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - C)
    n = Vt[-1]
    if n[2] < 0:
        n = -n
    return C, n

# --------------------------------------------------------------------------
# Build an orthonormal basis for the table plane
# --------------------------------------------------------------------------
def plane_basis(n):
    if abs(n[0]) < 1e-6 and abs(n[1]) < 1e-6:
        v = np.array([1, 0, 0])
    else:
        v = np.array([0, 0, 1])
    b1 = np.cross(n, v); b1 /= np.linalg.norm(b1)
    b2 = np.cross(n, b1); b2 /= np.linalg.norm(b2)
    return b1, b2

# --------------------------------------------------------------------------
# Compute head coordinate frame (x = snout→ear_mid, etc.)
# --------------------------------------------------------------------------
def head_frame(row):
    eL = np.array([row.kp1_x, row.kp1_y, row.kp1_z])
    eR = np.array([row.kp2_x, row.kp2_y, row.kp2_z])
    sn = np.array([row.kp3_x, row.kp3_y, row.kp3_z])
    mid = (eL + eR) / 2.0

    xh = sn - mid; xh /= np.linalg.norm(xh)
    ty = eR - eL
    yh = ty - xh * (xh.dot(ty)); yh /= np.linalg.norm(yh)
    zh = np.cross(xh, yh)
    return mid, np.column_stack((xh, yh, zh))

# --------------------------------------------------------------------------
# Plot one frame: 2D COM + boundary circle + head direction arrow, 3D plane & head arrow
# --------------------------------------------------------------------------
def plot_frame(df, idx, circ_xy, table_center, plane_n, cand_pts, circ3d, radius):
    row = df.iloc[idx]
    mid, R_head = head_frame(row)
    xh = R_head[:, 0]                     # forward unit vector
    fov2d = xh[:2]; norm = np.linalg.norm(fov2d)
    if norm < 1e-6:
        fov2d_plot = np.zeros(2)
    else:
        fov2d_plot = (fov2d / norm) * (radius * 0.5)

    fig = plt.figure(figsize=(10, 5))

    # — Left: 2D
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(df.com_x, df.com_y, s=2, alpha=0.2, label='COM data')
    ax1.plot(circ_xy[:, 0], circ_xy[:, 1], 'r-', label='Table circle')
    ax1.scatter([table_center[0]], [table_center[1]], c='r', label='Center')
    ax1.quiver(
        mid[0], mid[1],
        fov2d_plot[0], fov2d_plot[1],
        angles='xy', scale_units='xy', scale=1,
        width=0.005, label='FOV dir'
    )
    ax1.set_aspect('equal')
    ax1.set_title(f"Frame {idx}: 2D view")
    ax1.legend(loc='upper right')

    # — Right: 3D
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(cand_pts[:, 0], cand_pts[:, 1], cand_pts[:, 2],
                c='gray', s=1, label='Paw contacts')
    # plane surface
    gx = np.linspace(cand_pts[:, 0].min(), cand_pts[:, 0].max(), 10)
    gy = np.linspace(cand_pts[:, 1].min(), cand_pts[:, 1].max(), 10)
    gx, gy = np.meshgrid(gx, gy)
    gz = (table_center[2]
          - (plane_n[0] * (gx - table_center[0]) + plane_n[1] * (gy - table_center[1]))
          / plane_n[2])
    ax2.plot_surface(gx, gy, gz, alpha=0.3)

    ax2.plot(circ3d[:, 0], circ3d[:, 1], circ3d[:, 2], 'b-', label='Table circle')

    # 3D FOV arrow
    arrow3d_end = mid + xh * (radius * 0.5)
    ax2.plot(
        [mid[0], arrow3d_end[0]],
        [mid[1], arrow3d_end[1]],
        [mid[2], arrow3d_end[2]],
        '-k', lw=2, label='FOV dir'
    )
    ax2.set_title("3D view")
    ax2.legend()

    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == '__main__':
    session_dir = "/data/big_rim/rsync_dcc_sum/Oct3V1/2024_10_14/20240916v1r1_16_37"
    df = load_session_data(session_dir)

    # 2D circle fit
    x, y = df.com_x.to_numpy(), df.com_y.to_numpy()
    xc, yc, radius = fit_circle(x, y)
    theta = np.linspace(0, 2 * np.pi, 200)
    circ_xy = np.column_stack((xc + radius * np.cos(theta),
                               yc + radius * np.sin(theta)))

    # 3D plane fit (use hindpaws 17 & 20)
    paw_ids = [17, 20]
    cand = []
    thr = 5.0
    for pid in paw_ids:
        zs = df[f'kp{pid}_z'].to_numpy()
        mask = zs <= zs.min() + thr
        pts = np.vstack((df[f'kp{pid}_x'][mask],
                         df[f'kp{pid}_y'][mask],
                         zs[mask])).T
        cand.append(pts)
    cand_pts = np.vstack(cand)
    table_center, plane_n = fit_plane(cand_pts)

    # project circle into 3D
    b1, b2 = plane_basis(plane_n)
    circ3d = np.array([
        table_center + radius * (np.cos(t) * b1 + np.sin(t) * b2)
        for t in theta
    ])

    # render example frames
    for idx in (0, len(df)//2, len(df)-1):
        plot_frame(df, idx, circ_xy, table_center, plane_n, cand_pts, circ3d, radius)
