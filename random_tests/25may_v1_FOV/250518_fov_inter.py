#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.widgets import Slider

# --------------------------------------------------------------------------
# Load session DataFrame
# --------------------------------------------------------------------------
def load_session_data(rec_path):
    rec_path = Path(rec_path)
    aligned_dir = rec_path / "MIR_Aligned"
    h5_paths = list(aligned_dir.glob("aligned_predictions_with_ca_and_dF_F*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files found in {aligned_dir}")
    hdf5_file = h5_paths[0] if len(h5_paths) == 1 else max(h5_paths, key=lambda p: len(p.stem))
    print("Loading:", hdf5_file)
    df = pd.read_hdf(hdf5_file, key='df')
    return df

# --------------------------------------------------------------------------
# 2D circle fit (hull-based least squares)
# --------------------------------------------------------------------------
def fit_circle(x, y):
    pts = np.column_stack((x, y))
    hull = ConvexHull(pts)
    hx, hy = pts[hull.vertices].T
    def resid(p, xx, yy):
        return np.hypot(xx - p[0], yy - p[1]) - p[2]
    p0 = [hx.mean(), hy.mean(), np.mean(np.hypot(hx - hx.mean(), hy - hy.mean()))]
    res = least_squares(resid, p0, args=(hx, hy))
    return res.x  # xc, yc, r

# --------------------------------------------------------------------------
# 3D plane fit via SVD
# --------------------------------------------------------------------------
def fit_plane(points):
    C = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - C)
    n = Vt[-1]
    if n[2] < 0: n = -n
    return C, n

def plane_basis(n):
    v = np.array([1,0,0]) if abs(n[0])<1e-6 and abs(n[1])<1e-6 else np.array([0,0,1])
    b1 = np.cross(n, v); b1 /= np.linalg.norm(b1)
    b2 = np.cross(n, b1); b2 /= np.linalg.norm(b2)
    return b1, b2

# --------------------------------------------------------------------------
# Compute head frame and triangle
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
# Main interactive plotting
# --------------------------------------------------------------------------
if __name__ == '__main__':
    session_dir = "/data/big_rim/rsync_dcc_sum/Oct3V1/2024_10_14/20240916v1r1_16_37"

    df = load_session_data(session_dir)

    # Fit table circle (2D)
    x, y = df.com_x.to_numpy(), df.com_y.to_numpy()
    xc, yc, radius = fit_circle(x, y)
    theta = np.linspace(0, 2*np.pi, 200)
    circ_xy = np.column_stack((xc + radius*np.cos(theta),
                               yc + radius*np.sin(theta)))

    # Fit table plane (3D) using hindpaws
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
    b1, b2 = plane_basis(plane_n)
    circ3d = np.array([
        table_center + radius * (np.cos(t)*b1 + np.sin(t)*b2)
        for t in theta
    ])

    # Set up figure and slider
    fig = plt.figure(figsize=(10, 6))
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    plt.subplots_adjust(bottom=0.2)
    slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(slider_ax, 'Frame', 0, len(df)-1, valinit=0, valfmt='%0.0f')

    def update(val):
        idx = int(slider.val)
        row = df.iloc[idx]
        mid, R_head = head_frame(row)
        xh = R_head[:, 0]  # snout direction

        # 2D panel
        ax2d.clear()
        ax2d.scatter(df.com_x, df.com_y, s=2, alpha=0.2)
        ax2d.plot(circ_xy[:,0], circ_xy[:,1], 'r-')
        ax2d.scatter([table_center[0]], [table_center[1]], c='r')
        tri2d = np.array([
            [row.kp1_x, row.kp1_y],
            [row.kp2_x, row.kp2_y],
            [row.kp3_x, row.kp3_y],
            [row.kp1_x, row.kp1_y]
        ])
        ax2d.plot(tri2d[:,0], tri2d[:,1], '-k', lw=1.5)
        # FOV arrow in 2D
        fov2d = xh[:2]; nrm = np.linalg.norm(fov2d)
        if nrm > 1e-6:
            fov2d = fov2d / nrm * (radius * 0.5)
            ax2d.quiver(mid[0], mid[1], fov2d[0], fov2d[1],
                        angles='xy', scale_units='xy', scale=1, width=0.005)
        ax2d.set_aspect('equal')
        ax2d.set_title(f"Frame {idx}: 2D")

        # 3D panel
        ax3d.clear()
        ax3d.scatter(cand_pts[:,0], cand_pts[:,1], cand_pts[:,2],
                     c='gray', s=1)
        gx = np.linspace(cand_pts[:,0].min(), cand_pts[:,0].max(), 10)
        gy = np.linspace(cand_pts[:,1].min(), cand_pts[:,1].max(), 10)
        gx, gy = np.meshgrid(gx, gy)
        gz = (table_center[2]
              - (plane_n[0]*(gx-table_center[0])
                 + plane_n[1]*(gy-table_center[1]))
              / plane_n[2])
        ax3d.plot_surface(gx, gy, gz, alpha=0.3)
        ax3d.plot(circ3d[:,0], circ3d[:,1], circ3d[:,2], 'b-')
        # head triangle
        tri3d = np.array([
            [row.kp1_x, row.kp1_y, row.kp1_z],
            [row.kp2_x, row.kp2_y, row.kp2_z],
            [row.kp3_x, row.kp3_y, row.kp3_z],
            [row.kp1_x, row.kp1_y, row.kp1_z]
        ])
        ax3d.plot(tri3d[:,0], tri3d[:,1], tri3d[:,2], '-k', lw=1.5)
        # FOV arrow in 3D
        ax3d.quiver(
            mid[0], mid[1], mid[2],
            xh[0], xh[1], xh[2],
            length=radius*0.5, normalize=True, linewidth=2
        )
        ax3d.set_title("3D")

        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()
