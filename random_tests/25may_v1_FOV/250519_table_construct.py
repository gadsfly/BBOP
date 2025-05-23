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
    if n[2] < 0:
        n = -n
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
    # ← adjust to your session folder
    session_dir = "/data/big_rim/rsync_dcc_sum/Oct3V1/2024_10_14/20240916v1r1_16_37"
  
    df = load_session_data(session_dir)

    # 2D table circle fit (auto)
    x, y = df.com_x.to_numpy(), df.com_y.to_numpy()
    xc, yc, radius = fit_circle(x, y)
    theta = np.linspace(0, 2*np.pi, 200)
    circ_xy = np.column_stack((xc + radius*np.cos(theta),
                               yc + radius*np.sin(theta)))

    # Hindpaw contact points (for scatter only)
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

    # Manual table points
    manual_table_pts = np.array([
        [-175.1462, -435.8589,   -6.2563],
        [ -35.5797, -474.7771,   -8.4155],
        [-445.5756, -191.0506,    1.1713],
        [-499.7969,  179.2900,  -14.8579],
        [-383.7983,  421.2987,   -3.4232],
        [-283.6475,  522.4821,   -5.0816],
        [-148.8034,  607.4665,  -14.9617],
        [ 154.0142,  628.4564,   -9.5408],
        [ 408.4764,  511.2809,   -6.3291],
        [ 579.5506,  270.7174,    4.3911],
        [ 605.4905,  -20.2880,    6.3316],
        [ 558.8889, -164.7503,    2.6055],
        [ 375.3249, -380.8536,   -9.5462],
        [ 298.1854, -424.1200,   -9.0264],
        [ 118.3108, -477.2992,   -3.7381],
    ])

    # Fit plane to manual points, build basis
    table_center, plane_n = fit_plane(manual_table_pts)
    b1, b2 = plane_basis(plane_n)

    # Auto circle projected into 3D
    circ3d = np.array([
        table_center + radius * (np.cos(t)*b1 + np.sin(t)*b2)
        for t in theta
    ])

    # --- New: fit a circle to manual_table_pts in plane coords ---
    # project manual points into plane basis coordinates
    pt_coords = np.vstack([
        [(pt - table_center).dot(b1), (pt - table_center).dot(b2)]
        for pt in manual_table_pts
    ])
    mxc, myc, mr = fit_circle(pt_coords[:,0], pt_coords[:,1])
    # manual circle in 3D and in XY
    manual_center_3d = table_center + mxc*b1 + myc*b2
    manual_circ3d = np.array([
        manual_center_3d + mr * (np.cos(t)*b1 + np.sin(t)*b2)
        for t in theta
    ])
    manual_circ_xy = manual_circ3d[:, :2]

    # Set up figure with slider
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
        ax2d.scatter(df.com_x, df.com_y, s=2, alpha=0.2, label='COM')
        ax2d.plot(circ_xy[:,0], circ_xy[:,1], 'r-', label='auto fit')
        ax2d.plot(manual_circ_xy[:,0], manual_circ_xy[:,1], 'g--', label='manual fit')
        ax2d.scatter([table_center[0]], [table_center[1]], c='r', s=20)
        ax2d.scatter(manual_center_3d[0], manual_center_3d[1], c='g', s=20)
        # head triangle
        tri2d = np.array([
            [row.kp1_x, row.kp1_y],
            [row.kp2_x, row.kp2_y],
            [row.kp3_x, row.kp3_y],
            [row.kp1_x, row.kp1_y]
        ])
        ax2d.plot(tri2d[:,0], tri2d[:,1], '-k', lw=1.5, label='head tri')
        # fov arrow
        fov2d = xh[:2]; nrm = np.linalg.norm(fov2d)
        if nrm > 1e-6:
            fov2d = fov2d/nrm * (radius*0.5)
            ax2d.quiver(mid[0], mid[1], fov2d[0], fov2d[1],
                        angles='xy', scale_units='xy', scale=1, width=0.005)
        ax2d.set_aspect('equal')
        ax2d.set_title(f"Frame {idx}: 2D")
        ax2d.legend(loc='upper right')

        # 3D panel
        ax3d.clear()
        ax3d.scatter(cand_pts[:,0], cand_pts[:,1], cand_pts[:,2],
                     c='gray', s=1, label='hindpaws')
        ax3d.scatter(manual_table_pts[:,0], manual_table_pts[:,1], manual_table_pts[:,2],
                     c='orange', s=20, label='manual pts')
        # table surface
        gx = np.linspace(manual_table_pts[:,0].min(), manual_table_pts[:,0].max(), 10)
        gy = np.linspace(manual_table_pts[:,1].min(), manual_table_pts[:,1].max(), 10)
        gx, gy = np.meshgrid(gx, gy)
        gz = (table_center[2]
              - (plane_n[0]*(gx-table_center[0]) + plane_n[1]*(gy-table_center[1]))
              / plane_n[2])
        ax3d.plot_surface(gx, gy, gz, alpha=0.3)
        ax3d.plot(circ3d[:,0], circ3d[:,1], circ3d[:,2], 'b-', label='auto circ')
        ax3d.plot(manual_circ3d[:,0], manual_circ3d[:,1], manual_circ3d[:,2],
                  'g--', label='manual circ')
        # head triangle
        tri3d = np.array([
            [row.kp1_x, row.kp1_y, row.kp1_z],
            [row.kp2_x, row.kp2_y, row.kp2_z],
            [row.kp3_x, row.kp3_y, row.kp3_z],
            [row.kp1_x, row.kp1_y, row.kp1_z]
        ])
        ax3d.plot(tri3d[:,0], tri3d[:,1], tri3d[:,2], '-k', lw=1.5)
        # fov arrow
        ax3d.quiver(
            mid[0], mid[1], mid[2],
            xh[0], xh[1], xh[2],
            length=radius*0.5, normalize=True, linewidth=2, color='k'
        )
        ax3d.set_title("3D")
        ax3d.legend()

        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()
