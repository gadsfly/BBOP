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
    hdf5 = h5_paths[0] if len(h5_paths)==1 else max(h5_paths, key=lambda p: len(p.stem))
    print("Loading:", hdf5)
    return pd.read_hdf(hdf5, key='df')

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
    return res.x  # (xc, yc, r)

# --------------------------------------------------------------------------
# 3D plane fit via SVD on hindpaw contacts
# --------------------------------------------------------------------------
def fit_plane(points):
    C = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - C)
    n = Vt[-1]
    if n[2] < 0:
        n = -n
    return C, n

def plane_basis(n):
    v = np.array([1, 0, 0]) if abs(n[0])<1e-6 and abs(n[1])<1e-6 else np.array([0, 0, 1])
    b1 = np.cross(n, v); b1 /= np.linalg.norm(b1)
    b2 = np.cross(n, b1); b2 /= np.linalg.norm(b2)
    return b1, b2

# --------------------------------------------------------------------------
# Head coordinate frame and triangle
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
# Eye position & optic-axis estimation using SpineF (kp4)
# --------------------------------------------------------------------------
def add_eye_estimates_with_spine(df, a=0.5, b=0.1):
    # kp1 = EarL, kp2 = EarR, kp3 = Snout, kp4 = SpineF
    eL = df[['kp1_x','kp1_y','kp1_z']].to_numpy()
    eR = df[['kp2_x','kp2_y','kp2_z']].to_numpy()
    sn = df[['kp3_x','kp3_y','kp3_z']].to_numpy()
    sf = df[['kp4_x','kp4_y','kp4_z']].to_numpy()

    # forward vectors per eye
    fL = sn - eL
    fL /= np.linalg.norm(fL, axis=1, keepdims=True)
    fR = sn - eR
    fR /= np.linalg.norm(fR, axis=1, keepdims=True)
    # upward axis from SpineF→Snout
    u  = sn - sf
    u  /= np.linalg.norm(u, axis=1, keepdims=True)

    # compute eye centers
    E_l = eL + a * fL + b * u
    E_r = eR + a * fR + b * u
    df['eyel_x'], df['eyel_y'], df['eyel_z'] = E_l.T
    df['eyer_x'], df['eyer_y'], df['eyer_z'] = E_r.T

    # gaze (optic) axes
    G_l = sn - E_l
    G_l /= np.linalg.norm(G_l, axis=1, keepdims=True)
    G_r = sn - E_r
    G_r /= np.linalg.norm(G_r, axis=1, keepdims=True)
    df['uL_x'], df['uL_y'], df['uL_z'] = G_l.T
    df['uR_x'], df['uR_y'], df['uR_z'] = G_r.T

    return df

# --------------------------------------------------------------------------
# Main: interactive 2D+3D FOV plot
# --------------------------------------------------------------------------
if __name__ == '__main__':
    session_dir = 
    # "/data/big_rim/rsync_dcc_sum/Oct3V1/2024_10_14/20240916v1r1_16_37"
    df = load_session_data(session_dir)

    import numpy as np
    import matplotlib.pyplot as plt

    # Compute key vectors
    eL = df[['kp1_x','kp1_y','kp1_z']].to_numpy()
    eR = df[['kp2_x','kp2_y','kp2_z']].to_numpy()
    sn = df[['kp3_x','kp3_y','kp3_z']].to_numpy()
    sf = df[['kp4_x','kp4_y','kp4_z']].to_numpy()

    mid = 0.5*(eL + eR)
    fwd = sn - mid
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)

    spine_vec = sn - sf
    spine_vec /= np.linalg.norm(spine_vec, axis=1, keepdims=True)

    # Eye gaze axis as before
    a, b = 0.5, 0.1
    E_l = eL + a*fwd + b*spine_vec
    G_l = sn - E_l
    G_l /= np.linalg.norm(G_l, axis=1, keepdims=True)

    # Plot histograms
    plt.figure(); plt.hist(fwd[:,2], bins=50); plt.title('xh_z distribution')
    plt.figure(); plt.hist(spine_vec[:,2], bins=50); plt.title('spine_vec_z distribution')
    plt.figure(); plt.hist(G_l[:,2], bins=50); plt.title('uL_z distribution')

    # Plot time series
    plt.figure(); plt.plot(fwd[:,2]); plt.title('xh_z over frames')
    plt.figure(); plt.plot(spine_vec[:,2]); plt.title('spine_vec_z over frames')
    plt.figure(); plt.plot(G_l[:,2]); plt.title('uL_z over frames')

    plt.show()


    # 1) Fit 2D circle to COM
    x, y = df.com_x.to_numpy(), df.com_y.to_numpy()
    xc, yc, radius = fit_circle(x, y)
    theta = np.linspace(0, 2*np.pi, 200)
    circ_xy = np.column_stack((xc + radius*np.cos(theta),
                               yc + radius*np.sin(theta)))

    # 2) Fit table plane from hindpaw contacts (IDs 17 & 20)
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

    # 3) Project 2D circle into 3D table plane
    circ3d = np.array([
        table_center + radius * (np.cos(t)*b1 + np.sin(t)*b2)
        for t in theta
    ])

    # 4) Estimate eyes & gaze axes with SpineF
    df = add_eye_estimates_with_spine(df, a=0.5, b=0.1)

    # 5) Interactive plotting setup
    fig = plt.figure(figsize=(11, 6))
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    plt.subplots_adjust(bottom=0.2)
    ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, len(df)-1, valinit=0, valfmt='%0.0f')

    def update(val):
        idx = int(slider.val)
        row = df.iloc[idx]
        mid, R_head = head_frame(row)

        # 2D view
        ax2d.clear()
        ax2d.scatter(df.com_x, df.com_y, s=2, alpha=0.2, label='COM')
        ax2d.plot(circ_xy[:,0], circ_xy[:,1], 'r-', label='table circ')
        ax2d.scatter([xc], [yc], c='r', s=20)
        tri2d = np.array([
            [row.kp1_x, row.kp1_y],
            [row.kp2_x, row.kp2_y],
            [row.kp3_x, row.kp3_y],
            [row.kp1_x, row.kp1_y]
        ])
        ax2d.plot(tri2d[:,0], tri2d[:,1], '-k', lw=1.5, label='head tri')
        # eye arrows
        for side, label in (('l','L'), ('r','R')):
            ex = row[f'eye{side}_x']; ey = row[f'eye{side}_y']
            ux = row[f'u{label}_x']; uy = row[f'u{label}_y']
            vec = np.array([ux, uy])
            vec = vec / np.linalg.norm(vec) * (radius * 0.4)
            ax2d.quiver(ex, ey, vec[0], vec[1],
                        angles='xy', scale_units='xy', scale=1,
                        width=0.005, label=f'eye{label}')
        ax2d.set_aspect('equal')
        ax2d.set_title(f"Frame {idx}: 2D")
        ax2d.legend(loc='upper right')

        # 3D view
        ax3d.clear()
        ax3d.scatter(cand_pts[:,0], cand_pts[:,1], cand_pts[:,2],
                     c='gray', s=1, label='hindpaws')
        # table surface
        gx = np.linspace(cand_pts[:,0].min(), cand_pts[:,0].max(), 10)
        gy = np.linspace(cand_pts[:,1].min(), cand_pts[:,1].max(), 10)
        gx, gy = np.meshgrid(gx, gy)
        gz = (table_center[2]
              - (plane_n[0]*(gx-table_center[0]) + plane_n[1]*(gy-table_center[1]))
              / plane_n[2])
        ax3d.plot_surface(gx, gy, gz, alpha=0.3)
        ax3d.plot(circ3d[:,0], circ3d[:,1], circ3d[:,2], 'b-', label='table circ3d')
        tri3d = np.array([
            [row.kp1_x, row.kp1_y, row.kp1_z],
            [row.kp2_x, row.kp2_y, row.kp2_z],
            [row.kp3_x, row.kp3_y, row.kp3_z],
            [row.kp1_x, row.kp1_y, row.kp1_z]
        ])
        ax3d.plot(tri3d[:,0], tri3d[:,1], tri3d[:,2], '-k', lw=1.5)
        # eye axes 3D
        for side, label in (('l','L'), ('r','R')):
            ex = row[f'eye{side}_x']; ey = row[f'eye{side}_y']; ez = row[f'eye{side}_z']
            ux = row[f'u{label}_x']; uy = row[f'u{label}_y']; uz = row[f'u{label}_z']
            ax3d.quiver(ex, ey, ez, ux, uy, uz,
                        length=radius*0.4, normalize=True,
                        linewidth=2, label=f'eye{label}')
        ax3d.set_title("3D")
        ax3d.legend()

        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()
