#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
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
# 3D plane fit & basis (using manual table points)
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
# Eye estimation (mid‐ear + spineF offset)
# --------------------------------------------------------------------------
def add_eye_estimates_with_spine(df, a=0.5, b=0.1):
    eL = df[['kp1_x','kp1_y','kp1_z']].to_numpy()
    eR = df[['kp2_x','kp2_y','kp2_z']].to_numpy()
    sn = df[['kp3_x','kp3_y','kp3_z']].to_numpy()
    sf = df[['kp4_x','kp4_y','kp4_z']].to_numpy()

    # forward axes
    fL = sn - eL; fL /= np.linalg.norm(fL, axis=1, keepdims=True)
    fR = sn - eR; fR /= np.linalg.norm(fR, axis=1, keepdims=True)
    # upward from spine
    u  = sn - sf;  u  /= np.linalg.norm(u, axis=1, keepdims=True)

    E_l = eL + a*fL + b*u
    E_r = eR + a*fR + b*u
    df['eyel_x'], df['eyel_y'], df['eyel_z'] = E_l.T
    df['eyer_x'], df['eyer_y'], df['eyer_z'] = E_r.T

    G_l = sn - E_l; G_l /= np.linalg.norm(G_l, axis=1, keepdims=True)
    G_r = sn - E_r; G_r /= np.linalg.norm(G_r, axis=1, keepdims=True)
    df['uL_x'], df['uL_y'], df['uL_z'] = G_l.T
    df['uR_x'], df['uR_y'], df['uR_z'] = G_r.T

    return df

# --------------------------------------------------------------------------
# Main: interactive 2D+3D FOV plot (manual‐plane)
# --------------------------------------------------------------------------
if __name__ == '__main__':
    session_dir = "/data/big_rim/rsync_dcc_sum/Oct3V1/2025_02_12/20241001PMCRE2mini_13_57"
    # "/data/big_rim/rsync_dcc_sum/Oct3V1/2024_10_25/20241002PMCr2_17_05"
    # "/data/big_rim/rsync_dcc_sum/Oct3V1/2024_10_14/20240916v1r1_16_37"
    df = load_session_data(session_dir)

    # 1) 2D COM circle
    x, y = df.com_x.to_numpy(), df.com_y.to_numpy()
    xc, yc, radius = fit_circle(x, y)
    theta = np.linspace(0, 2*np.pi, 200)
    circ_xy = np.column_stack((xc + radius*np.cos(theta),
                               yc + radius*np.sin(theta)))

    # 2) Manual table points → plane
    manual_pts = np.array([
        [-175.1462, -435.8589,  -6.2563],
        [ -35.5797, -474.7771,  -8.4155],
        [-445.5756, -191.0506,   1.1713],
        [-499.7969,  179.2900, -14.8579],
        [-383.7983,  421.2987,  -3.4232],
        [-283.6475,  522.4821,  -5.0816],
        [-148.8034,  607.4665, -14.9617],
        [ 154.0142,  628.4564,  -9.5408],
        [ 408.4764,  511.2809,  -6.3291],
        [ 579.5506,  270.7174,   4.3911],
        [ 605.4905,  -20.2880,   6.3316],
        [ 558.8889, -164.7503,   2.6055],
        [ 375.3249, -380.8536,  -9.5462],
        [ 298.1854, -424.1200,  -9.0264],
        [ 118.3108, -477.2992,  -3.7381],
    ])
    table_center, plane_n = fit_plane(manual_pts)
    b1, b2 = plane_basis(plane_n)

    # project circle into that plane
    circ3d = np.array([
        table_center + radius * (np.cos(t)*b1 + np.sin(t)*b2)
        for t in theta
    ])

    # 3) Eye & gaze axes
    df = add_eye_estimates_with_spine(df, a=0.5, b=0.1)

    # 4) Hindpaw contacts for context
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

    ###################
    import numpy as np
    import matplotlib.pyplot as plt

    # 1) Extract raw keypoints
    eL = df[['kp1_x','kp1_y','kp1_z']].to_numpy()  # EarL
    eR = df[['kp2_x','kp2_y','kp2_z']].to_numpy()  # EarR
    sn = df[['kp3_x','kp3_y','kp3_z']].to_numpy()  # Snout
    sf = df[['kp4_x','kp4_y','kp4_z']].to_numpy()  # SpineF
    sm = df[['kp5_x','kp5_y','kp5_z']].to_numpy()  # SpineM
    tb = df[['kp6_x','kp6_y','kp6_z']].to_numpy()  # Tail Base
    tm = df[['kp7_x','kp7_y','kp7_z']].to_numpy()  # Tail Mid

    # 2) Build and normalize the three axes
    #   A) Head axis: mid‐ear → snout
    midEar = 0.5*(eL + eR)
    head_axis = sn - midEar
    head_axis /= np.linalg.norm(head_axis, axis=1, keepdims=True)

    #   B) Spine axis: SpineF → SpineM
    spine_axis = sm - sf
    spine_axis /= np.linalg.norm(spine_axis, axis=1, keepdims=True)

    #   C) Tail axis: Tail Base → Tail Mid
    tail_axis = tm - tb
    tail_axis /= np.linalg.norm(tail_axis, axis=1, keepdims=True)

    # 3) Compute “pitch” relative to horizontal (table plane)
    #    angle = arctan2(z_component, horizontal_magnitude)
    def pitch_angle(v):
        hor = np.linalg.norm(v[:,:2], axis=1)
        return np.degrees(np.arctan2(v[:,2], hor))

    head_pitch  = pitch_angle(head_axis)
    spine_pitch = pitch_angle(spine_axis)
    tail_pitch  = pitch_angle(tail_axis)

    # 4) Plot distributions
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.hist(head_pitch, 50, range=(-90,90)); plt.title("Head axis pitch"); plt.xlabel("deg")
    plt.subplot(1,3,2)
    plt.hist(spine_pitch,50, range=(-90,90)); plt.title("Spine axis pitch"); plt.xlabel("deg")
    plt.subplot(1,3,3)
    plt.hist(tail_pitch, 50, range=(-90,90)); plt.title("Tail axis pitch"); plt.xlabel("deg")
    plt.tight_layout()
    plt.show()

    # 5) Plot timecourses (to see walk vs. groom)
    plt.figure(figsize=(12,6))
    plt.plot(head_pitch,  label='head')
    plt.plot(spine_pitch, label='spine')
    plt.plot(tail_pitch,  label='tail')
    plt.ylim(-90,90)
    plt.xlabel("Frame"); plt.ylabel("Pitch (deg)")
    plt.legend()
    plt.title("Axis pitch over time")
    plt.show()

    #######################


    # 5) Interactive plot
    fig = plt.figure(figsize=(11, 6))
    ax2d = fig.add_subplot(1, 2, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    plt.subplots_adjust(bottom=0.2)
    ax_slider = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, len(df)-1, valinit=0, valfmt='%0.0f')

    def update(val):
        idx = int(slider.val)
        row = df.iloc[idx]
        mid, _ = head_frame(row)

        # 2D
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

        # 3D
        ax3d.clear()
        ax3d.scatter(cand_pts[:,0], cand_pts[:,1], cand_pts[:,2],
                     c='gray', s=1, label='hindpaws')
        ax3d.scatter(manual_pts[:,0], manual_pts[:,1], manual_pts[:,2],
                     c='orange', s=20, label='manual pts')
        gx = np.linspace(manual_pts[:,0].min(), manual_pts[:,0].max(), 10)
        gy = np.linspace(manual_pts[:,1].min(), manual_pts[:,1].max(), 10)
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
