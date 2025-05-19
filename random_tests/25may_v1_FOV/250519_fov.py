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
    pts = np.column_stack((x,y))
    hull = ConvexHull(pts)
    hx, hy = pts[hull.vertices].T
    def resid(p, xx, yy): return np.hypot(xx-p[0], yy-p[1]) - p[2]
    p0 = [hx.mean(), hy.mean(), np.mean(np.hypot(hx-hx.mean(), hy-hy.mean()))]
    res = least_squares(resid, p0, args=(hx, hy))
    return res.x  # xc, yc, r

# --------------------------------------------------------------------------
# 3D plane fit & basis
# --------------------------------------------------------------------------
def fit_plane(pts):
    C = pts.mean(axis=0)
    _,_,Vt = np.linalg.svd(pts - C)
    n = Vt[-1]
    if n[2]<0: n=-n
    return C, n

def plane_basis(n):
    v = np.array([1,0,0]) if abs(n[0])<1e-6 and abs(n[1])<1e-6 else np.array([0,0,1])
    b1 = np.cross(n,v); b1/=np.linalg.norm(b1)
    b2 = np.cross(n,b1); b2/=np.linalg.norm(b2)
    return b1, b2

# --------------------------------------------------------------------------
# Head frame & triangle
# --------------------------------------------------------------------------
def head_frame(row):
    eL = np.array([row.kp1_x, row.kp1_y, row.kp1_z])
    eR = np.array([row.kp2_x, row.kp2_y, row.kp2_z])
    sn = np.array([row.kp3_x, row.kp3_y, row.kp3_z])
    mid = (eL + eR)/2
    xh = sn - mid; xh/=np.linalg.norm(xh)
    ty = eR - eL
    yh = ty - xh*(xh.dot(ty)); yh/=np.linalg.norm(yh)
    zh = np.cross(xh, yh)
    return mid, np.column_stack((xh, yh, zh))

# --------------------------------------------------------------------------
# Eye estimation
# --------------------------------------------------------------------------
def add_eye_estimates(df, w=0.5):
    for side, kp in (('l',1), ('r',2)):
        ear = df[[f'kp{kp}_x',f'kp{kp}_y',f'kp{kp}_z']].values
        sn  = df[['kp3_x','kp3_y','kp3_z']].values
        E = (1-w)*ear + w*sn
        df[f'eye{side}_x'], df[f'eye{side}_y'], df[f'eye{side}_z'] = E.T
        V = sn - E
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        U = V/np.clip(norms,1e-8,None)
        df[f'u{side.upper()}_x'], df[f'u{side.upper()}_y'], df[f'u{side.upper()}_z'] = U.T
    return df

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__=='__main__':
    session_dir = "/data/big_rim/rsync_dcc_sum/Oct3V1/2024_10_14/20240916v1r1_16_37"
    df = load_session_data(session_dir)

    # 1) Manual table pts + plane‐fit analysis
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
    C_pt, n_pt = fit_plane(manual_pts)
    dists = np.dot(manual_pts - C_pt, n_pt)
    print("Manual‐pt plane fit residuals (mm):",
          "\n mean = ", dists.mean(),
          "\n std  = ", dists.std(),
          "\n max  = ", dists.max(),
          "\n min  = ", dists.min())

    # 2) Auto COM circle fit
    x, y = df.com_x.to_numpy(), df.com_y.to_numpy()
    xc, yc, r_auto = fit_circle(x, y)
    theta = np.linspace(0,2*np.pi,200)
    circ_auto_xy = np.column_stack((xc + r_auto*np.cos(theta),
                                    yc + r_auto*np.sin(theta)))

    # 3) Build 3D projections for both circles
    b1_pt, b2_pt = plane_basis(n_pt)
    circ_auto_3d = np.array([
        C_pt + r_auto*(np.cos(t)*b1_pt + np.sin(t)*b2_pt)
        for t in theta
    ])
    # manual circle in 3D
    proj = np.vstack([[(p-C_pt).dot(b1_pt), (p-C_pt).dot(b2_pt)] for p in manual_pts])
    mcx, mcy, r_man = fit_circle(proj[:,0], proj[:,1])
    C_man_3d = C_pt + mcx*b1_pt + mcy*b2_pt
    circ_man_3d = np.array([
        C_man_3d + r_man*(np.cos(t)*b1_pt + np.sin(t)*b2_pt)
        for t in theta
    ])
    circ_man_xy = circ_man_3d[:,:2]

    # 4) Estimate eyes
    df = add_eye_estimates(df, w=0.5)

    # 5) Hindpaw contact cloud
    paw_ids=[17,20]; cand=[]
    thr=5.0
    for pid in paw_ids:
        zs=df[f'kp{pid}_z'].to_numpy()
        mask=zs<=zs.min()+thr
        pts=np.vstack((df[f'kp{pid}_x'][mask],
                       df[f'kp{pid}_y'][mask],
                       zs[mask])).T
        cand.append(pts)
    cand_pts = np.vstack(cand)

    # 6) Interactive plot
    fig = plt.figure(figsize=(11,6))
    ax2d = fig.add_subplot(1,2,1)
    ax3d = fig.add_subplot(1,2,2,projection='3d')
    plt.subplots_adjust(bottom=0.2)
    ax_slider=fig.add_axes([0.2,0.05,0.6,0.03])
    slider = Slider(ax_slider,'Frame',0,len(df)-1,valinit=0,valfmt='%0.0f')

    def update(val):
        idx = int(slider.val)
        row = df.iloc[idx]
        mid, R_head = head_frame(row)

        # 2D
        ax2d.clear()
        ax2d.scatter(df.com_x,df.com_y,s=2,alpha=0.2,label='COM')
        ax2d.plot(circ_auto_xy[:,0],circ_auto_xy[:,1],'r-',label='auto circ')
        ax2d.plot(circ_man_xy[:,0],circ_man_xy[:,1],'g--',label='manual circ')
        ax2d.scatter([xc],[yc],c='r',s=20)
        ax2d.scatter([C_man_3d[0]],[C_man_3d[1]],c='g',s=20)
        tri2d = np.array([[row.kp1_x,row.kp1_y],
                          [row.kp2_x,row.kp2_y],
                          [row.kp3_x,row.kp3_y],
                          [row.kp1_x,row.kp1_y]])
        ax2d.plot(tri2d[:,0],tri2d[:,1],'-k',lw=1.5,label='head tri')
        for side,label in (('l','L'),('r','R')):
            ex,ey = row[f'eye{side}_x'], row[f'eye{side}_y']
            ux,uy = row[f'u{label}_x'], row[f'u{label}_y']
            vec = np.array([ux,uy])
            vec = vec/np.linalg.norm(vec)*(r_auto*0.4)
            ax2d.quiver(ex,ey,vec[0],vec[1],
                        angles='xy',scale_units='xy',scale=1,width=0.005,
                        label=f'eye{label}')
        ax2d.set_aspect('equal')
        ax2d.set_title(f"Frame {idx}: 2D")
        ax2d.legend(loc='upper right')

        # 3D
        ax3d.clear()
        ax3d.scatter(cand_pts[:,0],cand_pts[:,1],cand_pts[:,2],
                     c='gray',s=1,label='hindpaws')
        ax3d.scatter(manual_pts[:,0],manual_pts[:,1],manual_pts[:,2],
                     c='orange',s=20,label='manual pts')
        gx = np.linspace(manual_pts[:,0].min(),manual_pts[:,0].max(),10)
        gy = np.linspace(manual_pts[:,1].min(),manual_pts[:,1].max(),10)
        gx,gy = np.meshgrid(gx,gy)
        gz = (C_pt[2] - (n_pt[0]*(gx-C_pt[0]) + n_pt[1]*(gy-C_pt[1]))/n_pt[2])
        ax3d.plot_surface(gx,gy,gz,alpha=0.3)
        ax3d.plot(circ_auto_3d[:,0],circ_auto_3d[:,1],circ_auto_3d[:,2],
                  'b-',label='auto circ3d')
        ax3d.plot(circ_man_3d[:,0],circ_man_3d[:,1],circ_man_3d[:,2],
                  'g--',label='manual circ3d')
        tri3d = np.array([[row.kp1_x,row.kp1_y,row.kp1_z],
                          [row.kp2_x,row.kp2_y,row.kp2_z],
                          [row.kp3_x,row.kp3_y,row.kp3_z],
                          [row.kp1_x,row.kp1_y,row.kp1_z]])
        ax3d.plot(tri3d[:,0],tri3d[:,1],tri3d[:,2],'-k',lw=1.5)
        for side,label in (('l','L'),('r','R')):
            ex,ey,ez = row[f'eye{side}_x'],row[f'eye{side}_y'],row[f'eye{side}_z']
            ux,uy,uz = row[f'u{label}_x'],row[f'u{label}_y'],row[f'u{label}_z']
            ax3d.quiver(ex,ey,ez,ux,uy,uz,
                        length=r_auto*0.4,normalize=True,linewidth=2,
                        label=f'eye{label}')
        ax3d.set_title("3D")
        ax3d.legend()

        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()
