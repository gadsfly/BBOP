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
# 1) FOV & depth settings (C57BL/6)
# --------------------------------------------------------------------------
USE_SNOUT_SPINE_ORIGIN = False #True
SPINE_ORIGIN_RATIO    = 0.2

H_MONOCULAR_HALF_ANGLE = np.deg2rad(70)
H_BINOCULAR_HALF_ANGLE = np.deg2rad(20)
V_MONOCULAR_HALF_ANGLE = np.deg2rad(70)
V_BINOCULAR_HALF_ANGLE = np.deg2rad(25)

NEAR_DEPTH   = 100.  # mm (10 cm)
FAR_DEPTH    = 220.  # mm (22 cm)

CONE_THETA_RES = 30
CONE_Z_RES     = 10

# --------------------------------------------------------------------------
# 2) Data loading & utils
# --------------------------------------------------------------------------
def load_session_data(rec_path):
    rec_path = Path(rec_path)
    h5s = list((rec_path / "MIR_Aligned").glob("aligned_predictions_with_ca_and_dF_F*.h5"))
    if not h5s:
        raise FileNotFoundError("No .h5 found")
    h5 = h5s[0] if len(h5s)==1 else max(h5s, key=lambda p: len(p.stem))
    print("Loading:", h5)
    return pd.read_hdf(h5, key='df')

def fit_circle(x, y):
    pts = np.column_stack((x, y))
    h = ConvexHull(pts)
    hx, hy = pts[h.vertices].T
    def r(p, xx, yy): return np.hypot(xx-p[0], yy-p[1]) - p[2]
    p0 = [hx.mean(), hy.mean(), np.mean(np.hypot(hx-hx.mean(), hy-hy.mean()))]
    sol = least_squares(r, p0, args=(hx, hy))
    return sol.x

def fit_plane(pts):
    C = pts.mean(axis=0)
    _,_,Vt = np.linalg.svd(pts-C)
    n = Vt[-1]
    return (C, -n if n[2]<0 else n)

def plane_basis(n):
    v = np.array([1,0,0]) if abs(n[0])<1e-6 and abs(n[1])<1e-6 else np.array([0,0,1])
    b1 = np.cross(n, v); b1 /= np.linalg.norm(b1)
    b2 = np.cross(n, b1); b2 /= np.linalg.norm(b2)
    return b1, b2

def add_eye_estimates_with_spine(df, a=0.5, b=0.1):
    eL = df[['kp1_x','kp1_y','kp1_z']].to_numpy()
    eR = df[['kp2_x','kp2_y','kp2_z']].to_numpy()
    sn = df[['kp3_x','kp3_y','kp3_z']].to_numpy()
    sf = df[['kp4_x','kp4_y','kp4_z']].to_numpy()
    fL = sn-eL; fL/=np.linalg.norm(fL,axis=1,keepdims=True)
    fR = sn-eR; fR/=np.linalg.norm(fR,axis=1,keepdims=True)
    u  = sn-sf; u /=np.linalg.norm(u,axis=1,keepdims=True)
    E_l = eL + a*fL + b*u
    E_r = eR + a*fR + b*u
    df['eyel_x'],df['eyel_y'],df['eyel_z']=E_l.T
    df['eyer_x'],df['eyer_y'],df['eyer_z']=E_r.T
    G_l = sn-E_l; G_l/=np.linalg.norm(G_l,axis=1,keepdims=True)
    G_r = sn-E_r; G_r/=np.linalg.norm(G_r,axis=1,keepdims=True)
    df['uL_x'],df['uL_y'],df['uL_z']=G_l.T
    df['uR_x'],df['uR_y'],df['uR_z']=G_r.T
    return df

def generate_elliptical_cone(apex, axis, ha, va, length, n_theta=CONE_THETA_RES, n_z=CONE_Z_RES):
    axis = axis/np.linalg.norm(axis)
    tmp  = np.array([1,0,0]) if abs(axis[0])<0.9 else np.array([0,1,0])
    u = np.cross(axis, tmp); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    thetas = np.linspace(0,2*np.pi,n_theta)
    zs = np.linspace(0,length,n_z)
    X = np.zeros((n_z,n_theta)); Y = X.copy(); Z = X.copy()
    for i,z in enumerate(zs):
        rh = z*np.tan(ha); rv = z*np.tan(va)
        circ = (apex + np.outer(np.ones(n_theta),axis*z)
                    + np.outer(np.cos(thetas)*rh,u)
                    + np.outer(np.sin(thetas)*rv,v))
        X[i],Y[i],Z[i] = circ.T
    return X, Y, Z

# --------------------------------------------------------------------------
# 3) Main: interactive 2D + 3D + clamped cones
# --------------------------------------------------------------------------
if __name__=='__main__':
    session_dir = "/data/big_rim/rsync_dcc_sum/Oct3V1/2025_02_12/20241001PMCRE2mini_13_57"
    df = load_session_data(session_dir)

    # 2D circle
    xc,yc,r = fit_circle(df.com_x, df.com_y)
    theta = np.linspace(0,2*np.pi,200)
    circ_xy = np.column_stack((xc+r*np.cos(theta), yc+r*np.sin(theta)))

    # table plane
    manual_pts = np.array([
        [-175.1462,-435.8589,-6.2563],
        [-35.5797,-474.7771,-8.4155],
        [-445.5756,-191.0506,1.1713],
        [-499.7969,179.2900,-14.8579],
        [-383.7983,421.2987,-3.4232],
        [-283.6475,522.4821,-5.0816],
        [-148.8034,607.4665,-14.9617],
        [154.0142,628.4564,-9.5408],
        [408.4764,511.2809,-6.3291],
        [579.5506,270.7174,4.3911],
        [605.4905,-20.2880,6.3316],
        [558.8889,-164.7503,2.6055],
        [375.3249,-380.8536,-9.5462],
        [298.1854,-424.1200,-9.0264],
        [118.3108,-477.2992,-3.7381],
    ])
    table_C, table_n = fit_plane(manual_pts)
    b1,b2 = plane_basis(table_n)
    circ3d = np.array([table_C + r*(np.cos(t)*b1 + np.sin(t)*b2) for t in theta])

    # eyes & gaze
    df = add_eye_estimates_with_spine(df, a=0.5,b=0.1)

    # paws
    cand = []
    for pid in [17,20]:
        zs = df[f'kp{pid}_z']
        m = zs <= zs.min()+5.0
        pts = np.vstack((df[f'kp{pid}_x'][m], df[f'kp{pid}_y'][m], zs[m])).T
        cand.append(pts)
    cand_pts = np.vstack(cand)

    # plot setup
    fig = plt.figure(figsize=(11,6))
    ax2d = fig.add_subplot(1,2,1)
    ax3d = fig.add_subplot(1,2,2, projection='3d')
    plt.subplots_adjust(bottom=0.2)
    axs = fig.add_axes([0.2,0.05,0.6,0.03])
    slider = Slider(axs, 'Frame', 0, len(df)-1, valinit=0, valfmt='%0.0f')

    def update(val):
        i = int(slider.val)
        row = df.iloc[i]

        # 2D
        ax2d.clear()
        ax2d.scatter(df.com_x, df.com_y, s=2, alpha=0.2)
        ax2d.plot(circ_xy[:,0], circ_xy[:,1], 'r-')
        ax2d.scatter([xc],[yc],c='r',s=20)
        tri2d = np.array([[row.kp1_x,row.kp1_y],
                          [row.kp2_x,row.kp2_y],
                          [row.kp3_x,row.kp3_y],
                          [row.kp1_x,row.kp1_y]])
        ax2d.plot(tri2d[:,0],tri2d[:,1],'-k',lw=1.2)
        for s,L in (('l','L'),('r','R')):
            ex,ey = row[f'eye{s}_x'], row[f'eye{s}_y']
            ux,uy = row[f'u{L}_x'], row[f'u{L}_y']
            v = np.array([ux,uy])
            v = v/np.linalg.norm(v)*(r*0.4)
            ax2d.quiver(ex,ey,v[0],v[1],angles='xy',scale_units='xy',scale=1,width=0.005)
        ax2d.set_aspect('equal')
        ax2d.set_title(f"Frame {i}")

        # 3D
        ax3d.clear()
        ax3d.scatter(cand_pts[:,0],cand_pts[:,1],cand_pts[:,2],c='gray',s=1)
        ax3d.scatter(manual_pts[:,0],manual_pts[:,1],manual_pts[:,2],c='orange',s=20)
        gx = np.linspace(manual_pts[:,0].min(),manual_pts[:,0].max(),10)
        gy = np.linspace(manual_pts[:,1].min(),manual_pts[:,1].max(),10)
        gx,gy = np.meshgrid(gx,gy)
        gz = (table_C[2] - (table_n[0]*(gx-table_C[0]) + table_n[1]*(gy-table_C[1]))/table_n[2])
        ax3d.plot_surface(gx,gy,gz,alpha=0.3)
        ax3d.plot(circ3d[:,0],circ3d[:,1],circ3d[:,2],'b-')

        # head tri
        tri3d = np.array([[row.kp1_x,row.kp1_y,row.kp1_z],
                          [row.kp2_x,row.kp2_y,row.kp2_z],
                          [row.kp3_x,row.kp3_y,row.kp3_z],
                          [row.kp1_x,row.kp1_y,row.kp1_z]])
        ax3d.plot(tri3d[:,0],tri3d[:,1],tri3d[:,2],'-k',lw=1.2)

        # cones (clamped)
        for s,L in (('l','L'),('r','R')):
            sn = np.array([row.kp3_x,row.kp3_y,row.kp3_z])
            sf = np.array([row.kp4_x,row.kp4_y,row.kp4_z])
            if USE_SNOUT_SPINE_ORIGIN:
                apex = sn + SPINE_ORIGIN_RATIO*(sn-sf)
            else:
                apex = np.array([row[f'eye{s}_x'],row[f'eye{s}_y'],row[f'eye{s}_z']])
            axis = np.array([row[f'u{L}_x'],row[f'u{L}_y'],row[f'u{L}_z']])
            axis /= np.linalg.norm(axis)

            for depth,col in [(NEAR_DEPTH,'red'),(FAR_DEPTH,'blue')]:
                Xc,Yc,Zc = generate_elliptical_cone(apex,axis,
                            H_MONOCULAR_HALF_ANGLE,V_MONOCULAR_HALF_ANGLE,depth)
                # clamp at table
                Z_plane = (table_C[2] - 
                    (table_n[0]*(Xc-table_C[0]) + table_n[1]*(Yc-table_C[1]))/table_n[2])
                Zc_clamped = np.maximum(Zc, Z_plane)

                ax3d.plot_surface(
                    Xc, Yc, Zc_clamped,
                    color=col, alpha=0.2,
                    linewidth=0, shade=True
                )

        ax3d.set_title("3D + FOV cones")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()
