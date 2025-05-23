#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

# --- Load BNO data ---
bno_csv = "/data/big_rim/rsync_dcc_sum/Oct3V1/2024_11_06/20241015pmcr2_17_13/MIR_Aligned/headOrientation_filtered.csv"
df_bno = pd.read_csv(bno_csv).rename(columns={'Time Stamp (ms)': 'time_ms'})
times = df_bno['time_ms'].to_numpy()
quats = df_bno[['qx','qy','qz','qw']].to_numpy()
bno_mats = R.from_quat(quats).as_matrix()
print(f"BNO frames: {len(times)}, mats shape: {bno_mats.shape}")

# --- Load Predicted keypoints & compute head‐frame rotations ---
base_path  = "/data/big_rim/rsync_dcc_sum/Oct3V1/2024_11_06/20241015pmcr2_17_13"
aligned_dir = Path(base_path) / 'MIR_Aligned'
h5 = next(aligned_dir.glob('aligned_predictions_with_ca_and_dF_F_*.h5'), None)
if h5 is None:
    raise FileNotFoundError(f"No .h5 in {aligned_dir}")
df_pred = pd.read_hdf(str(h5), key='df')
print(f"Predicted frames: {len(df_pred)}")

def normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(n<1e-8, v, v/n)

R_heads = []
for _, row in df_pred.iterrows():
    earL  = row[['kp1_x','kp1_y','kp1_z']].to_numpy()
    earR  = row[['kp2_x','kp2_y','kp2_z']].to_numpy()
    snout = row[['kp3_x','kp3_y','kp3_z']].to_numpy()
    mid   = (earL + earR)/2

    x = normalize((snout-mid).reshape(1,3))[0]
    temp = (earR-earL)
    y = normalize((temp - np.dot(temp,x)*x).reshape(1,3))[0]
    z = np.cross(x,y); z /= (np.linalg.norm(z)+1e-8)

    R_heads.append(np.column_stack((x,y,z)))
R_heads = np.stack(R_heads)
assert R_heads.shape == bno_mats.shape

# --- 1) Build mean‐shape basis from Pred keypoints ---
earL_all  = df_pred[['kp1_x','kp1_y','kp1_z']].to_numpy()
earR_all  = df_pred[['kp2_x','kp2_y','kp2_z']].to_numpy()
snout_all = df_pred[['kp3_x','kp3_y','kp3_z']].to_numpy()

earL_m   = earL_all.mean(axis=0)
earR_m   = earR_all.mean(axis=0)
snout_m  = snout_all.mean(axis=0)
mid_m    = (earL_m + earR_m)/2

verts_avg = np.vstack([earL_m-mid_m, earR_m-mid_m, snout_m-mid_m])

x_avg = normalize((snout_m-mid_m).reshape(1,3))[0]
temp  = earR_m - earL_m
y_avg = normalize((temp - np.dot(temp,x_avg)*x_avg).reshape(1,3))[0]
z_avg = np.cross(x_avg,y_avg); z_avg /= (np.linalg.norm(z_avg)+1e-8)

R_base = np.column_stack((x_avg,y_avg,z_avg))

# --- 2) Compute relative rotations in that same basis ---
R_pred_rel = np.einsum('ij,nkj->nki', np.linalg.inv(R_base), R_heads)
R_bno_rel  = np.einsum('ij,nkj->nki', np.linalg.inv(R_base), bno_mats)

# --- 3) Sanity‐check at frame 0 ---
p0 = verts_avg @ R_pred_rel[0].T
b0 = verts_avg @ R_bno_rel[0].T

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(*p0.T, c='r', label='Pred (frame0)')
ax.scatter(*b0.T, c='b', label='BNO  (frame0)')
ax.legend()
plt.show()

# --- 4) Animate without any fitting step ---
lim = np.max(np.abs(np.vstack([p0,b0]))) * 1.1

fig2 = plt.figure(figsize=(6,6))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_xlim(-lim,lim); ax2.set_ylim(-lim,lim); ax2.set_zlim(-lim,lim)
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')

pred_line, = ax2.plot([],[],[],'r-o', label='Pred')
bno_line,  = ax2.plot([],[],[],'b--o', label='BNO')
ax2.legend()

def update(i):
    pts_p = verts_avg @ R_pred_rel[i].T
    pts_b = verts_avg @ R_bno_rel[i].T
    pred_line.set_data(pts_p[:,0], pts_p[:,1])
    pred_line.set_3d_properties(pts_p[:,2])
    bno_line.set_data(pts_b[:,0], pts_b[:,1])
    bno_line.set_3d_properties(pts_b[:,2])
    ax2.set_title(f"t = {times[i]} ms")
    return pred_line, bno_line

ani = FuncAnimation(fig2, update, frames=len(times), interval=50, blit=False)

# --- 5) Save ---
out_mp4 = "ana_ave_pred_bno_comparison.mp4"
try:
    ani.save(out_mp4, writer='ffmpeg', fps=10)
    print("Saved:", out_mp4)
except Exception:
    out_gif = "ana__ave_pred_bno_comparison.gif"
    ani.save(out_gif, writer='pillow', fps=10)
    print("Saved:", out_gif)
