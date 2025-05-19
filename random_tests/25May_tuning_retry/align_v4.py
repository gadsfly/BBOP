#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

# --- Load filtered BNO data ---
bno_csv = "/data/big_rim/rsync_dcc_sum/Oct3V1/2024_11_06/20241015pmcr2_17_13/MIR_Aligned/headOrientation_filtered.csv"
df_bno = pd.read_csv(bno_csv).rename(columns={'Time Stamp (ms)': 'time_ms'})
times = df_bno['time_ms'].to_numpy()
quats = df_bno[['qx','qy','qz','qw']].to_numpy()
rots = R.from_quat(quats)
bno_mats = rots.as_matrix()

print(f"BNO frames: {len(times)}")
print(f"BNO rotation mats shape: {bno_mats.shape}")

# --- Load prediction H5 and compute head frames ---
base_path = "/data/big_rim/rsync_dcc_sum/Oct3V1/2024_11_06/20241015pmcr2_17_13"
aligned_dir = Path(base_path) / 'MIR_Aligned'
h5_paths = list(aligned_dir.glob('aligned_predictions_with_ca_and_dF_F_*.h5'))
if not h5_paths:
    raise FileNotFoundError(f"No .h5 files found in {aligned_dir}")
print("H5 files:", h5_paths)

df_pred = pd.read_hdf(str(h5_paths[0]), key='df')
print(f"Predicted frames: {len(df_pred)}")

# --- Compute head rotation matrices ---
def normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(n < 1e-8, v, v / n)

R_heads = []
for _, row in df_pred.iterrows():
    earL  = row[['kp1_x','kp1_y','kp1_z']].to_numpy()
    earR  = row[['kp2_x','kp2_y','kp2_z']].to_numpy()
    snout = row[['kp3_x','kp3_y','kp3_z']].to_numpy()

    mid = (earL + earR) / 2.0
    x = normalize((snout - mid).reshape(1,3))[0]
    temp = (earR - earL).reshape(1,3)
    y = normalize((temp - np.dot(temp, x) * x).reshape(1,3))[0]
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-8)
    R_heads.append(np.column_stack((x, y, z)))

R_heads = np.stack(R_heads)
print(f"Head rotation mats shape: {R_heads.shape}")

assert R_heads.shape == bno_mats.shape, \
    f"Frame mismatch: pred {R_heads.shape[0]}, bno {bno_mats.shape[0]}"

# --- Precompute relative rotations (correct order) ---
R0_bno = bno_mats[0]
R_bno_rel = np.einsum('ij,nkj->nki', np.linalg.inv(R0_bno), bno_mats)
R0_pred = R_heads[0]
R_pred_rel = np.einsum('ij,nkj->nki', np.linalg.inv(R0_pred), R_heads)

# --- Initial triangle vertices (relative) ---
earL0 = df_pred.iloc[0][['kp1_x','kp1_y','kp1_z']].to_numpy()
earR0 = df_pred.iloc[0][['kp2_x','kp2_y','kp2_z']].to_numpy()
sn0   = df_pred.iloc[0][['kp3_x','kp3_y','kp3_z']].to_numpy()
mid0  = (earL0 + earR0) / 2.0
verts0 = np.vstack([earL0-mid0, earR0-mid0, sn0-mid0])

# --- Frame 0 sanity check ---
p0 = verts0 @ R_pred_rel[0].T
b0 = verts0 @ R_bno_rel[0].T

print("Frame 0 pred pts:\n", p0)
print("Frame 0 bno pts:\n", b0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(p0[:,0], p0[:,1], p0[:,2], c='r', label='Pred')
ax.scatter(b0[:,0], b0[:,1], b0[:,2], c='b', label='BNO')
ax.legend()
plt.show()

# --- Compute and apply the “fit” rotation to BNO at frame 0 ---
# This matches b0 → p0 with essentially zero RMSD.
rot_fit, rmsd = R.align_vectors(p0, b0)
R_fit_mat = rot_fit.as_matrix()
print(f"RMSD after alignment: {rmsd}")
print("Alignment rotation matrix:\n", R_fit_mat)

# Pre-rotate your BNO‐triangle basis
verts0_bno = verts0 @ R_fit_mat.T

# Verify the new Frame 0 overlay
b0_aligned = verts0_bno @ R_bno_rel[0].T
print("Frame 0 BNO pts after pre-rotation:\n", b0_aligned)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(p0[:,0], p0[:,1], p0[:,2], c='r', label='Pred')
ax.scatter(b0_aligned[:,0], b0_aligned[:,1], b0_aligned[:,2], c='b', label='BNO aligned')
ax.legend()
plt.show()

# --- Determine plot limits dynamically based on initial (aligned) frame ---
all0 = np.vstack([p0, b0_aligned])
lim = np.max(np.abs(all0)) * 1.1

# --- Set up animation ---
fig2 = plt.figure(figsize=(8,8))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_xlim(-lim, lim)
ax2.set_ylim(-lim, lim)
ax2.set_zlim(-lim, lim)
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
pred_line, = ax2.plot([], [], [], 'r-o', label='Predicted')
bno_line,  = ax2.plot([], [], [], 'b--o', label='BNO')
ax2.legend()

def update(i):
    p_pts = verts0 @ R_pred_rel[i].T
    b_pts = verts0_bno @ R_bno_rel[i].T   # <— use the pre-rotated basis here

    pred_line.set_data(p_pts[:,0], p_pts[:,1])
    pred_line.set_3d_properties(p_pts[:,2])

    bno_line.set_data(b_pts[:,0], b_pts[:,1])
    bno_line.set_3d_properties(b_pts[:,2])

    ax2.set_title(f"t = {times[i]} ms")
    return pred_line, bno_line

ani = FuncAnimation(fig2, update, frames=len(times), interval=50, blit=False)

# --- Save animation (mp4, fallback to GIF) ---
output_mp4 = "pred_bno_test_orientation_anim.mp4"
try:
    ani.save(output_mp4, writer='ffmpeg', fps=10)
    print("Saved animation to", output_mp4)
except Exception as e:
    print("ffmpeg save failed:", e)
    try:
        output_gif = "pred_bno_test_orientation_anim.gif"
        ani.save(output_gif, writer='pillow', fps=10)
        print("Saved animation to", output_gif)
    except Exception as e2:
        print("Pillow save failed:", e2)

