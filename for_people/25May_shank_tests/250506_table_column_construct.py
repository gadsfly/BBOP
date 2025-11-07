import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
import scipy.io as sio
import os

base_path =  '/data/big_rim/rsync_dcc_sum/25Apri_social/2025_05_02/1shank3KOM_boxedshank3KOF' #'/hpc/group/tdunn/Bryan_Rigs/BigOpenField/Oct3V1/2024_11_13/20240910v1r_cricket_cyliner_test_16_17' #'/hpc/group/tdunn/Bryan_Rigs/BigOpenField/Oct3V1/2024_11_01/20240910V1r_BO_11_35' #'/hpc/group/tdunn/Bryan_Rigs/BigOpenField/Oct3V1/2024_11_13/20240910v1r_cricket_cyliner_test_16_17'
pred_folder = 'DANNCE/predict00'
# label3d_path = '/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/2024_06_28/1686941_left_right_2/pos_synced_1686941_left_right_2_2024_06_28_1686941_left_label3d_dannce.mat' #calib
pred_path = os.path.join(base_path, pred_folder, 'save_data_AVG.mat') #  smoothed_prediction_AVG0.mat

pred_3d = sio.loadmat(pred_path)['pred']
pred      = np.squeeze(pred_3d, axis=1)

com_file = os.path.join(base_path,pred_folder,'com3d_used.mat')
com_data = sio.loadmat(com_file)['com']

print(f"com_data {com_data.shape},  pred {pred.shape}")


x, y = com_data[:,0], com_data[:,1]
pts  = np.stack([x,y], axis=1)
hull = ConvexHull(pts)
hx, hy = pts[hull.vertices,0], pts[hull.vertices,1]


def residuals(p, xx, yy):
    xc,yc,r = p
    return np.sqrt((xx-xc)**2 + (yy-yc)**2) - r

init = [hx.mean(), hy.mean(), np.mean(np.hypot(hx-hx.mean(), hy-hy.mean()))]
res  = least_squares(residuals, init, args=(hx, hy))
xc, yc, r = res.x

print(f"Circle → center=({xc:.1f},{yc:.1f}),  radius={r:.1f} mm")


keypoint_labels = [
    'EarL','EarR','Snout','SpineF','SpineM','Tail(base)','Tail(mid)','Tail(end)',
    'ForepawL','WristL','ElbowL','ShoulderL','ForepawR','WristR','ElbowR','ShoulderR',
    'HindpawL','AnkleL','KneeL','HindpawR','AnkleR','KneeR'
]
paw_labels = ['HindpawL','HindpawR']
paw_idxs   = [keypoint_labels.index(lbl) for lbl in paw_labels]

threshold = 5.0  # mm above each paw’s minimum Z
candidate_pts = []
for idx in paw_idxs:
    coords = pred[:,:,idx]   # (F,3)
    xs, ys, zs = coords[:,0], coords[:,1], coords[:,2]
    sel = zs <= zs.min() + threshold
    candidate_pts.append(np.stack([xs[sel], ys[sel], zs[sel]], axis=1))
candidate_pts = np.vstack(candidate_pts)
print("Candidate contact pts:", candidate_pts.shape)

# SVD plane fit
centroid = candidate_pts.mean(axis=0)
_,_,Vt = np.linalg.svd(candidate_pts - centroid)
normal = Vt[-1]
if normal[2] < 0:
    normal = -normal

# table Z at (xc,yc)
table_z = centroid[2] - (normal[0]*(xc-centroid[0]) + normal[1]*(yc-centroid[1])) / normal[2]
tilt_rad = np.arccos(np.clip(normal.dot([0,0,1]), -1, 1))
tilt_deg = np.degrees(tilt_rad)
print(f"Table Z @ center = {table_z:.2f} mm,  tilt = {tilt_deg:.2f}°")



# ─── Cell 2: PLOT 2D + 3D ───────────────────────────────────────────────
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# reuse everything from Cell 1: x,y,hx,hy,xc,yc,r,candidate_pts,centroid,normal

theta = np.linspace(0, 2*np.pi, 100)

# basis for table plane
def plane_basis(n):
    v = np.array([1,0,0]) if abs(n[:2]).sum()<1e-6 else np.array([0,0,1])
    b1 = np.cross(n, v);  b1 /= np.linalg.norm(b1)
    b2 = np.cross(n, b1); b2 /= np.linalg.norm(b2)
    return b1, b2

b1, b2 = plane_basis(normal)
table_center = np.array([xc, yc, table_z])
circle3d = np.array([table_center + r*(np.cos(t)*b1 + np.sin(t)*b2) for t in theta])

fig = plt.figure(figsize=(14,6))

# 2D
ax1 = fig.add_subplot(1,2,1)
ax1.scatter(x, y,     s=4, alpha=0.3, label='COM')
ax1.scatter(hx, hy,   s=8, color='g', label='Hull')
ax1.plot(xc + r*np.cos(theta), yc + r*np.sin(theta), 'r-', lw=2, label='Fit')
ax1.scatter(xc, yc,   color='b', s=40, label='Center')
ax1.set_aspect('equal'); ax1.set_title('2D Circle Fit'); ax1.legend()

# 3D
ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.scatter(*candidate_pts.T, color='r', s=8, label='Paw pts')
# plot plane
grid_x = np.linspace(candidate_pts[:,0].min(), candidate_pts[:,0].max(), 20)
grid_y = np.linspace(candidate_pts[:,1].min(), candidate_pts[:,1].max(), 20)
gx, gy = np.meshgrid(grid_x, grid_y)
gz = centroid[2] - (normal[0]*(gx-centroid[0]) + normal[1]*(gy-centroid[1]))/normal[2]
ax2.plot_surface(gx, gy, gz, alpha=0.4, color='cyan')
# plot circle in plane
ax2.plot(circle3d[:,0], circle3d[:,1], circle3d[:,2], 'm-', lw=2, label='Table circle')
ax2.scatter(*table_center, color='k', s=60, marker='x', label='Table center')
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z'); ax2.set_title('3D View')
ax2.legend()

plt.tight_layout()
plt.show()
