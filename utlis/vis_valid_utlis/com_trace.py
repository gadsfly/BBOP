import os
import numpy as np
import scipy.io as sio
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import json
import pandas as pd

#some tryings of trying to get circle when the mice ran little... maybe too complicated...
# def plot_com_circle_for_path(
#     base_path,
#     pred_folder='DANNCE/predict00',
#     vis_dir='vis',
#     filename=None,
#     meta_csv=None,            # path to your shared meta file
#     reuse_dims=False,         # if True, try to reuse from meta_csv first
#     manual_dims=None          # override dims: (xc, yc, r) or None
# ):
#     """
#     Load COM data, then either:
#       • use manual_dims if provided,
#       • else reuse from meta_csv if requested and found,
#       • else fit a new circle.

#     Always appends whichever dims were used to meta_csv (if given).
#     """
#     # locate COM
#     primary = os.path.join(base_path, pred_folder, 'com3d_used.mat')
#     fallback = os.path.join(base_path, 'COM/predict00', 'com3d0.mat')
#     if   os.path.isfile(primary):   com_file = primary
#     elif os.path.isfile(fallback):
#         com_file = fallback
#         print(f"Primary COM file not found; using fallback: {com_file}")
#     else:
#         raise FileNotFoundError(f"Neither {primary} nor {fallback} exists.")

#     mat = sio.loadmat(com_file)
#     com = mat.get('com')
#     if com is None:
#         raise KeyError(f"'com' variable not found in {com_file}")
#     x, y = com[:,0], com[:,1]

#     # ensure vis folder exists
#     save_folder = os.path.join(base_path, vis_dir)
#     os.makedirs(save_folder, exist_ok=True)

#     # 1) manual override
#     if manual_dims is not None:
#         xc, yc, r = manual_dims
#         print(f"Using manual dimensions: xc={xc}, yc={yc}, r={r}")

#     else:
#         # 2) reuse from meta_csv
#         xc = yc = r = None
#         if reuse_dims and meta_csv and os.path.isfile(meta_csv):
#             df_meta = pd.read_csv(meta_csv)
#             prev = df_meta.loc[df_meta['session_path'] == base_path]
#             if not prev.empty:
#                 xc, yc, r = prev[['xc','yc','r']].iloc[0].values
#                 print(f"Reusing dims for {base_path} from {meta_csv}")

#         # 3) fit new if still None
#         if xc is None:
#             pts = np.stack([x, y], axis=1)
#             hull = ConvexHull(pts)
#             hx, hy = pts[hull.vertices,0], pts[hull.vertices,1]
#             init = [
#                 hx.mean(), hy.mean(),
#                 np.mean(np.hypot(hx - hx.mean(), hy - hy.mean()))
#             ]
#             def residuals(p, xx, yy):
#                 return np.hypot(xx - p[0], yy - p[1]) - p[2]
#             res = least_squares(residuals, init, args=(hx, hy))
#             xc, yc, r = res.x
#             print(f"Fitted new dims: xc={xc:.2f}, yc={yc:.2f}, r={r:.2f}")

#     # append to meta_csv if requested
#     if meta_csv:
#         row = {
#             'session_path': base_path,
#             'xc': float(xc), 'yc': float(yc), 'r': float(r)
#         }
#         df_row = pd.DataFrame([row])
#         write_header = not os.path.isfile(meta_csv)
#         df_row.to_csv(meta_csv, mode='a', header=write_header, index=False)
#         print(f"Appended dims for {base_path} to {meta_csv}")

#     # --- plotting ---
#     plt.figure(figsize=(7,6))
#     theta = np.linspace(0,2*np.pi,200)
#     plt.scatter(x, y, s=4, alpha=0.3, label='COM')

#     # optionally replot hull each time
#     hull = ConvexHull(np.stack([x,y],axis=1))
#     hx, hy = x[hull.vertices], y[hull.vertices]
#     plt.scatter(hx, hy, s=8, color='g', label='Hull')

#     plt.plot(
#         xc + r*np.cos(theta),
#         yc + r*np.sin(theta),
#         'r-', lw=2, label='Fit'
#     )
#     plt.scatter(xc, yc, s=40, color='b', label='Center')

#     ax = plt.gca()
#     ax.set_aspect('equal')
#     lim = r * 1.1
#     ax.set_xlim(xc - lim, xc + lim)
#     ax.set_ylim(yc - lim, yc + lim)

#     plt.title(os.path.basename(base_path))
#     plt.legend()

#     # save figure
#     if filename is None:
#         filename = 'com_circle.png'
#     save_path = os.path.join(save_folder, filename)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.show()
#     plt.close()

#     print(f"Saved COM circle plot to:\n  {save_path}")

# if animal not moving, then have a large plotter region...
# def plot_com_circle_for_path(
#     base_path,
#     pred_folder='DANNCE/predict00',
#     vis_dir='vis',
#     filename=None
# ):
#     """
#     Load COM data, fit a circle to its convex hull, plot COM + hull + fit,
#     and save under base_path/vis/.
#     """
#     #     Fallbacks:
# #       1) base_path/pred_folder/com3d_used.mat
# #       2) base_path/COM/pred_folder/com3d0.mat
# #     """
# #     # primary and fallback locations
#     primary = os.path.join(base_path, pred_folder, 'com3d_used.mat')
#     fallback = os.path.join(base_path, 'COM/predict00', 'com3d0.mat')

#     if os.path.isfile(primary):
#         com_file = primary
#     elif os.path.isfile(fallback):
#         com_file = fallback
#         print(f"Primary COM file not found; using fallback: {com_file}")
#     else:
#         raise FileNotFoundError(
#             f"Neither {primary} nor {fallback} exists."
#         )

#     mat = sio.loadmat(com_file)
#     com = mat.get('com')
#     if com is None:
#         raise KeyError(f"'com' variable not found in {com_file}")
#     x, y = com[:,0], com[:,1]

#     # convex hull & circle fit
#     pts = np.stack([x, y], axis=1)
#     hull = ConvexHull(pts)
#     hx, hy = pts[hull.vertices,0], pts[hull.vertices,1]
#     init = [
#         hx.mean(),
#         hy.mean(),
#         np.mean(np.hypot(hx - hx.mean(), hy - hy.mean()))
#     ]

#     def residuals(params, xh, yh):
#         return np.hypot(xh - params[0], yh - params[1]) - params[2]

#     res = least_squares(residuals, init, args=(hx, hy))
#     xc, yc, r = res.x

#     # plotting
#     plt.figure(figsize=(7, 6))
#     theta = np.linspace(0, 2*np.pi, 200)
#     plt.scatter(x, y, s=4, alpha=0.3, label='COM')
#     plt.scatter(hx, hy, s=8, color='g', label='Hull')
#     plt.plot(xc + r*np.cos(theta), yc + r*np.sin(theta),
#              'r-', lw=2, label='Fit')
#     plt.scatter(xc, yc, s=40, color='b', label='Center')
#     plt.gca().set_aspect('equal')
#     plt.title(os.path.basename(base_path))
#     plt.legend()

#     # # adjust axis limits if radius is small
#     # if r < 400:
#     #     # padding can be fixed or a fraction of radius
#     #     pad = max(50, r * 0.2)
#     #     plt.xlim(xc - r - pad, xc + r + pad)
#     #     plt.ylim(yc - r - pad, yc + r + pad)
#     #     plt.gca().set_aspect('equal')  # re-enforce equal aspect

#     min_radius = 500       # the radius you want as your visual floor
#     pad_frac   = 0.05 #0.05 will be hopefully mimicing the default by matplotlib #0.2       # how much extra space around the circle

#     # pick the larger of your true r or the visual floor
#     eff_r = max(r, min_radius)

#     # now pad off of that
#     pad = eff_r * pad_frac

#     ax = plt.gca()
#     ax.set_aspect('equal')

#     # set limits centered on (xc, yc)
#     ax.set_xlim(xc - eff_r - pad, xc + eff_r + pad)
#     ax.set_ylim(yc - eff_r - pad, yc + eff_r + pad)

#     # save
#     save_folder = os.path.join(base_path, vis_dir)
#     os.makedirs(save_folder, exist_ok=True)
#     if filename is None:
#         filename = 'com_circle.png'
#     save_path = os.path.join(save_folder, filename)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.show()
#     plt.close()
#     print(f"Saved COM circle plot to:\n  {save_path}")


def plot_com_circle_for_path( #NO CIRCLE/OPTIONAL VERSION
    base_path,
    pred_folder='DANNCE/predict00',
    vis_dir='vis',
    filename=None,
    show_hull=True,
    show_fit=True,
    show_center=True
):
    """
    Load COM data, fit a circle to its convex hull, plot COM + hull + fit,
    and save under base_path/vis/.
    
    Parameters
    ----------
    base_path : str
        Base directory path
    pred_folder : str
        Prediction folder name
    vis_dir : str
        Visualization output directory
    filename : str, optional
        Output filename (default: 'com_circle.png')
    show_hull : bool
        Whether to show convex hull points (default: True)
    show_fit : bool
        Whether to show fitted circle (default: True)
    show_center : bool
        Whether to show circle center (default: True)
    """
    # primary and fallback locations
    primary = os.path.join(base_path, pred_folder, 'com3d_used.mat')
    fallback = os.path.join(base_path, 'COM/predict00', 'com3d0.mat')

    if os.path.isfile(primary):
        com_file = primary
    elif os.path.isfile(fallback):
        com_file = fallback
        print(f"Primary COM file not found; using fallback: {com_file}")
    else:
        raise FileNotFoundError(
            f"Neither {primary} nor {fallback} exists."
        )

    mat = sio.loadmat(com_file)
    com = mat.get('com')
    if com is None:
        raise KeyError(f"'com' variable not found in {com_file}")
    x, y = com[:,0], com[:,1]

    # convex hull & circle fit (still compute even if not shown)
    pts = np.stack([x, y], axis=1)
    hull = ConvexHull(pts)
    hx, hy = pts[hull.vertices,0], pts[hull.vertices,1]
    init = [
        hx.mean(),
        hy.mean(),
        np.mean(np.hypot(hx - hx.mean(), hy - hy.mean()))
    ]

    def residuals(params, xh, yh):
        return np.hypot(xh - params[0], yh - params[1]) - params[2]

    res = least_squares(residuals, init, args=(hx, hy))
    xc, yc, r = res.x

    # plotting
    plt.figure(figsize=(7, 6))
    theta = np.linspace(0, 2*np.pi, 200)
    
    # Always show COM
    plt.scatter(x, y, s=4, alpha=0.3, label='COM')
    
    # Optional elements
    if show_hull:
        plt.scatter(hx, hy, s=8, color='g', label='Hull')
    
    if show_fit:
        plt.plot(xc + r*np.cos(theta), yc + r*np.sin(theta),
                 'r-', lw=2, label='Fit')
    
    if show_center:
        plt.scatter(xc, yc, s=40, color='b', label='Center')
    
    plt.gca().set_aspect('equal')
    plt.title(os.path.basename(base_path))
    plt.legend()

    min_radius = 500
    pad_frac   = 0.05

    eff_r = max(r, min_radius)
    pad = eff_r * pad_frac

    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim(xc - eff_r - pad, xc + eff_r + pad)
    ax.set_ylim(yc - eff_r - pad, yc + eff_r + pad)

    # save
    save_folder = os.path.join(base_path, vis_dir)
    os.makedirs(save_folder, exist_ok=True)
    if filename is None:
        filename = 'com_circle.png'
    save_path = os.path.join(save_folder, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Saved COM circle plot to:\n  {save_path}")


# below works well and amazing, except that it will adjust and estimate, when sometimes animal did not move much it would not work lol
# def plot_com_circle_for_path(
#     base_path,
#     pred_folder='DANNCE/predict00',
#     vis_dir='vis',
#     filename=None
# ):
#     """
#     Load COM data, fit a circle to its convex hull, plot COM + hull + fit,
#     and save under base_path/vis/.

#     Fallbacks:
#       1) base_path/pred_folder/com3d_used.mat
#       2) base_path/COM/pred_folder/com3d0.mat
#     """
#     # primary and fallback locations
#     primary = os.path.join(base_path, pred_folder, 'com3d_used.mat')
#     fallback = os.path.join(base_path, 'COM/predict00', 'com3d0.mat')

#     if os.path.isfile(primary):
#         com_file = primary
#     elif os.path.isfile(fallback):
#         com_file = fallback
#         print(f"Primary COM file not found; using fallback: {com_file}")
#     else:
#         raise FileNotFoundError(
#             f"Neither {primary} nor {fallback} exists."
#         )

#     mat = sio.loadmat(com_file)
#     com = mat.get('com')
#     if com is None:
#         raise KeyError(f"'com' variable not found in {com_file}")
#     x, y = com[:,0], com[:,1]

#     # convex hull & circle fit
#     pts = np.stack([x, y], axis=1)
#     hull = ConvexHull(pts)
#     hx, hy = pts[hull.vertices,0], pts[hull.vertices,1]
#     init = [
#         hx.mean(),
#         hy.mean(),
#         np.mean(np.hypot(hx - hx.mean(), hy - hy.mean()))
#     ]

#     def residuals(params, xh, yh):
#         return np.hypot(xh - params[0], yh - params[1]) - params[2]

#     res = least_squares(residuals, init, args=(hx, hy))
#     xc, yc, r = res.x

#     # plotting
#     plt.figure(figsize=(7, 6))
#     theta = np.linspace(0, 2*np.pi, 200)
#     plt.scatter(x, y, s=4, alpha=0.3, label='COM')
#     plt.scatter(hx, hy, s=8, color='g', label='Hull')
#     plt.plot(
#         xc + r*np.cos(theta),
#         yc + r*np.sin(theta),
#         'r-', lw=2, label='Fit'
#     )
#     plt.scatter(xc, yc, s=40, color='b', label='Center')
#     plt.gca().set_aspect('equal')
#     plt.title(os.path.basename(base_path))
#     plt.legend()
    
    

#     # save
#     save_folder = os.path.join(base_path, vis_dir)
#     os.makedirs(save_folder, exist_ok=True)
#     if filename is None:
#         filename = 'com_circle.png'
#     save_path = os.path.join(save_folder, filename)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.show()
#     plt.close()
#     print(f"Saved COM circle plot to:\n  {save_path}")


# def plot_group_com(paths, label,
#                    pred_folder='COM/predict00',
#                    vis_dir='vis'):
#     fig, ax = plt.subplots(figsize=(7, 6))
#     for p in paths:
#         com_file = os.path.join(p, pred_folder, 'com3d0.mat')
#         try:
#             com = sio.loadmat(com_file)['com']
#         except Exception as e:
#             print(f"  ⚠️ skip {p!r}: {e}")
#             continue
#         x, y = com[:,0], com[:,1]
#         ax.scatter(x, y, s=4, alpha=0.3, label=os.path.basename(p))
#     ax.set_aspect('equal')
#     ax.set_title(f"{label} group ({len(paths)} recordings)")
#     ax.legend(fontsize='small', ncol=2)

#     # save under the first folder’s vis_dir
#     save_folder = os.path.join(paths[0], vis_dir)
#     os.makedirs(save_folder, exist_ok=True)
#     out_file = os.path.join(save_folder, f"{label}_combined.png")
#     fig.tight_layout()
#     fig.savefig(out_file, dpi=300)
#     plt.show()
#     plt.close(fig)
#     print(f"✔️ Saved {label} plot to {out_file}")




########################social##################3


import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares


def plot_com_circle_for_path_social(
    base_path,
    pred_folder='DANNCE/predict00',
    vis_dir='COM/predict00/vis',
    filename=None
):
    """
    Load multi-animal COM (x,y,z), extract x/y for each animal,
    fit circles to each convex hull in x-y space,
    plot both COM trajectories in distinct colors,
    and save under base_path/vis/.

    Fallbacks:
      1) base_path/pred_folder/com3d_used.mat
      2) base_path/COM/predict00/com3d0.mat
    """
    # determine COM file
    primary = os.path.join(base_path, pred_folder, 'com3d_used.mat')
    fallback = os.path.join(base_path, 'COM/predict00', 'com3d0.mat')
    if os.path.isfile(primary):
        com_file = primary
    elif os.path.isfile(fallback):
        com_file = fallback
        print(f"Primary COM file not found; using fallback: {com_file}")
    else:
        raise FileNotFoundError(f"Neither {primary} nor {fallback} exists.")

    # load raw COM
    mat = sio.loadmat(com_file)
    raw = mat.get('com')
    if raw is None:
        raise KeyError(f"'com' variable not found in {com_file}")

    # reshape to (frames,2,n_animals)
    if raw.ndim == 3 and raw.shape[1] == 3:
        # already (frames,3,n); take x,y only
        com_xy = raw[:, :2, :]
    elif raw.ndim == 3 and raw.shape[1] == 2:
        # already (frames,2,n)
        com_xy = raw
    elif raw.ndim == 2 and raw.shape[1] % 3 == 0:
        # flattened (frames,3*n)
        n = raw.shape[1] // 3
        com3 = raw.reshape(-1, 3, n)
        com_xy = com3[:, :2, :]
    elif raw.ndim == 2 and raw.shape[1] % 2 == 0:
        # flattened (frames,2*n)
        n = raw.shape[1] // 2
        com_xy = raw.reshape(-1, 2, n)
    else:
        raise ValueError(
            f"Unexpected COM array shape {raw.shape},"
            " expected (frames,3,n) or (frames,3*n) or (frames,2,n) or (frames,2*n)"
        )

    frames, _, n_animals = com_xy.shape
    if n_animals < 2:
        raise ValueError(f"Expected at least 2 animals, found {n_animals}")

    # prepare plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # color list for animals
    colors = ['C0', 'C1']

    def residuals(params, xh, yh):
        return np.hypot(xh - params[0], yh - params[1]) - params[2]

    # plot each animal's COM and circle fit
    for i in range(min(n_animals, 2)):
        x = com_xy[:, 0, i]
        y = com_xy[:, 1, i]
        # plot trajectory
        ax.scatter(x, y, color=colors[i], s=10, alpha=0.6, label=f'COM{i+1}')
        # convex hull and circle
        pts = np.stack([x, y], axis=1)
        hull = ConvexHull(pts)
        hx, hy = pts[hull.vertices, 0], pts[hull.vertices, 1]
        init = [hx.mean(), hy.mean(), np.mean(np.hypot(hx - hx.mean(), hy - hy.mean()))]
        res = least_squares(residuals, init, args=(hx, hy))
        xc, yc, r = res.x
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(xc + r*np.cos(theta), yc + r*np.sin(theta),
                linestyle='--', lw=2, color=colors[i],
                label=f'Fit{i+1}')
        ax.scatter(xc, yc, s=50, marker='x', color=colors[i],
                   label=f'Center{i+1}')

    ax.set_aspect('equal', 'box')
    ax.set_title(os.path.basename(base_path))
    ax.legend(loc='best', fontsize='small')

    # # save
    # save_folder = os.path.join(base_path, vis_dir)
    # os.makedirs(save_folder, exist_ok=True)
    # if filename is None:
    #     filename = 'com_circle.png'
    # save_path = os.path.join(save_folder, filename)
    # plt.tight_layout()
    # plt.savefig(save_path, dpi=300)
    # plt.close(fig)
    # print(f"Saved COM circle plot to: {save_path}")
