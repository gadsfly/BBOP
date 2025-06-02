import os
import numpy as np
import scipy.io as sio
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def plot_com_circle_for_path(
    base_path,
    pred_folder='DANNCE/predict00',
    vis_dir='vis',
    filename=None
):
    """
    Load COM data, fit a circle to its convex hull, plot COM + hull + fit,
    and save under base_path/vis/.

    Fallbacks:
      1) base_path/pred_folder/com3d_used.mat
      2) base_path/COM/pred_folder/com3d0.mat
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

    # convex hull & circle fit
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
    plt.scatter(x, y, s=4, alpha=0.3, label='COM')
    plt.scatter(hx, hy, s=8, color='g', label='Hull')
    plt.plot(
        xc + r*np.cos(theta),
        yc + r*np.sin(theta),
        'r-', lw=2, label='Fit'
    )
    plt.scatter(xc, yc, s=40, color='b', label='Center')
    plt.gca().set_aspect('equal')
    plt.title(os.path.basename(base_path))
    plt.legend()
    plt.show()

    # save
    save_folder = os.path.join(base_path, vis_dir)
    os.makedirs(save_folder, exist_ok=True)
    if filename is None:
        filename = 'com_circle.png'
    save_path = os.path.join(save_folder, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved COM circle plot to:\n  {save_path}")


def plot_group_com(paths, label,
                   pred_folder='COM/predict00',
                   vis_dir='vis'):
    fig, ax = plt.subplots(figsize=(7, 6))
    for p in paths:
        com_file = os.path.join(p, pred_folder, 'com3d0.mat')
        try:
            com = sio.loadmat(com_file)['com']
        except Exception as e:
            print(f"  ⚠️ skip {p!r}: {e}")
            continue
        x, y = com[:,0], com[:,1]
        ax.scatter(x, y, s=4, alpha=0.3, label=os.path.basename(p))
    ax.set_aspect('equal')
    ax.set_title(f"{label} group ({len(paths)} recordings)")
    ax.legend(fontsize='small', ncol=2)

    # save under the first folder’s vis_dir
    save_folder = os.path.join(paths[0], vis_dir)
    os.makedirs(save_folder, exist_ok=True)
    out_file = os.path.join(save_folder, f"{label}_combined.png")
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.show()
    plt.close(fig)
    print(f"✔️ Saved {label} plot to {out_file}")




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
