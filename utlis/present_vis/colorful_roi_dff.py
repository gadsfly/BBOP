import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import patheffects as pe
import scipy.sparse as sp

# ---------- robust A-handling ----------
def _as_array(A):
    """xarray/sparse -> plain squeezed ndarray."""
    if hasattr(A, "values"):
        A = A.values
    if sp.issparse(A):
        A = A.toarray()
    A = np.asarray(A)
    return np.squeeze(A)

def _make_mask_getter(A, H, W):
    """
    Return (n_rois, mask_of) where mask_of(i) -> (H,W) bool.
    Supports:
      (n, H*W), (H*W, n), (H, W, n), (n, H, W)
    """
    A = _as_array(A)
    if A.ndim == 2:
        n0, n1 = A.shape
        if n1 == H * W:           # (n, H*W)
            n = n0
            def mask_of(i): return (A[i] > 0).reshape(H, W)
            return n, mask_of
        if n0 == H * W:           # (H*W, n)
            n = n1
            def mask_of(i): return (A[:, i] > 0).reshape(H, W)
            return n, mask_of
        raise ValueError(f"A shape {A.shape} doesn't match H*W={H*W}.")
    elif A.ndim == 3:
        s0, s1, s2 = A.shape
        if (s0, s1) == (H, W):    # (H, W, n)
            n = s2
            def mask_of(i): return A[:, :, i] > 0
            return n, mask_of
        if (s1, s2) == (H, W):    # (n, H, W)
            n = s0
            def mask_of(i): return A[i, :, :] > 0
            return n, mask_of
        raise ValueError(f"A shape {A.shape} doesn't match (H,W,_) or (_,H,W).")
    else:
        raise ValueError(f"A ndim={A.ndim} not supported. Got shape {A.shape}.")

def _centroid(mask_bool):
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0: return None
    return float(xs.mean()), float(ys.mean())

# ---------- core painter ----------
def draw_roi_overlay_on_ax(
    ax,
    max_proj,
    A,
    selected,
    show_ids=True,
    lw_all=0.6,
    lw_sel=1.8,
    all_color="#bbbbbb",
    all_alpha=0.8,
):
    """
    Draw all ROI edges (light gray) and highlight `selected` in color.
    Returns {roi_idx: rgba_color} for selected.
    """
    # accept RGB or grayscale max_proj
    if max_proj.ndim == 2:
        H, W = max_proj.shape
    elif max_proj.ndim == 3:
        H, W = max_proj.shape[:2]
    else:
        raise ValueError(f"max_proj shape {max_proj.shape} unsupported.")

    n, mask_of = _make_mask_getter(A, H, W)

    selected = [int(i) for i in dict.fromkeys(selected) if 0 <= int(i) < n]
    palette = [cm.get_cmap("tab20")(i % 20) for i in range(len(selected))]
    color_map = {roi: palette[i] for i, roi in enumerate(selected)}

    # background
    if max_proj.ndim == 2:
        ax.imshow(max_proj, cmap="gray", interpolation="nearest")
    else:
        ax.imshow(max_proj, interpolation="nearest")

    # all ROI edges (light gray)
    for r in range(n):
        m = mask_of(r)
        if m.any():
            ax.contour(m.astype(float), levels=[0.5], colors=[all_color],
                      linewidths=lw_all, alpha=all_alpha, zorder=2)

    # selected overlays (colored + optional ids)
    for roi in selected:
        m = mask_of(roi)
        if not m.any(): continue
        c = color_map[roi]
        ax.contour(m.astype(float), levels=[0.5], colors=[c],
                   linewidths=lw_sel, alpha=1.0, zorder=3)
        if show_ids:
            cen = _centroid(m)
            if cen is not None:
                txt = ax.text(cen[0], cen[1], str(roi), color="white", fontsize=10,
                              ha="center", va="center", zorder=4)
                txt.set_path_effects([pe.Stroke(linewidth=2.5, foreground="black"), pe.Normal()])

    ax.axis("off")
    return color_map

# ---------- single-image overlay ----------
def roi_overlay_colored(
    data,
    max_proj,
    selected,
    show_ids=True,
    figsize=(8, 8),
    **kwargs
):
    """
    Poster-ready overlay. All ROIs in light gray; selected highlighted in color.
    """
    A = data['A']
    fig, ax = plt.subplots(figsize=figsize)
    _ = draw_roi_overlay_on_ax(ax, max_proj, A, selected, show_ids=show_ids, **kwargs)
    ax.set_title("ROI overlay (colored = selected)" + (" + ids" if show_ids else ""))
    return fig, ax

# ---------- side-by-side overlay + dF/F ----------
def overlay_and_dff(
    data,
    max_proj,
    dff,        # (n_rois, T)
    ts,         # (T,)
    selected,
    show_ids=True,
    shift=6.0,
    figsize=(14, 8),
    **kwargs
):
    """
    Left: overlay as above. Right: stacked ΔF/F traces with matching colors.
    """
    A = data['A']
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.05)
    ax_img = fig.add_subplot(gs[0, 0])
    ax_plot = fig.add_subplot(gs[0, 1])

    color_map = draw_roi_overlay_on_ax(ax_img, max_proj, A, selected, show_ids=show_ids, **kwargs)

    selected = [int(i) for i in dict.fromkeys(selected)]
    for i, roi in enumerate(selected):
        c = color_map.get(roi, (0, 0, 0, 1))
        y = dff[roi]
        ax_plot.plot(ts, y + i * shift, lw=0.9, color=c)
        if show_ids:
            ax_plot.text(ts[0], (y + i * shift)[0], f"{roi}", fontsize=9, color=c, va="bottom")

    ax_plot.set_xlabel("Time (ms)")
    ax_plot.set_yticks([])
    ax_plot.set_title("ΔF/F (colors = ROI edges)" + (" + ids" if show_ids else ""))
    return fig, (ax_img, ax_plot)
