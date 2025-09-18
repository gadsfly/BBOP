import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_two_coms_from_pred_df(
    df: pd.DataFrame,
    smooth_window: int = 5,          # rolling window (samples); 1 = no smoothing
    threshold_mm: float = None,      # e.g., 250.0 to draw a dashed threshold; None = no line
    out_path: str = None,            # e.g., os.path.join(rec_path, 'MIR_Aligned', 'two_coms.png')
    title: str = None
):
    """
    Visualize the first two COMs (X/Y/Z) and their 3D distance using the df produced by
    load_flat_with_frame_map(...). Works even if df has no Ca columns.

    Time axis behavior:
      - If 'timestamp_ms_mini' exists as a COLUMN, use it (ms).
      - Else use the DataFrame index (frames), which is what load_flat_with_frame_map sets by default.

    Notes:
      - Expects COM columns like 'com1_x','com1_y','com1_z','com2_x','com2_y','com2_z', ...
      - If fewer than two COM triplets are present, the function prints a message and returns.
    """
    if df is None or df.empty:
        print("[SKIP] Empty df"); 
        return

    # ---- time axis
    if 'timestamp_ms_mini' in df.columns:
        time = df['timestamp_ms_mini'].to_numpy()
        t_label = "Time (ms)"
    else:
        # load_flat_with_frame_map uses a 0-based integer index named 'frame'
        time = df.index.to_numpy()
        t_label = "Frame"

    # ---- detect COM prefixes present (com1, com2, ...)
    com_prefixes = sorted({c.split('_')[0] for c in df.columns if c.startswith('com')})
    def has_xyz(p): return all(f"{p}_{ax}" in df.columns for ax in ('x','y','z'))
    com_prefixes = [p for p in com_prefixes if has_xyz(p)]

    if len(com_prefixes) < 2:
        print("[SKIP] Need at least two COMs (e.g., com1/com2) in df columns"); 
        return

    p1, p2 = com_prefixes[0], com_prefixes[1]

    # ---- optional smoothing for COM trajectories
    if smooth_window and smooth_window > 1:
        df_sm = df[[f"{p1}_x", f"{p1}_y", f"{p1}_z",
                    f"{p2}_x", f"{p2}_y", f"{p2}_z"]].rolling(
                        window=int(smooth_window), center=True, min_periods=1
                    ).mean()
    else:
        df_sm = df[[f"{p1}_x", f"{p1}_y", f"{p1}_z",
                    f"{p2}_x", f"{p2}_y", f"{p2}_z"]]

    # ---- distance (raw) and smoothed
    dx = df[f'{p1}_x'].to_numpy() - df[f'{p2}_x'].to_numpy()
    dy = df[f'{p1}_y'].to_numpy() - df[f'{p2}_y'].to_numpy()
    dz = df[f'{p1}_z'].to_numpy() - df[f'{p2}_z'].to_numpy()
    dist = np.sqrt(dx*dx + dy*dy + dz*dz)

    if smooth_window and smooth_window > 1:
        dist_sm = pd.Series(dist).rolling(
            window=int(smooth_window), center=True, min_periods=1
        ).mean().to_numpy()
    else:
        dist_sm = None

    # ---- plot
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.2, 1.2, 1.0])

    # Panel 1: COM1 X/Y/Z
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(time, df_sm[f'{p1}_x'], label=f'{p1} X')
    ax0.plot(time, df_sm[f'{p1}_y'], label=f'{p1} Y')
    ax0.plot(time, df_sm[f'{p1}_z'], label=f'{p1} Z')
    ax0.set_ylabel('Position (mm)')
    ax0.set_title(f'{p1} X/Y/Z (w={int(smooth_window)})' if smooth_window > 1 else f'{p1} X/Y/Z')
    ax0.legend(loc='upper right', fontsize=9)
    ax0.tick_params(labelbottom=False)

    # Panel 2: COM2 X/Y/Z
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1.plot(time, df_sm[f'{p2}_x'], label=f'{p2} X')
    ax1.plot(time, df_sm[f'{p2}_y'], label=f'{p2} Y')
    ax1.plot(time, df_sm[f'{p2}_z'], label=f'{p2} Z')
    ax1.set_ylabel('Position (mm)')
    ax1.set_title(f'{p2} X/Y/Z (w={int(smooth_window)})' if smooth_window > 1 else f'{p2} X/Y/Z')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.tick_params(labelbottom=False)

    # Panel 3: Distance (raw + optional smoothed)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax2.plot(time, dist, lw=1.2, alpha=0.8, label='COM Distance (raw)')
    if dist_sm is not None:
        ax2.plot(time, dist_sm, lw=1.8, label=f'COM Distance (smoothed, w={int(smooth_window)})')
    if threshold_mm is not None:
        ax2.axhline(threshold_mm, ls='--', lw=1.2, color='black', label=f"Threshold = {threshold_mm:.3f} mm")
    ax2.set_ylabel('Distance (mm)')
    ax2.set_xlabel(t_label)
    ax2.set_title('COM Distance')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.margins(x=0)

    if title:
        fig.suptitle(title, y=1.02, fontsize=14)

    plt.tight_layout()

    if out_path:
        try:
            fig.savefig(out_path, dpi=300)
            print(f"[OK] Saved: {out_path}")
        except Exception as e:
            print(f"[WARN] Save failed: {e}")

    plt.show()
    plt.close(fig)
