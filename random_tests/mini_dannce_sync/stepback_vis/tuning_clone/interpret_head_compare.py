#!/usr/bin/env python3
"""
Interpretation of head_compare.csv: compute circular errors, zero‐reference each axis,
error metrics, diagnostic plots, and simple smoothing.

Usage:
    python interpret_head_compare.py \
        --csv /path/to/head_compare.csv \
        --outdir /path/to/output_directory
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circmean, circstd


def wrap_deg(err: np.ndarray) -> np.ndarray:
    """Wrap angular error into [-180, +180] range"""
    return ((err + 180) % 360) - 180


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True, help='Path to head_compare.csv')
    parser.add_argument('--outdir', default='.', help='Directory to save outputs')
    parser.add_argument('--smooth', type=int, default=5,
                        help='Window size for rolling mean smoothing of errors')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    # compute time in seconds relative to first sample
    df['time_s'] = df['time_ms'] / 1000.0

    # 1) Zero‐reference each axis:
    pitch_zero = df['pitch_bno'].iloc[0]
    df['pitch_calc_zeroed'] = df['pitch_calc'] - pitch_zero
    df['pitch_bno_zeroed']  = df['pitch_bno']  - pitch_zero

    # 2) Compute wrapped raw errors
    for axis in ['yaw', 'pitch', 'roll']:
        if axis == 'pitch':
            raw = df['pitch_calc_zeroed'] - df['pitch_bno_zeroed']
        else:
            raw = df[f'{axis}_calc'] - df[f'{axis}_bno']
        df[f'{axis}_err_raw'] = wrap_deg(raw)

    # 3) Remove static bias per axis and optional smoothing
    for axis in ['yaw', 'pitch', 'roll']:
        arr = df[f'{axis}_err_raw']
        bias = circmean(arr, high=180, low=-180)
        zeroed = wrap_deg(arr - bias)
        df[f'{axis}_err'] = zeroed
        # rolling smoothing (centered)
        df[f'{axis}_err_smooth'] = df[f'{axis}_err'].rolling(
            window=args.smooth, center=True, min_periods=1).mean()

    # 4) Summary statistics on zeroed errors (circular std)
    stats = []
    for axis in ['yaw', 'pitch', 'roll']:
        arr = df[f'{axis}_err']
        stats.append({
            'axis': axis,
            'mean_err': circmean(arr, high=180, low=-180),
            'circ_std': circstd(arr, high=180, low=-180),
            'min_err': arr.min(),
            'max_err': arr.max()
        })
    stats_df = pd.DataFrame(stats).set_index('axis')
    stats_file = os.path.join(args.outdir, 'error_summary.csv')
    stats_df.to_csv(stats_file)
    print(f"Saved error summary → {stats_file}")

    # 5) Time‐series plots of error and smoothed error
    for axis in ['yaw', 'pitch', 'roll']:
        plt.figure()
        plt.plot(df['time_s'], df[f'{axis}_err'], alpha=0.6, label='raw')
        plt.plot(df['time_s'], df[f'{axis}_err_smooth'], linewidth=2, label='smooth')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (deg)')
        plt.title(f'{axis.capitalize()} Error Over Time')
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(args.outdir, f'{axis}_error_time.png')
        plt.savefig(out_png)
        plt.close()
        print(f"Saved time-series plot → {out_png}")

    # 6) Combined histogram of zeroed errors
    plt.figure()
    for axis in ['yaw', 'pitch', 'roll']:
        plt.hist(df[f'{axis}_err'], bins=50, alpha=0.6, label=axis)
    plt.xlabel('Error (deg)')
    plt.ylabel('Count')
    plt.title('Error Distribution (zeroed & wrapped)')
    plt.legend()
    hist_png = os.path.join(args.outdir, 'error_histograms.png')
    plt.savefig(hist_png)
    plt.close()
    print(f"Saved error histograms → {hist_png}")

    # 7) Outlier detection: frames where abs(error) > 3*circ_std
    outlier_info = []
    for axis in ['yaw', 'pitch', 'roll']:
        std = stats_df.loc[axis,'circ_std']
        mask = df[f'{axis}_err'].abs() > 3*std
        outlier_count = mask.sum()
        outlier_info.append((axis, outlier_count))
        print(f"{axis}: {outlier_count} outliers (>3σ)")
    
    # save outlier frames
    outlier_frames = pd.DataFrame({
        'frame': df.index,
        **{axis+'_outlier': df[f'{axis}_err'].abs() > 3*stats_df.loc[axis,'circ_std']
           for axis in ['yaw','pitch','roll']}
    })
    outlier_file = os.path.join(args.outdir, 'outlier_frames.csv')
    outlier_frames.to_csv(outlier_file, index=False)
    print(f"Saved outlier frames → {outlier_file}")


if __name__ == '__main__':
    main()
