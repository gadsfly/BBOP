#!/usr/bin/env python3
import sys
import os
# —— 根据你的项目结构调整这行 ——  
sys.path.append(os.path.abspath('../..'))

# 使用无界面后端，避免任何交互式显示
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# —— 从你自己的工具包中导入 ——  
from utlis.Ca_tools.roi_spike_vis_utlis import load_session_data, visualize_session


def quat_to_physical_angles(quats):
    """
    quats: (N,4) array of [qx, qy, qz, qw]
    返回 raw_yaw(-180…180), yaw(0–360), pitch(–90–90), roll(–180–180)
    """
    rot = R.from_quat(quats)
    fw = rot.apply(np.array([1.0, 0.0, 0.0]))
    up = rot.apply(np.array([0.0, 0.0, 1.0]))
    raw_yaw = np.degrees(np.arctan2(fw[:,1], fw[:,0]))
    yaw     = (raw_yaw + 360) % 360
    horiz   = np.linalg.norm(fw[:,:2], axis=1)
    pitch   = np.degrees(np.arctan2(fw[:,2], horiz))
    hd = np.radians(yaw)
    fx = np.stack([np.cos(hd), np.sin(hd), np.zeros_like(hd)], axis=1)
    fy = np.cross(fx, np.array([0.0, 0.0, 1.0]))
    ur = np.sum(up * fy, axis=1)
    uf = np.sum(up * fx, axis=1)
    roll    = np.degrees(np.arctan2(ur, uf))
    return raw_yaw, yaw, pitch, roll


def save_original_euler_plot(quats, path):
    """
    原始示例：检查范数，归一化（如有），并使用 SciPy as_euler 计算并绘制 [Yaw(PZ), Pitch(Y), Roll(X)]
    """
    norms = np.linalg.norm(quats, axis=1)
    print(f"Quaternion norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
    tol = 1e-3
    if not np.allclose(norms, 1.0, atol=tol):
        quats = quats / norms[:,None]
        print("Detected norms ≠ 1.0 → 已归一化所有四元数")
    rot = R.from_quat(quats)
    angles = rot.as_euler('zyx', degrees=True)
    yaw_o   = angles[:,0] % 360
    pitch_o = angles[:,1]
    roll_o  = angles[:,2]
    plt.figure(figsize=(10, 6))
    plt.plot(yaw_o,   label='Yaw (Z)')
    plt.plot(pitch_o, label='Pitch (Y)')
    plt.plot(roll_o,  label='Roll (X)')
    plt.xlabel('Frame Index')
    plt.ylabel('Angle (degrees)')
    plt.title('Head Orientation Euler Angles over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved original Euler angles plot: {path}")


def save_scipy_euler_plot(quats, path):
    """
    使用 SciPy as_euler('zyx') 方法绘制 yaw/pitch/roll (无归一化检查)
    """
    rot = R.from_quat(quats)
    angles = rot.as_euler('zyx', degrees=True)
    yaw_s   = angles[:, 0] % 360
    pitch_s = angles[:, 1]
    roll_s  = angles[:, 2]
    plt.figure(figsize=(12, 4))
    plt.plot(yaw_s, label='SciPy Yaw (0–360°)')
    plt.plot(pitch_s, label='SciPy Pitch')
    plt.plot(roll_s, label='SciPy Roll')
    plt.xlabel('Frame Index')
    plt.ylabel('Angle (°)')
    plt.title('SciPy Euler Angles Over Time')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved SciPy Euler angles plot: {path}")


def save_custom_euler_plot(yaw, pitch, roll, path):
    """
    使用自定义物理方法绘制 Euler 角
    """
    plt.figure(figsize=(12, 4))
    plt.plot(yaw,   label='Custom Yaw (°)')
    plt.plot(pitch, label='Custom Pitch')
    plt.plot(roll,  label='Custom Roll')
    plt.xlabel('Frame Index')
    plt.ylabel('Angle (°)')
    plt.title('Custom Physical Euler Angles Over Time')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved custom Euler angles plot: {path}")


def save_yaw_representations(raw_yaw, yaw, path):
    yaw_unwrapped = np.degrees(np.unwrap(np.radians(raw_yaw)))
    plt.figure(figsize=(10, 4))
    plt.plot(yaw, label='Yaw 0–360°')
    plt.plot(raw_yaw, label='Yaw –180…180°')
    plt.plot(yaw_unwrapped, label='Yaw unwrapped')
    plt.xlabel('Frame Index')
    plt.ylabel('Yaw (°)')
    plt.title('Different Yaw Representations')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved yaw representations plot: {path}")


def save_roi_tuning_plots(df, roi_cols, out_dir, prefix, n_bins=36):
    yaw = df['yaw'].to_numpy()
    pitch = df['pitch'].to_numpy()
    roll = df['roll'].to_numpy()
    bins_y = np.linspace(0, 360, n_bins+1)
    centers_y = (bins_y[:-1] + bins_y[1:]) / 2
    theta = np.deg2rad(centers_y)
    inds_y = np.digitize(yaw, bins_y) - 1
    inds_y %= n_bins
    fig, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(6,5))
    for roi in roi_cols:
        tuning = [df.loc[inds_y==i, roi].mean() for i in range(n_bins)]
        ax.plot(theta, tuning, alpha=0.5)
    ax.set_title('Yaw Tuning (all ROIs)')
    ax.set_xticks(np.deg2rad(np.arange(0,360,45)))
    yaw_file = os.path.join(out_dir, f'{prefix}_roi_yaw_tuning.png')
    fig.savefig(yaw_file)
    plt.close(fig)
    print(f"Saved ROI yaw tuning curve: {yaw_file}")
    bins_p = np.linspace(-90, 90, n_bins+1)
    centers_p = (bins_p[:-1] + bins_p[1:]) / 2
    inds_p = np.digitize(pitch, bins_p) - 1
    fig = plt.figure(figsize=(6,4))
    for roi in roi_cols:
        tuning = [df.loc[inds_p==i, roi].mean() for i in range(n_bins)]
        plt.plot(centers_p, tuning, alpha=0.5)
    pitch_file = os.path.join(out_dir, f'{prefix}_roi_pitch_tuning.png')
    plt.title('Pitch Tuning (all ROIs)')
    plt.xlabel('Pitch (°)')
    plt.ylabel('ΔF/F')
    plt.tight_layout()
    fig.savefig(pitch_file)
    plt.close(fig)
    print(f"Saved ROI pitch tuning curve: {pitch_file}")
    bins_r = np.linspace(-180, 180, n_bins+1)
    centers_r = (bins_r[:-1] + bins_r[1:]) / 2
    inds_r = np.digitize(roll, bins_r) - 1
    fig = plt.figure(figsize=(6,4))
    for roi in roi_cols:
        tuning = [df.loc[inds_r==i, roi].mean() for i in range(n_bins)]
        plt.plot(centers_r, tuning, alpha=0.5)
    roll_file = os.path.join(out_dir, f'{prefix}_roi_roll_tuning.png')
    plt.title('Roll Tuning (all ROIs)')
    plt.xlabel('Roll (°)')
    plt.ylabel('ΔF/F')
    plt.tight_layout()
    fig.savefig(roll_file)
    plt.close(fig)
    print(f"Saved ROI roll tuning curve: {roll_file}")


def batch_process_paths(paths_txt, output_root):
    with open(paths_txt, 'r') as f:
        ms_bases = [line.strip() for line in f if line.strip()]
    for ms_base in ms_bases:
        time = os.path.basename(ms_base)
        date = os.path.basename(os.path.dirname(ms_base))
        session_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(ms_base))))
        prefix = f"{session_id}_{date}_{time}"
        ms_folder = os.path.join(ms_base, 'My_V4_Miniscope')
        out_dir = os.path.join(output_root, session_id, date, time)
        os.makedirs(out_dir, exist_ok=True)
        df = load_session_data(ms_folder)
        quats = df[['qx','qy','qz','qw']].to_numpy()
        # 原始示例
        save_original_euler_plot(quats, os.path.join(out_dir, f'{prefix}_original_euler_plot.png'))
        # SciPy Euler
        save_scipy_euler_plot(quats, os.path.join(out_dir, f'{prefix}_scipy_euler_plot.png'))
        # Custom Euler
        raw_yaw, yaw, pitch, roll = quat_to_physical_angles(quats)
        df['yaw'], df['pitch'], df['roll'] = yaw, pitch, roll
        save_custom_euler_plot(yaw, pitch, roll, os.path.join(out_dir, f'{prefix}_custom_euler_plot.png'))
        # Yaw reps
        save_yaw_representations(raw_yaw, yaw, os.path.join(out_dir, f'{prefix}_yaw_reps.png'))
        # Save CSVs
        df.to_csv(os.path.join(out_dir, f'{prefix}_euler.csv'), index=False)
        dff = visualize_session(ms_folder)
        dff.to_csv(os.path.join(out_dir, f'{prefix}_dff.csv'), index=False)
        print(f"Saved Euler CSV and ΔF/F CSV for {prefix}")
        roi_cols = [c for c in df.columns if c.startswith('roi_')]
        save_roi_tuning_plots(df, roi_cols, out_dir, prefix)

if __name__ == '__main__':
    paths_txt   = '/home/lq53/mir_repos/BBOP/random_tests/25May_tuning_retry/combined_output.txt'
    output_root = '/home/lq53/mir_repos/BBOP/random_tests/25May_tuning_retry/2508_random_iter'
    batch_process_paths(paths_txt, output_root)
