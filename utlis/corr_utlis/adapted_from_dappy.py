"""
This utlis is adapted from Joshua Wu's Dappy/neuroposelib, as his already established functions are much better than some random shit if i would write on my own
note that! IMPORTANT: ALL KEYPOINTST START FROM 1 TO 22!!!!!! NOT FROM 0!!!!
"""

import pandas as pd
import numpy as np


def get_angles_from_h5_simple(df: pd.DataFrame, connectivity: list):
    angles, labels = [], []
    print("Calculating joint angles ... ")

    for triplet in connectivity:
        joint1, joint2, joint3 = triplet

        # Column names for each joint
        j1_cols = [f"kp{joint1}_x", f"kp{joint1}_y", f"kp{joint1}_z"]
        j2_cols = [f"kp{joint2}_x", f"kp{joint2}_y", f"kp{joint2}_z"]
        j3_cols = [f"kp{joint3}_x", f"kp{joint3}_y", f"kp{joint3}_z"]

        # Calculate vectors
        v1 = df[j1_cols].values - df[j2_cols].values
        v2 = df[j3_cols].values - df[j2_cols].values

        # Normalize vectors
        v1_u = v1 / np.linalg.norm(v1, axis=1)[:, None]
        v2_u = v2 / np.linalg.norm(v2, axis=1)[:, None]

        # Calculate angles
        angle = np.arccos(np.clip(np.sum(v1_u * v2_u, axis=1), -1, 1))
        angles.append(angle[:, None])

        # Label for the angle
        labels.append(f"ang_kp{joint1}_kp{joint2}_kp{joint3}")

    angles = np.concatenate(angles, axis=1)
    return angles, labels

def get_velocities_from_h5_simple(
    df: pd.DataFrame,
    joints: list = [0, 3, 5],
    widths: list = [3, 31, 89],
    abs_val: bool = False,
    f_s: int = 90,
    std: bool = True,
):
    vel = []
    labels = []
    print("Calculating velocities ... ")

    for joint in joints:
        joint_cols = [f"kp{joint}_x", f"kp{joint}_y", f"kp{joint}_z"]

        for width in widths:
            # Frame differences
            forward = df[joint_cols].shift(-width).values
            backward = df[joint_cols].shift(width).values
            dxyz = forward - backward

            # Velocity magnitude
            velocity = np.linalg.norm(dxyz, axis=1) * f_s / (2 * width + 1)

            if abs_val:
                velocity = np.abs(velocity)

            vel.append(velocity[:, None])
            labels.append(f"vel_kp{joint}_{2 * width + 1}")

    vel = np.concatenate(vel, axis=1)

    if std:
        vel_stds = []
        for joint in joints:
            joint_cols = [f"kp{joint}_x", f"kp{joint}_y", f"kp{joint}_z"]
            dxyz = df[joint_cols].diff().values
            for width in widths:
                rolling_std = (
                    pd.DataFrame(dxyz).rolling(2 * width + 1, min_periods=1).std().values
                )
                vel_stds.append(rolling_std[:, None])
                labels.append(f"vel_std_kp{joint}_{2 * width + 1}")
        vel = np.hstack((vel, np.concatenate(vel_stds, axis=1)))

    return vel, labels
