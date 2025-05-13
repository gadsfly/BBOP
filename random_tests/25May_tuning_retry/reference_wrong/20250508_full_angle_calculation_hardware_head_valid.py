#!/usr/bin/env python3
"""
nooooooooooooooooooooooooooooooo the aligned and not aliged things should not be on the same thing... this is shit
Standalone comparison of computed head Euler angles vs. hardware BNO measurements.

Usage:
    python full_angle_calculation_hardware_head_valid.py \
        --hdf5 /path/to/aligned_predictions_with_ca_and_dF_F.h5 \
        --mapping /path/to/mini_to_rec_mapping.json
"""
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v if norm < 1e-8 else v / norm


def compute_head_euler(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute head Euler angles (xyz, degrees) for every row in df.
    Assumes columns kp1_x,kp1_y,kp1_z (EarL), kp2_*, kp3_* (Snout).
    """
    times = []
    yaws, pitches, rolls = [], [], []

    for idx, row in df.iterrows():
        earL = row[['kp1_x','kp1_y','kp1_z']].to_numpy()
        earR = row[['kp2_x','kp2_y','kp2_z']].to_numpy()
        snout = row[['kp3_x','kp3_y','kp3_z']].to_numpy()
        ear_mid = (earL + earR) / 2.0
        head_x = normalize(snout - ear_mid)
        temp_y = earR - earL
        head_y = normalize(temp_y - np.dot(temp_y, head_x) * head_x)
        head_z = normalize(np.cross(head_x, head_y))
        R_head = np.column_stack((head_x, head_y, head_z))
        # euler = R.from_matrix(R_head).as_euler('xyz', degrees=True)
        # times.append(idx)
        # yaws.append(euler[0])
        # pitches.append(euler[1])
        # rolls.append(euler[2])
        roll, pitch, yaw = R.from_matrix(R_head).as_euler('xyz', degrees=True)
        rolls.append(roll)
        pitches.append(pitch)
        yaws.append(yaw)


    out = pd.DataFrame({
        'time': times,
        'yaw_calc': np.degrees(np.unwrap(np.radians(yaws))),
        'pitch_calc': np.degrees(np.unwrap(np.radians(pitches))),
        'roll_calc': np.degrees(np.unwrap(np.radians(rolls)))
    })
    return out


def load_bno_euler(bno_csv: str) -> pd.DataFrame:
    """
    Read headOrientation.csv (ms, qw,qx,qy,qz) and convert to euler xyz (deg).
    """
    df = pd.read_csv(bno_csv)
    quat = df[['qx','qy','qz','qw']].to_numpy()
    # euler = R.from_quat(quat).as_euler('xyz', degrees=True)
    # df['yaw_bno'] = np.degrees(np.unwrap(np.radians(euler[:,0])))
    # df['pitch_bno'] = np.degrees(np.unwrap(np.radians(euler[:,1])))
    # df['roll_bno'] = np.degrees(np.unwrap(np.radians(euler[:,2])))
    roll_bno, pitch_bno, yaw_bno = R.from_quat(quat).as_euler('xyz', degrees=True).T
    df['yaw_bno']   = np.degrees(np.unwrap(np.radians(yaw_bno)))
    df['pitch_bno'] = np.degrees(np.unwrap(np.radians(pitch_bno)))
    df['roll_bno']  = np.degrees(np.unwrap(np.radians(roll_bno)))

    df.rename(columns={'Time Stamp (ms)':'time_ms'}, inplace=True)
    return df[['time_ms','yaw_bno','pitch_bno','roll_bno']]


def merge_and_save(calc: pd.DataFrame, bno: pd.DataFrame, out_folder: str):
    """
    Align on nearest timestamp and save comparison CSV.
    """
    calc = calc.copy()
    start = calc['time'].iloc[0]
    # compute time_ms: handle datetime or numeric
    if pd.api.types.is_datetime64_any_dtype(calc['time']):
        time_ms = (calc['time'] - start).dt.total_seconds() * 1000
    else:
        time_ms = (calc['time'] - start) * 1000
    calc['time_ms'] = time_ms.astype(int)

    merged = pd.merge_asof(
        calc.sort_values('time_ms'),
        bno.sort_values('time_ms'),
        on='time_ms',
        suffixes=('_calc','_bno'),
        direction='nearest'
    )
    out_csv = os.path.join(out_folder, 'head_compare.csv')
    merged.to_csv(out_csv, index=False)
    print(f"Saved comparison → {out_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hdf5',   required=True, help='aligned_predictions .h5 file')
    p.add_argument('--mapping', required=True, help='mini_to_rec_mapping.json')
    args = p.parse_args()

    h5 = os.path.abspath(args.hdf5)
    rec_folder = os.path.dirname(os.path.dirname(h5))
    print(f"Recording folder: {rec_folder}")

    # 1) load and compute
    df = pd.read_hdf(h5, key='df')
    calc = compute_head_euler(df)
    out_calc = os.path.join(rec_folder, 'head_calc.csv')
    calc.to_csv(out_calc, index=False)
    print(f"Saved calculated head angles → {out_calc}")

    # 2) load mapping and find entry by rec_path
    with open(args.mapping) as f:
        mapping = json.load(f)
    candidates = [k for k,v in mapping.items()
                  if v.get('rec_path') and rec_folder in v['rec_path']]
    if not candidates:
        print("ERROR: could not find mapping entry via rec_path containing rec_folder.")
        print("Available rec_path values:")
        for v in mapping.values():
            rp = v.get('rec_path')
            if rp: print("  ", rp)
        sys.exit(1)
    daq_path = candidates[0]
    print(f"Found DAQ path: {daq_path}")

    # 3) load BNO data
    bno_csv = os.path.join(daq_path, 'headOrientation.csv')
    if not os.path.exists(bno_csv):
        print(f"ERROR: {bno_csv} not found.")
        sys.exit(1)
    bno = load_bno_euler(bno_csv)
    out_bno = os.path.join(rec_folder, 'head_bno_euler.csv')
    bno.to_csv(out_bno, index=False)
    print(f"Saved BNO-derived Euler angles → {out_bno}")

    # 4) merge and save
    merge_and_save(calc, bno, rec_folder)


if __name__ == '__main__':
    main()
