"""
Drop-frame handler: detect and fix dropped frames across 6 cameras.

When cameras drop frames during recording, the frame counts become inconsistent.
This module aligns all cameras to a standard 30fps timeline using nearest-neighbor
matching (searchsorted), then rewrites the sync data_frame in the MAT file.

Must run AFTER sync (requires df_* calibration file).
Output: df_dh_<original>.mat  (original moved to prev_df_calib/)

Status codes (dropf_handle):
  0 = needs handling (inconsistent frame counts)
  1 = done (or consistent, no handling needed)
  2 = no need (old sessions before 2024_11_01)
  3 = failed
"""

import numpy as np
import os
import json
import shutil
import datetime
import scipy.io as sio
from utlis.sync_utlis.sync_df_utlis import find_calib_file


def load_frametimes(base_path, num_cameras=6):
    """Load frametimes.npy for each camera.

    Returns dict: {'Camera1': array(2, N), ...}
    """
    frametimes = {}
    for i in range(1, num_cameras + 1):
        camera_path = os.path.join(base_path, 'videos', f'Camera{i}', 'frametimes.npy')
        try:
            frametimes[f'Camera{i}'] = np.load(camera_path)
        except FileNotFoundError:
            print(f"  [dropf] frametimes.npy not found: {camera_path}")
    return frametimes


def check_max_shapes_consistency(frametimes_data):
    """Check if all cameras have the same number of frames.

    Returns (is_consistent: bool, max_shape: int)
    """
    max_shapes = {cam: data.shape[1] for cam, data in frametimes_data.items()}
    unique_shapes = set(max_shapes.values())
    return len(unique_shapes) == 1, max(max_shapes.values())


def create_standard_timeline(frametimes_data, max_shape_value, fps=30):
    """Build an ideal timeline at the given fps from the max end-time."""
    end_times = {cam: frametimes_data[cam][1][-1] for cam in frametimes_data}
    max_end_time = max(end_times.values())
    max_frame_number = max_end_time * fps
    frame_interval = 1 / fps

    standard_timeline = np.arange(0, max_end_time, frame_interval)
    standard_frame_numbers = np.arange(1, max_frame_number + 1)
    return standard_frame_numbers, standard_timeline, max_frame_number


def align_to_standard_timeline(frametimes_data, standard_frame_numbers, standard_timeline, max_frame_numbers):
    """Snap each camera's timestamps to the nearest standard timeline point.

    Uses searchsorted for O(n log n) alignment — handles both dropped
    and extra frames.
    """
    aligned_data = {}
    for camera, data in frametimes_data.items():
        frame_numbers = np.array(data[0])
        timestamps = np.array(data[1])

        indices = np.searchsorted(timestamps, standard_timeline)
        idx_left = np.clip(indices - 1, 0, len(timestamps) - 1)
        idx_right = np.clip(indices, 0, len(timestamps) - 1)

        time_left = timestamps[idx_left]
        time_right = timestamps[idx_right]

        diff_left = np.abs(standard_timeline - time_left)
        diff_right = np.abs(time_right - standard_timeline)

        use_left = diff_left <= diff_right
        closest_indices = np.where(use_left, idx_left, idx_right)

        aligned_data[camera] = {
            "frames": frame_numbers[closest_indices],
            "timestamps": timestamps[closest_indices],
        }
    return aligned_data


def update_data_frame(base_path, aligned_frametimes):
    """Rewrite sync data_frame in the MAT, save as df_dh_*, move original.

    Returns True on success, False on failure.
    """
    calib_path = find_calib_file(base_path)
    if not calib_path:
        print(f"  [dropf] No calib file found in {base_path}")
        return False

    calib_data = sio.loadmat(calib_path)
    sync = calib_data['sync']
    cameras = [f'Camera{i}' for i in range(1, 7)]

    for cam_idx, camera in enumerate(cameras):
        if camera in aligned_frametimes:
            data_frame = sync[cam_idx][0]['data_frame'][0][0][0]
            aligned_frames = aligned_frametimes[camera]['frames']
            frame_start = data_frame[0]
            offset = frame_start - 1
            mapped_frames = aligned_frames + offset
            sync[cam_idx][0]['data_frame'][0][0] = mapped_frames

    calib_data['sync'] = sync
    calib_name = os.path.basename(calib_path)
    save_path = os.path.join(base_path, f'df_dh_{calib_name}')
    sio.savemat(save_path, calib_data)
    print(f"  [dropf] Saved: {save_path}")

    prev_folder = os.path.join(base_path, 'prev_df_calib')
    os.makedirs(prev_folder, exist_ok=True)
    shutil.move(calib_path, prev_folder)
    print(f"  [dropf] Moved original to {prev_folder}")
    return True


def process_drop_frames(base_path, fps=30):
    """Main entry point: detect and fix dropped frames for one session.

    Requires sync to have run first (calib file must start with 'df').

    Returns:
      True  — frames were inconsistent and successfully fixed
      False — fix failed
      None  — frames already consistent, nothing to do
    """
    calib_path = find_calib_file(base_path)
    if not calib_path:
        print(f"  [dropf] No calib file in {base_path}. Run mir_generate_param + sync first.")
        return False

    if not os.path.basename(calib_path).startswith("df"):
        print(f"  [dropf] Calib file doesn't start with 'df' — run sync first: {calib_path}")
        return False

    # Already handled?
    if os.path.basename(calib_path).startswith("df_dh_"):
        print(f"  [dropf] Already drop-frame handled: {calib_path}")
        return None

    frametimes_data = load_frametimes(base_path)
    if not frametimes_data:
        print(f"  [dropf] No frametimes data loaded for {base_path}")
        return False

    consistent, max_shape = check_max_shapes_consistency(frametimes_data)
    if consistent:
        print(f"  [dropf] All cameras consistent ({max_shape} frames), skipping.")
        return None

    # Build per-camera diagnostics
    cam_info = {}
    for cam in sorted(frametimes_data.keys()):
        n = frametimes_data[cam].shape[1]
        frame_nums = frametimes_data[cam][0].astype(int)
        gaps = np.where(np.diff(frame_nums) > 1)[0]
        cam_info[cam] = {
            'n_frames': int(n),
            'diff_from_max': int(max_shape - n),
            'n_gaps': int(len(gaps)),
        }

    print(f"  [dropf] Inconsistent frames detected, aligning...")
    std_frames, std_timeline, max_fn = create_standard_timeline(frametimes_data, max_shape, fps)
    aligned = align_to_standard_timeline(frametimes_data, std_frames, std_timeline, max_shape)
    success = update_data_frame(base_path, aligned)

    # Save JSON log
    log = {
        'timestamp': datetime.datetime.now().isoformat(),
        'session': base_path,
        'max_frames': int(max_shape),
        'aligned_frames': int(len(std_timeline)),
        'fps': fps,
        'success': bool(success),
        'cameras': cam_info,
    }
    log_path = os.path.join(base_path, 'dropf_handle_log.json')
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"  [dropf] Log saved: {log_path}")

    return success
