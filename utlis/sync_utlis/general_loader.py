import os, sys
sys.path.append(os.path.abspath('../..'))
from utlis.sync_utlis.mini_dannce_sync import load_and_reshape_com
import numpy as np
import pandas as pd
import scipy.io as sio
import os, glob, json
import xarray as xr

def load_flat_predictions_simple(
    rec_path,
    pred_folder='SDANNCE/predict00',
    pred_filename='save_data_AVG.mat',
    com_folder=None,
    kp_names=None
):
    """
    Assumes predictions are already (F, A, 3, J). No shape handling.
    Flattens to DataFrame with explicit names. COM is optional fallback/add-on.

    Column order (matches memory layout of (A, 3, J)):
        For each animal a: x over all joints, then y over all joints, then z over all joints.
        Example: kp1_x_a1, kp2_x_a1, ..., kpJ_x_a1, kp1_y_a1, ..., kpJ_y_a1, kp1_z_a1, ..., kpJ_z_a1,
                 then repeat for a2, ...
    """
    # ---- predictions
    pred_path = os.path.join(rec_path, pred_folder, pred_filename)
    pred_df = None
    if os.path.exists(pred_path):
        md = sio.loadmat(pred_path)
        if 'pred' not in md:
            raise KeyError(f"'pred' not found in {pred_path}. Keys: {list(md.keys())}")
        pred = md['pred']

        # STRICT expectation: (F, A, 3, J)
        if not (pred.ndim == 4 and pred.shape[2] == 3):
            raise ValueError(f"Expected pred shape (F, A, 3, J); got {pred.shape}")

        F, A, _, J = pred.shape

        # keypoint names
        if kp_names is not None:
            if len(kp_names) != J:
                raise ValueError(f"kp_names length {len(kp_names)} != J {J}")
            kp_labels = list(kp_names)
        else:
            kp_labels = [f"kp{i+1}" for i in range(J)]

        coords = ('x', 'y', 'z')

        # flatten without reordering axes: (F, A*3*J)
        pred_flat = pred.reshape(F, A * 3 * J)

        # column names follow the actual memory order: a -> coord -> j
        pred_cols = [
            f"{kp_labels[j]}_{coords[c]}_a{a+1}"
            for a in range(A) for c in range(3) for j in range(J)
        ]
        pred_df = pd.DataFrame(pred_flat, columns=pred_cols)
        pred_df.index.name = 'frame'

    # ---- COM (optional)
    com_df = None
    # try com in (com_folder or pred_folder), then fallback to 'COM/predict00'
    for root in [com_folder or pred_folder, 'COM/predict00']:
        if root is None:
            continue
        for fname in ('com3d_used.mat', 'com3d0.mat'):
            cpath = os.path.join(rec_path, root, fname)
            if os.path.exists(cpath):
                com = load_and_reshape_com(cpath)  # (F, 3, nA)
                Fc, _, nA = com.shape
                # axis-first then animal (your original style)
                com_cols = [f'com{a+1}_{ax}' for ax in ('x', 'y', 'z') for a in range(nA)]
                com_df = pd.DataFrame(com.reshape(Fc, 3 * nA), columns=com_cols)
                com_df.index.name = 'frame'
                break
        if com_df is not None:
            break

    # ---- combine
    if pred_df is None and com_df is None:
        raise FileNotFoundError(f"No pred at {pred_path} and no COM found.")

    if pred_df is not None and com_df is not None:
        Fmin = min(len(pred_df), len(com_df))
        out = pd.concat(
            [com_df.iloc[:Fmin].reset_index(drop=True),
             pred_df.iloc[:Fmin].reset_index(drop=True)],
            axis=1
        )
        out.index.name = 'frame'
        return out

    return pred_df if pred_df is not None else com_df

# assumes load_flat_predictions_simple is already defined/imported

def load_flat_with_frame_map(
    rec_path,
    pred_folder='SDANNCE/predict00',
    pred_filename='save_data_AVG.mat',
    com_folder=None,
    kp_names=None,
    mapping_path=None,
    json_is_zero_based=True,   # your JSON currently is 0-based
    out_index='python'         # 'python' -> 0-based index, 'matlab' -> 1-based index
):
    """
    1) Load flattened preds/COM.
    2) Filter rows by mapped six-cam frame indices from frame_mapping.json.
    3) Optionally present 1-based frame numbering for display/interop.
    """
    df = load_flat_predictions_simple(
        rec_path,
        pred_folder=pred_folder,
        pred_filename=pred_filename,
        com_folder=com_folder,
        kp_names=kp_names
    )

    # locate mapping json (default: MIR_Aligned/frame_mapping.json)
    if mapping_path is None:
        mapping_path = os.path.join(rec_path, 'MIR_Aligned', 'frame_mapping.json')
    if not os.path.exists(mapping_path):
        # nothing to filter; just optionally reindex
        if out_index == 'matlab':
            df = df.copy()
            df.index = np.arange(1, len(df) + 1)
            df.index.name = 'frame'
        return df

    with open(mapping_path, 'r') as f:
        mp = json.load(f)

    idx = np.asarray(mp['mapped_sixcam_frame_indices'], dtype=int)
    # convert to 0-based if your JSON were 1-based
    if not json_is_zero_based:
        idx = idx - 1

    # keep only in-range positions
    in_range = (idx >= 0) & (idx < len(df))
    idx = idx[in_range]
    df_mapped = df.iloc[idx].copy()
    # keep the original six-cam frame id as a column
    df_mapped['camera_frame_sixcam'] = idx if out_index == 'python' else (idx + 1)

    # present index as requested
    if out_index == 'matlab':
        df_mapped.index = np.arange(1, len(df_mapped) + 1)
    df_mapped.index.name = 'frame'
    return df_mapped



def read_mini_path(rec_path, filename='sync_to_mini_path.txt'):
    """Get miniscope root from a txt file in rec_path."""
    with open(os.path.join(rec_path, filename), 'r') as f:
        return f.read().strip()


def read_mini_path(rec_path, filename='sync_to_mini_path.txt'):
    with open(os.path.join(rec_path, filename), 'r') as f:
        return f.read().strip()

def read_miniscope_raw_timestamps(rec_path, time_col='Time Stamp (ms)'):
    mini_path = read_mini_path(rec_path)
    ts_csv = os.path.join(mini_path, 'My_V4_Miniscope', 'timeStamps.csv')
    ts_df = pd.read_csv(ts_csv)
    if time_col not in ts_df.columns:
        raise KeyError(f"Column '{time_col}' not in {ts_csv}. Got: {list(ts_df.columns)}")
    ts = ts_df[time_col].to_numpy()
    return ts  # full CSV timestamps, unmodified

def load_miniscope_signals_df_no_crop(
    mini_path,
    nc_file='wnd1500_stp700_max25_diff3.5_pnr1.1',
    time_col='Time Stamp (ms)',
    dff_percentile=20
):
    miniscope_path = os.path.join(mini_path, 'My_V4_Miniscope')
    ts_csv = os.path.join(miniscope_path, 'timeStamps.csv')

    # choose .nc
    if nc_file is not None and os.path.exists(nc_file):
        ca_file = nc_file
    elif nc_file is not None:
        ca_file = os.path.join(miniscope_path, f"minian_dataset_{nc_file}.nc")
        if not os.path.exists(ca_file):
            raise FileNotFoundError(f"No .nc from key '{nc_file}' at {ca_file}")
    else:
        nc_files = sorted(glob.glob(os.path.join(miniscope_path, '*.nc')))
        if not nc_files:
            raise FileNotFoundError(f"No .nc under {miniscope_path}")
        ca_file = nc_files[0]

    # raw timestamps (full CSV)
    ts_df = pd.read_csv(ts_csv)
    if time_col not in ts_df.columns:
        raise KeyError(f"Column '{time_col}' not in {ts_csv}. Got: {list(ts_df.columns)}")
    ts = ts_df[time_col].to_numpy()                    # length N_csv

    # Ca
    with xr.open_dataset(ca_file) as ds:
        C = ds['C'].values                              # (n_rois, n_frames)
    n_rois, n_frames = C.shape
    N_csv = ts.shape[0]

    # ΔF/F
    F0 = np.percentile(C, dff_percentile, axis=1, keepdims=True)
    F0 = np.where(F0 == 0, np.nan, F0)
    dFF = (C - F0) / F0                                 # (n_rois, n_frames)

    # allocate full-size arrays (pad with NaN if Ca shorter than CSV)
    Ca_full  = np.full((N_csv, n_rois), np.nan, dtype=float)
    dFF_full = np.full((N_csv, n_rois), np.nan, dtype=float)
    M = min(N_csv, n_frames)
    Ca_full[:M, :]  = C.T[:M, :]
    dFF_full[:M, :] = dFF.T[:M, :]

    calcium_cols = [f'calcium_roi{i}' for i in range(n_rois)]
    dff_cols     = [f'dF_F_roi{i}'    for i in range(n_rois)]
    df_ca  = pd.DataFrame(Ca_full,  index=ts, columns=calcium_cols)
    df_dff = pd.DataFrame(dFF_full, index=ts, columns=dff_cols)
    out = pd.concat([df_ca, df_dff], axis=1)
    out.index.name = 'timestamp_ms_mini'
    return out

def miniscope_mapped_to_json(
    rec_path,
    mini_path_file='sync_to_mini_path.txt',
    mapping_filename='frame_mapping.json',
    nc_file='wnd1500_stp700_max25_diff3.5_pnr1.1',
    time_col='Time Stamp (ms)',
    dff_percentile=20,
    assert_exact=True,
):
    """
    Map Miniscope Ca/ΔF/F to the miniscope timestamps listed in MIR_Aligned/frame_mapping.json.
    - Reads miniscope root from {rec_path}/{mini_path_file}
    - Uses your existing `load_miniscope_signals_df(mini_path, nc_file=...)` (not redefined here)
    - Subsets/orders rows by `mini_cam_timestamps` from the JSON
    """
    # miniscope root from txt
    with open(os.path.join(rec_path, mini_path_file), 'r') as f:
        mini_path = f.read().strip()
        if not os.path.exists(mini_path):
            mini_path = mini_path.replace('/data/big_rim/rsync_dcc_sum/', '/hpc/group/tdunn/Bryan_Rigs/BigOpenField/')

    # full miniscope DF indexed by miniscope timestamps
    df_full = load_miniscope_signals_df_no_crop(
        mini_path,
        nc_file=nc_file,
        time_col=time_col,
        dff_percentile=dff_percentile
    )

    # mapping json
    mapping_path = os.path.join(rec_path, 'MIR_Aligned', mapping_filename)
    with open(mapping_path, 'r') as f:
        mp = json.load(f)
    mini_ts = np.asarray(mp['mini_cam_timestamps'])

    # subset & order; keep exact timestamps
    df_map = df_full.reindex(mini_ts)
    df_map.index.name = 'timestamp_ms_mini'

    if assert_exact:
        # raise if any timestamps from JSON not found in the miniscope CSV index
        missing = df_map.index[df_map.isna().all(axis=1)]
        if len(missing) > 0:
            raise ValueError(f"{len(missing)} timestamps from JSON not found in Miniscope index. "
                             f"Example missing: {missing[:5].tolist()}")

    return df_map



# note the column is called camera_frame_sixcam.... 
def merge_pred_with_miniscope(
    rec_path,
    nc_key,
    dannce_folder='SDANNCE/predict00',
    pred_filename='save_data_AVG.mat',
    com_folder=None,
    mapping_filename='frame_mapping.json',
    save_h5=False,
    save_csv=False,
):
    """
    Merge Script A outputs:
      - Predictions/COM mapped via load_flat_with_frame_map(...)
      - Miniscope Ca and ΔF/F mapped via miniscope_mapped_to_json(...)

    Index of the merged DF = miniscope timestamps from MIR_Aligned/frame_mapping.json.
    Keeps SixCam indices in a column ('camera_frame_sixcam').

    Requirements (already defined elsewhere in your notebook):
      - load_flat_with_frame_map(...)
      - miniscope_mapped_to_json(...)
    """
    # 1) Load predictions/COM mapped to the JSON selection (still 0-based frame index here)
    df_pred = load_flat_with_frame_map(
        rec_path,
        pred_folder=dannce_folder,
        pred_filename=pred_filename,
        com_folder=com_folder,
        mapping_path=None,          # default MIR_Aligned/frame_mapping.json
        json_is_zero_based=True,
        out_index='python'
    )

    # 2) Load Miniscope signals (Ca, ΔF/F) mapped to the same JSON miniscope timestamps
    df_mini = miniscope_mapped_to_json(
        rec_path,
        mapping_filename=mapping_filename,
        nc_file=nc_key,
        assert_exact=True
    )

    # 3) Put predictions on the same miniscope-timestamp index
   
    with open(os.path.join(rec_path, 'MIR_Aligned', mapping_filename), 'r') as f:
        mp = json.load(f)
    mini_ts = np.asarray(mp['mini_cam_timestamps'])

    # Align lengths defensively (they should match under assert_exact=True)
    n = min(len(df_pred), len(df_mini), len(mini_ts))
    if (len(df_pred) != n) or (len(df_mini) != n):
        # keep it silent; caller can inspect shapes if desired
        df_pred = df_pred.iloc[:n].copy()
        df_mini = df_mini.iloc[:n].copy()
        mini_ts = mini_ts[:n]

    df_pred = df_pred.copy()
    df_pred.index = mini_ts
    df_pred.index.name = 'timestamp_ms_mini'

    # 4) Merge on miniscope timestamps
    merged = df_pred.join(df_mini, how='left')  # Ca/ΔF/F columns append to pred/COM

    # 5) Optional saves
    if save_h5:
        out_h5 = os.path.join(rec_path, 'MIR_Aligned',
                              f'merged_{os.path.basename(dannce_folder)}_{nc_key}.h5')
        merged.to_hdf(out_h5, key='df', mode='w')
    if save_csv:
        out_csv = os.path.join(rec_path, 'MIR_Aligned',
                               f'merged_{os.path.basename(dannce_folder)}_{nc_key}.csv')
        merged.to_csv(out_csv, index=True)

    return merged


