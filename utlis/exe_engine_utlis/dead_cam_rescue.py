# Dead-camera rescue utilities
# Extracted from random_tests/25Sept_calib_intrinsic_update/5camera_test.ipynb
#
# Detects missing cameras via metadata.csv presence, then:
#   1) copies donor camera videos into each missing Camera{i} folder
#   2) clones calibration params (K, RDistort, TDistort, r, t) into the MAT file
#   3) moves the original MAT file to prev_calib/ so find_calib_file() sees only the rescued one
#
# Usage:
#   from utlis.exe_engine_utlis.dead_cam_rescue import rescue_missing_cameras
#   result = rescue_missing_cameras(session_path)

import os
import glob
import shutil
from typing import Optional, List, Dict, Tuple

import numpy as np
import scipy.io as sio


# ──────────────────────────────────────────────
# 1) DETECT
# ──────────────────────────────────────────────

def detect_missing_cameras(base_path: str, n_cams: int = 6) -> Tuple[List[int], List[int]]:
    vid_dir = os.path.join(base_path, "videos")
    ids = list(range(1, n_cams + 1))
    present = [i for i in ids if os.path.isfile(os.path.join(vid_dir, f"Camera{i}", "metadata.csv"))]
    missing = [i for i in ids if i not in present]
    return present, missing


# ──────────────────────────────────────────────
# small helpers (MATLAB struct manipulation)
# ──────────────────────────────────────────────

def _find_label3d_mat(base_path: str) -> str:
    mats = [p for p in glob.glob(os.path.join(base_path, "*.mat"))
            if os.path.basename(p).lower().endswith("label3d_dannce.mat")]
    if len(mats) != 1:
        raise RuntimeError(f"Expected exactly 1 '*label3d_dannce.mat' in {base_path}, found {len(mats)}: {mats}")
    return mats[0]


def _is_1x1_obj(x) -> bool:
    return isinstance(x, np.ndarray) and x.dtype == object and x.shape == (1, 1)


def _strip1(x):
    return x[0, 0] if _is_1x1_obj(x) else x


def _get_rec(P, i):
    if isinstance(P, np.ndarray) and P.ndim == 2 and P.shape[1] == 1:
        return P[i, 0]
    return P[i]


def _set_if_has(dst, src, field: str):
    if hasattr(dst, field) and hasattr(src, field):
        setattr(dst, field, _strip1(getattr(src, field)))


def _camname_map(m) -> Tuple[List[str], dict]:
    v = np.squeeze(m["camnames"])
    camnames = [str(np.squeeze(x)) for x in (v.tolist() if isinstance(v.tolist(), list) else [v.tolist()])]
    name_to_idx = {nm: i for i, nm in enumerate(camnames)}
    return camnames, name_to_idx


def _camid_to_idx(name_to_idx: dict, cam_id: int) -> int:
    nm = f"Camera{cam_id}"
    if nm not in name_to_idx:
        raise KeyError(f"{nm} not in camnames.")
    return name_to_idx[nm]


def _is_struct_1x1(x) -> bool:
    return isinstance(x, np.ndarray) and (x.dtype.names is not None) and (x.shape == (1, 1))


def _strip1_any(x):
    if _is_1x1_obj(x) or _is_struct_1x1(x):
        return x[0, 0]
    return x


def _unwrap_sync_one_layer_inplace(sync_arr):
    """Unwrap exactly one layer for each sync cell, in place."""
    if not isinstance(sync_arr, np.ndarray):
        return
    for idx in np.ndindex(sync_arr.shape):
        cell = sync_arr[idx]
        cell2 = _strip1_any(cell)
        if cell2 is not cell:
            sync_arr[idx] = cell2


def _get_sync_cell(sync_arr, idx: int):
    return sync_arr[idx, 0] if (isinstance(sync_arr, np.ndarray) and sync_arr.ndim == 2 and sync_arr.shape[1] == 1) else sync_arr[idx]


def _list_videos(folder: str, video_exts: Tuple[str, ...]) -> List[str]:
    try:
        names = os.listdir(folder)
    except FileNotFoundError:
        return []
    return sorted([n for n in names if n.lower().endswith(video_exts)])


# ──────────────────────────────────────────────
# 2) CLONE calibration params into MAT file
# ──────────────────────────────────────────────

def clone_missing_cameras_in_mat(
    base_path: str,
    donor_cam_id: Optional[int] = None,
    n_cams: int = 6,
    fields_to_copy: Tuple[str, ...] = ("K", "RDistort", "TDistort", "r", "t"),
    out_prefix: str = "clonedMissing_",
    precomputed_missing: Optional[List[int]] = None,
    fix_sync_one_layer: bool = True,
) -> str:
    present, missing = detect_missing_cameras(base_path, n_cams) if precomputed_missing is None \
                       else ([i for i in range(1, n_cams+1) if i not in precomputed_missing], precomputed_missing)

    if not present:
        raise RuntimeError("No live cameras detected.")
    if not missing:
        return _find_label3d_mat(base_path)

    donor = donor_cam_id if (donor_cam_id in present) else present[0]

    in_mat = _find_label3d_mat(base_path)
    m = sio.loadmat(in_mat, struct_as_record=False, squeeze_me=False)
    if "params" not in m or "camnames" not in m:
        raise KeyError("MAT missing 'params' and/or 'camnames'.")

    P = m["params"]
    sync = m.get("sync", None)
    _, name_to_idx = _camname_map(m)

    donor_idx = _camid_to_idx(name_to_idx, donor)
    rec_src = _strip1(_get_rec(P, donor_idx))

    # ---- clone PARAMS only ----
    for cam in missing:
        dst_idx = _camid_to_idx(name_to_idx, cam)
        rec_dst = _strip1(_get_rec(P, dst_idx))

        if isinstance(P, np.ndarray):
            if P.ndim == 2 and P.shape[1] == 1 and _is_1x1_obj(P[dst_idx, 0]):
                P[dst_idx, 0] = rec_dst
            elif P.ndim == 1 and _is_1x1_obj(P[dst_idx]):
                P[dst_idx] = rec_dst

        for f in fields_to_copy:
            _set_if_has(rec_dst, rec_src, f)

    # ---- final unwrap pass for params ----
    if isinstance(P, np.ndarray):
        for idx in np.ndindex(P.shape):
            elem = P[idx]
            if _is_1x1_obj(elem):
                P[idx] = elem[0, 0]
                elem = P[idx]
            for f in fields_to_copy:
                if hasattr(elem, f):
                    setattr(elem, f, _strip1(getattr(elem, f)))

    # ---- normalize sync: unwrap exactly one layer per cell; do NOT change contents ----
    if fix_sync_one_layer and ("sync" in m) and isinstance(sync, np.ndarray):
        _unwrap_sync_one_layer_inplace(sync)

    out_mat = os.path.join(base_path, out_prefix + os.path.basename(in_mat))
    sio.savemat(out_mat, m, do_compression=True, long_field_names=True)
    return out_mat


# ──────────────────────────────────────────────
# 3) COPY donor videos into missing camera folders
# ──────────────────────────────────────────────

def copy_videos_from_donor(
    base_path: str,
    donor_cam_id: Optional[int] = None,
    targets: Optional[List[int]] = None,
    n_cams: int = 6,
    clean_target: bool = True,
) -> Dict[int, Dict[str, int]]:
    """
    Copy the entire contents of:
        <base_path>/videos/Camera{donor_cam_id}
    into each target:
        <base_path>/videos/Camera{cam}
    Recursively copies files and subfolders, overwriting existing files.
    Returns: {target_cam_id: {"files_copied": N, "dirs_created": M}}
    """
    present, missing = detect_missing_cameras(base_path, n_cams)
    if not present:
        raise RuntimeError("No live cameras detected to use as donor.")

    donor = donor_cam_id if (donor_cam_id in present) else present[0]
    if targets is None:
        targets = list(missing)

    vids_dir = os.path.join(base_path, "videos")
    donor_dir = os.path.join(vids_dir, f"Camera{donor}")
    if not os.path.isdir(donor_dir):
        raise NotADirectoryError(f"Donor folder not found: {donor_dir}")

    summary: Dict[int, Dict[str, int]] = {}
    for cam in targets:
        if cam == donor:
            continue
        tgt_dir = os.path.join(vids_dir, f"Camera{cam}")
        os.makedirs(tgt_dir, exist_ok=True)

        # optionally clean target contents
        if clean_target:
            for name in os.listdir(tgt_dir):
                p = os.path.join(tgt_dir, name)
                try:
                    if os.path.isdir(p) and not os.path.islink(p):
                        shutil.rmtree(p)
                    else:
                        os.remove(p)
                except OSError:
                    pass

        files_copied = 0
        dirs_created = 0

        for root, dirs, files in os.walk(donor_dir):
            rel = os.path.relpath(root, donor_dir)
            dest_root = tgt_dir if rel == "." else os.path.join(tgt_dir, rel)
            if not os.path.exists(dest_root):
                os.makedirs(dest_root, exist_ok=True)
                dirs_created += 1

            for d in dirs:
                dest_sub = os.path.join(dest_root, d)
                if not os.path.exists(dest_sub):
                    os.makedirs(dest_sub, exist_ok=True)
                    dirs_created += 1

            for f in files:
                src_file = os.path.join(root, f)
                dst_file = os.path.join(dest_root, f)
                try:
                    shutil.copy2(src_file, dst_file)
                    files_copied += 1
                except Exception:
                    pass

        summary[cam] = {"files_copied": files_copied, "dirs_created": dirs_created}

    return summary


# ──────────────────────────────────────────────
# 4) RESCUE wrapper: detect -> copy -> clone -> move original -> verify
# ──────────────────────────────────────────────

def rescue_missing_cameras(
    base_path: str,
    donor_cam_id: Optional[int] = None,
    n_cams: int = 6,
    clean_target: bool = True,
    out_prefix: str = "clonedMissing_",
    video_exts: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mjpeg", ".mjpg"),
    verify: bool = True,
) -> Dict:
    """
    One call:
      1) detect missing cams;
      2) copy donor videos into each missing Camera{i} folder;
      3) clone params (K,RDistort,TDistort,r,t) into those cams in the MAT file;
      4) move original MAT to prev_calib/ so find_calib_file sees only rescued version;
      5) optionally verify videos/params match donor.

    Returns a dict with present/missing, donor, out_mat, and verify results.
    """
    present, missing = detect_missing_cameras(base_path, n_cams=n_cams)
    if not missing:
        return {
            "present": present, "missing": [],
            "donor": donor_cam_id if donor_cam_id in present else (present[0] if present else None),
            "out_mat": _find_label3d_mat(base_path) if present else None,
            "videos_verified": True, "params_verified": True,
            "video_diffs": {}, "param_diffs": {},
        }
    if not present:
        raise RuntimeError("No live cameras detected to use as donor.")

    donor = donor_cam_id if (donor_cam_id in present) else present[0]
    print(f"[dead_cam_rescue] Missing cameras: {missing}, donor: Camera{donor}")

    # 1) copy videos
    video_summary = copy_videos_from_donor(
        base_path=base_path,
        donor_cam_id=donor,
        targets=missing,
        n_cams=n_cams,
        clean_target=clean_target,
    )
    print(f"[dead_cam_rescue] Video copy summary: {video_summary}")

    # 2) clone params into mat
    out_mat = clone_missing_cameras_in_mat(
        base_path=base_path,
        donor_cam_id=donor,
        n_cams=n_cams,
        out_prefix=out_prefix,
        precomputed_missing=missing,
        fields_to_copy=("K", "RDistort", "TDistort", "r", "t"),
    )
    print(f"[dead_cam_rescue] Rescued MAT file: {out_mat}")

    # 3) move original mat to prev_calib/ so find_calib_file only sees the rescued one
    original_mat = _find_label3d_mat_excluding(base_path, out_prefix)
    if original_mat is not None:
        prev_calib_dir = os.path.join(base_path, "prev_calib")
        os.makedirs(prev_calib_dir, exist_ok=True)
        shutil.move(original_mat, prev_calib_dir)
        print(f"[dead_cam_rescue] Moved original {os.path.basename(original_mat)} -> prev_calib/")

    # 4) verification (optional)
    videos_ok, params_ok = True, True
    video_diffs: Dict[int, Dict[str, List[str]]] = {}
    param_diffs: Dict[int, List[str]] = {}

    if verify:
        vids_dir = os.path.join(base_path, "videos")
        donor_dir = os.path.join(vids_dir, f"Camera{donor}")
        donor_files = _list_videos(donor_dir, video_exts)
        donor_set = set(donor_files)

        for cam in missing:
            tgt_dir = os.path.join(vids_dir, f"Camera{cam}")
            tgt_files = _list_videos(tgt_dir, video_exts)
            tgt_set = set(tgt_files)
            if tgt_set != donor_set:
                videos_ok = False
                video_diffs[cam] = {
                    "missing_in_target": sorted(list(donor_set - tgt_set)),
                    "extra_in_target": sorted(list(tgt_set - donor_set)),
                }

        m = sio.loadmat(out_mat, struct_as_record=False, squeeze_me=False)
        P = m["params"]
        sync = m.get("sync", None)
        _, name_to_idx = _camname_map(m)
        donor_idx = _camid_to_idx(name_to_idx, donor)
        rec_src = _strip1(_get_rec(P, donor_idx))

        def _eq(a, b):
            a = _strip1(a); b = _strip1(b)
            try:
                return np.array_equal(np.asarray(a), np.asarray(b))
            except Exception:
                return False

        fields = ("K", "RDistort", "TDistort", "r", "t")
        for cam in missing:
            dst_idx = _camid_to_idx(name_to_idx, cam)
            rec_dst = _strip1(_get_rec(P, dst_idx))
            diffs = [f for f in fields if not (_eq(getattr(rec_dst, f), getattr(rec_src, f)) if hasattr(rec_dst, f) and hasattr(rec_src, f) else False)]

            if sync is not None:
                s_src = _strip1(_get_sync_cell(sync, donor_idx))
                s_dst = _strip1(_get_sync_cell(sync, dst_idx))
                if (isinstance(s_src, np.void) and isinstance(s_dst, np.void)
                        and s_src.dtype.names and s_dst.dtype.names
                        and ("data_sampleID" in s_src.dtype.names) and ("data_sampleID" in s_dst.dtype.names)):
                    if not _eq(s_src["data_sampleID"], s_dst["data_sampleID"]):
                        diffs.append("sync.data_sampleID")
                else:
                    diffs.append("sync.data_sampleID")

            if diffs:
                params_ok = False
                param_diffs[cam] = diffs

    return {
        "present": present,
        "missing": missing,
        "donor": donor,
        "video_copy_summary": video_summary,
        "out_mat": out_mat,
        "videos_verified": videos_ok,
        "params_verified": params_ok,
        "video_diffs": video_diffs,
        "param_diffs": param_diffs,
    }


# ──────────────────────────────────────────────
# Helper: find the original MAT (excluding the clonedMissing_ one)
# NEW function — needed so we can move the original to prev_calib/
# without accidentally moving the rescued copy.
# ──────────────────────────────────────────────

def _find_label3d_mat_excluding(base_path: str, exclude_prefix: str = "clonedMissing_") -> Optional[str]:
    """Find the original *label3d_dannce.mat that does NOT start with exclude_prefix."""
    mats = [p for p in glob.glob(os.path.join(base_path, "*.mat"))
            if os.path.basename(p).lower().endswith("label3d_dannce.mat")
            and not os.path.basename(p).startswith(exclude_prefix)]
    if len(mats) == 1:
        return mats[0]
    return None
