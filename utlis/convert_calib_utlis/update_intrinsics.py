import os
import time
import glob
import shutil
import numpy as np
import scipy.io as sio


def update_intrinsics_from_new_calib(base_path: str, new_calib_mat: str) -> str:
    """
    Update intrinsics + extrinsics in the session's *label3d_dannce.mat* under `base_path` using `new_calib_mat`.

    Behavior:
      - Locate exactly one file in base_path whose name ends with 'label3d_dannce.mat' (case-insensitive).
      - Copy K, RDistort, TDistort, r, t per camera from new_calib_mat into old['params'].
      - Strip only one extra 1x1 object layer on the records and on those fields.
      - Save to intrinsicsUpdated_<original>.mat in base_path.
      - Move the original into base_path/prev_intrinsic_calib/ (timestamped if name collides).

    Returns:
      Path to the written intrinsicsUpdated_*.mat
    """
    if not os.path.isdir(base_path):
        raise NotADirectoryError(f"Not a directory: {base_path}")
    if not os.path.isfile(new_calib_mat):
        raise FileNotFoundError(f"New calib .mat not found: {new_calib_mat}")

    # ---- find exactly one *label3d_dannce.mat (non-recursive, case-insensitive) ----
    cand = [p for p in glob.glob(os.path.join(base_path, "*.mat"))
            if os.path.basename(p).lower().endswith("label3d_dannce.mat")]
    if len(cand) != 1:
        raise RuntimeError(
            f"Expected exactly 1 '*label3d_dannce.mat' in {base_path}, found {len(cand)}: {cand}"
        )
    old_mat = cand[0]

    # ---- load, preserve containers (no squeeze) ----
    old = sio.loadmat(old_mat, struct_as_record=False, squeeze_me=False)
    new = sio.loadmat(new_calib_mat, struct_as_record=False, squeeze_me=False)

    if "params" not in old or "camnames" not in old:
        raise KeyError("Old .mat missing required fields: 'params' and/or 'camnames'")
    if "params" not in new or "camnames" not in new:
        raise KeyError("New calib .mat missing required fields: 'params' and/or 'camnames'")

    P_old = old["params"]
    P_new = new["params"]

    def read_camnames(m):
        v = np.squeeze(m["camnames"])
        lst = v.tolist() if isinstance(v.tolist(), list) else [v.tolist()]
        return [str(np.squeeze(x)) for x in lst]

    names_old = read_camnames(old)
    names_new = read_camnames(new)
    idx_new = {nm: i for i, nm in enumerate(names_new)}

    missing = [nm for nm in names_old if nm not in idx_new]
    if missing:
        raise ValueError(f"Cameras missing in new MAT: {missing}")

    # ---- helpers ----
    def is_1x1_obj(x):
        return isinstance(x, np.ndarray) and x.dtype == object and x.shape == (1, 1)

    def strip_one_layer(x):
        return x[0, 0] if is_1x1_obj(x) else x

    def get_record(P, i):
        if isinstance(P, np.ndarray):
            if P.ndim == 2 and P.shape[1] == 1:
                return P[i, 0]
            return P[i]
        return P[i]

    def strip_field_one_layer(rec, field):
        v = getattr(rec, field)
        v2 = strip_one_layer(v)
        if v2 is not v:
            setattr(rec, field, v2)

    # ⬇️ include r, t here
    def copy_intrinsics(rec_dst, rec_src, fields=("K", "RDistort", "TDistort", "r", "t")):
        for f in fields:
            setattr(rec_dst, f, strip_one_layer(getattr(rec_src, f)))

    # ---- 1) overwrite intrinsics + extrinsics ----
    for i, nm in enumerate(names_old):
        j = idx_new[nm]
        rec_old = strip_one_layer(get_record(P_old, i))
        rec_new = strip_one_layer(get_record(P_new, j))

        # if P_old[i] itself was a 1x1 wrapper, replace that slot with the unwrapped rec
        if isinstance(P_old, np.ndarray):
            if P_old.ndim == 2 and P_old.shape[1] == 1:
                if is_1x1_obj(P_old[i, 0]):
                    P_old[i, 0] = rec_old
            else:
                if is_1x1_obj(P_old[i]):
                    P_old[i] = rec_old

        copy_intrinsics(rec_old, rec_new)

    # ---- 2) strip one layer on params elements and key fields ----
    # ⬇️ include r, t here
    fields = ("K", "RDistort", "TDistort", "r", "t")
    if isinstance(P_old, np.ndarray):
        for idx in np.ndindex(P_old.shape):
            elem = P_old[idx]
            if is_1x1_obj(elem):
                P_old[idx] = elem[0, 0]
                elem = P_old[idx]
            for f in fields:
                strip_field_one_layer(elem, f)

    # ---- 2b) strip one layer on sync (calib['sync']) ----
    if "sync" in old:
        S_old = old["sync"]

        def strip_sync_record(rec):
            if hasattr(rec, "_fieldnames"):
                for f in rec._fieldnames:
                    v = getattr(rec, f)
                    if is_1x1_obj(v):
                        setattr(rec, f, v[0, 0])

        if isinstance(S_old, np.ndarray):
            for idx in np.ndindex(S_old.shape):
                elem = S_old[idx]
                if is_1x1_obj(elem):
                    S_old[idx] = elem[0, 0]
                    elem = S_old[idx]
                strip_sync_record(elem)
        else:
            if is_1x1_obj(S_old):
                old["sync"] = S_old[0, 0]
            strip_sync_record(old["sync"])

    # ---- 3) save updated as intrinsicsUpdated_<original>.mat ----
    d, b = os.path.dirname(old_mat), os.path.basename(old_mat)
    out_mat = os.path.join(d, f"df_intriUpdated_{b}")
    sio.savemat(out_mat, old, do_compression=True, long_field_names=True)

    # ---- 4) move original into prev_intrinsic_calib/ ----
    prev_dir = os.path.join(base_path, "prev_intrinsic_calib")
    os.makedirs(prev_dir, exist_ok=True)
    target = os.path.join(prev_dir, os.path.basename(old_mat))
    if os.path.exists(target):
        stem, ext = os.path.splitext(os.path.basename(old_mat))
        ts = time.strftime("%Y%m%d_%H%M%S")
        target = os.path.join(prev_dir, f"{stem}_{ts}{ext}")
    shutil.move(old_mat, target)
    print(f"Moved original to: {target}")

    return out_mat
