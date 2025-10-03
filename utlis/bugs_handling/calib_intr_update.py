import os
import csv
from typing import Iterable, Dict, Optional, Tuple

def append_session_offsets_12(
    base_path: str,
    out_csv: str,
    camera_indices: Iterable[int] = range(1, 7),
    encoding: str = "utf-8",
) -> Dict[int, Optional[Tuple[int, int]]]:
    """
    Read offsetX/offsetY from /videos/Camera{i}/metadata.csv for i in camera_indices.
    Append a single CSV row with columns:
        base_path, C1_offsetX, C1_offsetY, ..., C6_offsetX, C6_offsetY

    Returns a dict {camera_index: (offsetX, offsetY) or None if missing}.
    """
    def parse_meta(meta_path: str) -> Optional[Tuple[int, int]]:
        if not os.path.isfile(meta_path):
            return None
        vals = {}
        with open(meta_path, "r", newline="", encoding=encoding) as f:
            rdr = csv.reader(f)
            for row in rdr:
                if len(row) < 2:
                    continue
                k = row[0].strip().strip('"').strip("'")
                v = row[1].strip().strip('"').strip("'")
                if k in ("offsetX", "offsetY"):
                    try:
                        vals[k] = int(v)
                    except ValueError:
                        vals[k] = int(float(v))  # tolerate "204.0"
        if "offsetX" in vals and "offsetY" in vals:
            return vals["offsetX"], vals["offsetY"]
        return None

    # collect per-camera offsets
    per_cam: Dict[int, Optional[Tuple[int, int]]] = {}
    for i in camera_indices:
        meta = os.path.join(base_path, "videos", f"Camera{i}", "metadata.csv")
        per_cam[i] = parse_meta(meta)

    # ensure parent dir exists
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # build header and row (interleaved X,Y per camera)
    cam_cols = []
    for i in camera_indices:
        cam_cols += [f"C{i}_offsetX", f"C{i}_offsetY"]
    header = ["base_path"] + cam_cols

    row = [base_path]
    for i in camera_indices:
        pair = per_cam[i]
        if pair is None:
            row += ["", ""]  # blank if missing
        else:
            row += [pair[0], pair[1]]

    # write (create header if file missing)
    write_header = not os.path.isfile(out_csv)
    with open(out_csv, "a", newline="", encoding=encoding) as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

    return per_cam
