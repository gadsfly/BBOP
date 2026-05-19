import os, json, subprocess, tempfile, shutil
from pathlib import Path
import pandas as pd


# --- 1) Exact lookup with Buffer Index included ---
def map_exact_timestamps_to_frames_with_buffer(csv_path, timestamps_ms):
    """
    Exact lookup. Returns DataFrame:
      timestamp_ms, frame_number (global), buffer_idx
    Raises if any timestamp is missing.
    """
    df = pd.read_csv(csv_path, usecols=["Frame Number", "Time Stamp (ms)", "Buffer Index"])
    q = pd.DataFrame({"Time Stamp (ms)": list(timestamps_ms)})
    out = q.merge(df, on="Time Stamp (ms)", how="left").rename(
        columns={
            "Time Stamp (ms)": "timestamp_ms",
            "Frame Number": "frame_number",
            "Buffer Index": "buffer_idx",
        }
    )
    if out["frame_number"].isna().any() or out["buffer_idx"].isna().any():
        missing = out.loc[out["frame_number"].isna() | out["buffer_idx"].isna(), "timestamp_ms"].tolist()
        raise ValueError(f"Timestamps not found in CSV: {missing}")
    out["frame_number"] = out["frame_number"].astype(int)
    out["buffer_idx"] = out["buffer_idx"].astype(int)
    return out[["timestamp_ms", "frame_number", "buffer_idx"]]



# --- 2) Video writing by file index (fast, robust for FFV1) ---


def load_miniscope_meta(video_dir):
    meta = json.load(open(Path(video_dir) / "metaData.json", "r"))
    return {
        "frameRate": float(meta.get("frameRate", 30)),
        "framesPerFile": int(meta.get("framesPerFile", 1000)),
    }

def _ffprobe_nb_frames(path: Path) -> int:
    """
    Return the number of frames ffprobe sees (as int). Raises if file can't be probed.
    """
    cmd = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1",
        str(path)
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}:\n{cp.stderr}")
    s = cp.stdout.strip()
    try:
        return int(s)
    except Exception:
        # Sometimes nb_read_frames can be "N/A" for some codecs; fall back to pkt-based probe
        cmd2 = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames", "-of", "default=nokey=1:noprint_wrappers=1",
            str(path)
        ]
        cp2 = subprocess.run(cmd2, capture_output=True, text=True)
        if cp2.returncode != 0:
            raise RuntimeError(f"ffprobe (nb_frames) failed on {path}:\n{cp2.stderr}")
        s2 = cp2.stdout.strip()
        return int(s2) if s2.isdigit() else 0

def _build_select_eq(local_indices):
    """
    Build a comma-free select string: select='eq(n,idx1)+eq(n,idx2)+...'
    Avoids all comma-escaping pitfalls with subprocess.
    """
    terms = [f"eq(n,{int(i)})" for i in sorted(set(local_indices))]
    expr = "+".join(terms) if terms else "0"  # '0' -> selects nothing
    return f"select='{expr}'"

def write_frames_video_by_fileidx_ffmpeg_fast(
    video_dir, selections_df, out_path, fps=None, preset="ultrafast"
):
    """
    Fast FFmpeg path, robust for FFV1.
    - Derives file index from global frame number and framesPerFile.
    - Validates indices against actual nb_frames per file (drops OOR gracefully).
    - Uses select='eq(n,...) + eq(n,...)' to avoid comma escaping.
    """
    video_dir = Path(video_dir)
    os.makedirs(Path(out_path).parent, exist_ok=True)

    meta = load_miniscope_meta(video_dir)
    fps_use = float(meta["frameRate"] if fps is None else fps)
    fpf = int(meta["framesPerFile"])

    S = selections_df.copy()
    S["file_idx"] = (S["frame_number"] // fpf).astype(int)
    S["local"]    = (S["frame_number"] %  fpf).astype(int)
    S = S.sort_values("timestamp_ms").reset_index(drop=True)

    tmpdir = Path(tempfile.mkdtemp(prefix="mini_fast_"))
    extracted = {}  # (file_idx, local) -> path
    try:
        # 1) Batch-extract per file
        for fi, group in S.groupby("file_idx"):
            fi = int(fi)
            src = video_dir / f"{fi}.avi"
            if not src.exists():
                raise FileNotFoundError(f"Missing split file: {src}")

            # Probe actual frame count and clip requested locals to valid range
            try:
                nb = _ffprobe_nb_frames(src)
            except Exception as e:
                raise RuntimeError(f"ffprobe failed on {src}: {e}")

            locals_req = sorted(set(int(x) for x in group["local"].tolist()))
            locals_ok  = [x for x in locals_req if 0 <= x < nb]
            if not locals_ok:
                # Nothing valid in this file; skip
                continue

            vf = _build_select_eq(locals_ok)
            out_pat = tmpdir / f"buf{fi}_%06d.png"

            cmd = [
                "ffmpeg","-y","-hide_banner","-loglevel","error",
                "-i", str(src),
                "-vf", vf,
                "-vsync","vfr",
                str(out_pat)
            ]
            cp = subprocess.run(cmd, capture_output=True, text=True)
            if cp.returncode != 0:
                # Show helpful context
                raise RuntimeError(
                    f"ffmpeg extract failed for {src}\n"
                    f"vf={vf}\n"
                    f"stderr:\n{cp.stderr}"
                )

            # Map emitted images back to requested locals (emitted in ascending n)
            emitted = sorted(out_pat.parent.glob(f"buf{fi}_*.png"))
            if len(emitted) != len(locals_ok):
                # Not fatal; keep best-effort mapping, but warn in exception if zero
                if len(emitted) == 0:
                    raise RuntimeError(f"No frames emitted for {src} with vf={vf}")
                # Partial emission: zip shortest
            for loc, png in zip(locals_ok, emitted):
                extracted[(fi, int(loc))] = png

        # 2) Assemble in requested order (skipping any that failed extraction)
        seq_dir = tmpdir / "seq"
        seq_dir.mkdir(exist_ok=True)
        seq_files = []
        for i, row in S.iterrows():
            fi = int(row["file_idx"]); loc = int(row["local"])
            key = (fi, loc)
            if key not in extracted:
                continue
            src_png = extracted[key]
            dst = seq_dir / f"f_{len(seq_files)+1:06d}.png"
            try:
                os.link(src_png, dst)
            except Exception:
                shutil.copy2(src_png, dst)
            seq_files.append(dst)

        if not seq_files:
            raise RuntimeError("No frames extracted — check indices vs nb_frames and codec support.")

        # 3) Single encode pass
        cmd = [
            "ffmpeg","-y","-hide_banner","-loglevel","error",
            "-framerate", str(fps_use),
            "-i", str(seq_dir / "f_%06d.png"),
            "-c:v","libx264","-preset", preset, "-pix_fmt","yuv420p",
            str(out_path)
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed:\n{cp.stderr}")

        return {
            "requested": len(S),
            "rendered_frames": len(seq_files),
            "fps": fps_use,
            "preset": preset,
            "out": str(out_path),
        }
    except Exception:
        # keep tmpdir for inspection on error
        raise
