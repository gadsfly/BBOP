import glob
import shutil
from pathlib import Path
import csv
# export_profiles.py
import os, re, csv, glob, fnmatch, shlex, subprocess, datetime as dt
import pyarrow as pa
import pyarrow.parquet as pq
import yaml



def _join(*a):
    return os.path.join(*[p for p in a if p not in (None, "", ".")])

def _ensure_dir(p):
    if p: os.makedirs(p, exist_ok=True)

def _run(cmd, dry):
    if dry:
        print("$", " ".join(shlex.quote(c) for c in cmd))
        return 0
    return subprocess.call(cmd)

def _append_parquet(path, new_table: pa.Table):
    if os.path.exists(path):
        old = pq.read_table(path)
        out = pa.concat_tables([old, new_table], promote_options="default")
    else:
        out = new_table
    pq.write_table(out, path)


def _normalize_log_path(path):
    if not path: return None
    if os.path.isdir(path): return os.path.join(path, "export_log.parquet")
    return path


def load_profiles(yaml_path):
    """
    YAML schema:

    profiles:
      paper-core:
        flags:
          include_videos: true
          include_visuals: false
        includes: []         # extra globs to add
        excludes: ["**/vis/**"]  # globs to remove
        conditions:          # optional row-aware tweaks
          - when: "social == 1"
            includes: ["MIR_Aligned/*with_ca*.h5"]
            excludes: []

      qc-review:
        flags: { include_videos: false, include_visuals: true }
        includes: ["COM/predict*/com3d*.mat"]
        excludes: []

      dannce-delta:
        flags: { include_videos: false, include_visuals: false }
        includes: ["DANNCE/predict01/save_data_AVG.mat"]
        excludes: ["DANNCE/predict00/**"]

      repro-full:
        flags: { include_videos: true, include_visuals: true }
        includes: []
        excludes: []

    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return (data.get("profiles") or {})

def _run_cleanup(export_root, cleanup_cfg, dry_run):
    """Remove old files from export directory before copying new ones."""
    mode = cleanup_cfg.get("mode", "trash")
    trash_dir = cleanup_cfg.get("trash_dir", ".trash")
    scope = cleanup_cfg.get("scope", [])
    protect = cleanup_cfg.get("protect", [])
    
    if not scope:
        return
    

    # Collect all paths to delete
    to_delete = []
    for pattern in scope:
        full_pattern = os.path.join(export_root, pattern)
        print(f"[CLEANUP DEBUG] Pattern: {pattern}")
        print(f"[CLEANUP DEBUG] Full pattern: {full_pattern}")
        
        matches = glob.glob(full_pattern, recursive=True)
        print(f"[CLEANUP DEBUG] Found {len(matches)} matches")
        for m in matches[:5]:  # Show first 5
            print(f"  - {m}")
        
        for filepath in matches:
            rel_path = os.path.relpath(filepath, export_root)
            is_protected = any(fnmatch.fnmatch(rel_path, p) for p in protect)
            
            if is_protected:
                print(f"[CLEANUP DEBUG] Protected: {rel_path}")
                continue
            
            to_delete.append((filepath, rel_path))
    
    print(f"\n[CLEANUP DEBUG] Total to delete: {len(to_delete)}")


    # Create trash directory with timestamp
    if mode == "trash":
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        trash_path = os.path.join(export_root, trash_dir, timestamp)
        if not dry_run:
            os.makedirs(trash_path, exist_ok=True)
    
    # Collect all paths to delete (files and dirs)
    to_delete = []
    for pattern in scope:
        full_pattern = os.path.join(export_root, pattern)
        matches = glob.glob(full_pattern, recursive=True)
        
        for filepath in matches:
            # Skip if protected
            rel_path = os.path.relpath(filepath, export_root)
            is_protected = any(fnmatch.fnmatch(rel_path, p) for p in protect)
            if is_protected:
                continue
            
            to_delete.append((filepath, rel_path))
    
    # Sort: files first (longer paths), then directories (shorter paths)
    # This ensures we delete contents before parent dirs
    to_delete.sort(key=lambda x: (-len(x[0]), x[0]))
    
    for filepath, rel_path in to_delete:
        if not os.path.exists(filepath):  # Already deleted as part of parent
            continue
            
        if mode == "trash":
            dest = os.path.join(trash_path, rel_path)
            print(f"[CLEANUP] Move to trash: {rel_path}")
            if not dry_run:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.move(filepath, dest)
        elif mode == "delete":
            print(f"[CLEANUP] Delete: {rel_path}")
            if not dry_run:
                if os.path.isdir(filepath):
                    shutil.rmtree(filepath)
                else:
                    os.remove(filepath)

def _extract_recording_date(path: str) -> str:
    """Extract recording date from the parent folder.
    
    Example: /data/.../2024_12_18/session_name -> 20241218
    """
    import re
    
    # Get parent folder name (the date folder)
    parent = os.path.basename(os.path.dirname(path.rstrip('/')))
    
    # Look for YYYY_MM_DD format
    match = re.match(r'(\d{4})_(\d{2})_(\d{2})', parent)
    if match:
        return match.group(1) + match.group(2) + match.group(3)
    
    # Fallback: look for YYYYMMDD in parent
    match = re.search(r'(20\d{6})', parent)
    if match:
        return match.group(1)
    
    # Last resort: today's date
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d")

def _post_copy_tidy(dest: str, dest_basename: str, dry_run: bool):
    """Reorganize exported folder after rsync.
    
    Changes:
    - MIR_Aligned/ -> metadata/
    - Rename df_*_label3d_dannce.mat -> label3d_dannce.mat
    - Rename overlay*.png -> overlay.png
    - Rename *.nc -> miniscope.nc
    """
    import shutil
    
    # 1. Move MIR_Aligned -> metadata
    mir_dir = os.path.join(dest, "MIR_Aligned")
    meta_dir = os.path.join(dest, "metadata")
    
    if os.path.exists(mir_dir):
        print(f"[TIDY] Moving MIR_Aligned/ -> metadata/")
        if not dry_run:
            _ensure_dir(meta_dir)
            for item in os.listdir(mir_dir):
                shutil.move(os.path.join(mir_dir, item), 
                           os.path.join(meta_dir, item))
            os.rmdir(mir_dir)
    
    # 2. Move folder_log.parquet to metadata/
    root_log = os.path.join(dest, "folder_log.parquet")
    if os.path.exists(root_log):
        print(f"[TIDY] Moving folder_log.parquet -> metadata/")
        if not dry_run:
            _ensure_dir(meta_dir)
            shutil.move(root_log, os.path.join(meta_dir, "folder_log.parquet"))
    
    # 3. Rename calibration file to simple name
    mat_files = glob.glob(os.path.join(dest, "df_*_label3d_dannce.mat"))
    if len(mat_files) == 1:
        old_name = mat_files[0]
        new_name = os.path.join(dest, "label3d_dannce.mat")
        print(f"[TIDY] Renaming {os.path.basename(old_name)} -> label3d_dannce.mat")
        if not dry_run:
            shutil.move(old_name, new_name)
    
    # 4. Rename overlay*.png to overlay.png in metadata/
    if os.path.exists(meta_dir):
        overlay_files = glob.glob(os.path.join(meta_dir, "overlay*.png"))
        if len(overlay_files) == 1:
            old_name = overlay_files[0]
            new_name = os.path.join(meta_dir, "overlay.png")
            print(f"[TIDY] Renaming {os.path.basename(old_name)} -> overlay.png")
            if not dry_run:
                shutil.move(old_name, new_name)
        elif len(overlay_files) > 1:
            print(f"[TIDY WARNING] Multiple overlay files found, skipping rename: {[os.path.basename(f) for f in overlay_files]}")
    
    # 5. Rename *.nc to miniscope.nc in miniscope/
    mini_dir = os.path.join(dest, "miniscope")
    if os.path.exists(mini_dir):
        nc_files = glob.glob(os.path.join(mini_dir, "*.nc"))
        if len(nc_files) == 1:
            old_name = nc_files[0]
            new_name = os.path.join(mini_dir, "miniscope.nc")
            print(f"[TIDY] Renaming {os.path.basename(old_name)} -> miniscope.nc")
            if not dry_run:
                shutil.move(old_name, new_name)
        elif len(nc_files) > 1:
            print(f"[TIDY WARNING] Multiple .nc files found, skipping rename: {[os.path.basename(f) for f in nc_files]}")

def export_with_rsync_preserve_tree(
    plan,
    export_root,
    profile,
    dry_run=True,
    mode="snapshot",
    dataset_version=None,
    profile_label="default",
    log_parquet_path=None,
    log_on_dry_run=False,
    copy_miniscope=False,  # ← Explicit flag, default OFF
    mini_csv_paths=None,
    mini_includes=None,
):
    if mode not in {"staged","snapshot"}:
        raise ValueError("mode must be 'staged' or 'snapshot'")
    root = _join(export_root, f"v{dataset_version}") if (mode=="snapshot" and dataset_version) else export_root

    # Load miniscope mappings ONLY if copy_miniscope=True
    mini_mapping = {}
    if copy_miniscope and mini_csv_paths:
        import pandas as pd
        for csv_path in mini_csv_paths:
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    mini_mapping[row['rec_path']] = row['nc_file']

    # Cleanup
    cleanup_cfg = profile.get("cleanup", {})
    if cleanup_cfg.get("enabled", False):
        _run_cleanup(root, cleanup_cfg, dry_run)

    log_rows = []
    for item in plan:
        src_root = item["source_rec_path"].rstrip("/")
        
        # Add date prefix
        dest_rel = item["dest_rel_path"]
        rec_date = _extract_recording_date(src_root)
        dest_basename = os.path.basename(dest_rel.rstrip('/'))
        
        if not dest_basename.startswith(rec_date):
            parent = os.path.dirname(dest_rel.rstrip('/'))
            dest_rel = os.path.join(parent, f"{rec_date}_{dest_basename}") + "/"
        
        dest = _join(root, dest_rel)
        dest_folder_name = os.path.basename(dest.rstrip('/'))

        if not dry_run:
            _ensure_dir(dest)

        # Main rsync
        includes = item.get("path_globs") or profile.get("includes", []) or []
        excludes = item.get("excludes", []) or profile.get("excludes", []) or []
        norm = lambda p: p[2:] if p.startswith("./") else p

        cmd = ["rsync", "-a", "--prune-empty-dirs"]
        if dry_run: cmd.append("-n")

        # ADD EXCLUDES FIRST
        for p in excludes: cmd += ["--exclude", norm(p)]

        # THEN INCLUDES
        cmd.append("--include")
        cmd.append("*/")
        for p in includes: cmd += ["--include", norm(p)]

        cmd += ["--exclude", "*", src_root + "/", dest + "/"]
        _run(cmd, dry_run)

        # Miniscope copying (ONLY if explicitly enabled)
        # Default miniscope files to copy
        if mini_includes is None:
            mini_includes = ["*.avi"]  # Just videos by default
        
        # ... existing code until miniscope section ...
        
        if copy_miniscope:
            txt_file = os.path.join(src_root, "sync_to_mini_path.txt")
            if os.path.exists(txt_file):
                with open(txt_file, 'r') as f:
                    mini_path = f.read().strip()
                
                mini_folder = os.path.join(mini_path, "My_V4_Miniscope")
                nc_key = mini_mapping.get(mini_folder)
                
                if nc_key:
                    nc_pattern = os.path.join(mini_folder, f"*{nc_key}*.nc")
                    nc_files = glob.glob(nc_pattern)
                    
                    if nc_files:
                        dest_mini = os.path.join(dest, "miniscope") #My_V4_Miniscope
                        print(f"[MINI] {mini_folder} -> {dest_mini}")
                        
                        if not dry_run:
                            _ensure_dir(dest_mini)
                        
                        # Build rsync command with custom includes
                        cmd = ["rsync", "-a"]
                        if dry_run: cmd.append("-n")
                        
                        # Add all include patterns
                        for pattern in mini_includes:
                            cmd += ["--include", pattern]
                        
                        cmd += ["--exclude", "*", mini_folder + "/", dest_mini + "/"]
                        _run(cmd, dry_run)
                        
                        # Copy the .nc file separately (since it's matched by key)
                        cmd_nc = ["rsync", "-a"]
                        if dry_run: cmd_nc.append("-n")
                        cmd_nc += [nc_files[0], dest_mini + "/"]
                        _run(cmd_nc, dry_run)
        
        _post_copy_tidy(dest, dest_folder_name, dry_run)
        
        log_rows.append({
            "export_time_utc": dt.datetime.utcnow().isoformat(timespec="seconds"),
            "profile_label": profile_label,
            "export_mode": mode,
            "dataset_version": dataset_version or "",
            "source_rec_path": src_root,
            "dest_rel_path": dest_rel,
            "id": item["id"],
            "is_social": int(item.get("is_social", 1)),
            "dry_run": bool(dry_run),
        })

    if log_parquet_path and (not dry_run or log_on_dry_run):
        log_path = _normalize_log_path(log_parquet_path)
        _ensure_dir(os.path.dirname(log_path) or ".")
        _append_parquet(log_path, pa.Table.from_pylist(log_rows))