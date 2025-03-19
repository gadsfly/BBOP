import os
import subprocess
import argparse
import csv
from datetime import datetime

# Parameter file template – adjust as needed.
param_template = r"""
import os
import numpy as np
import pathlib

param_seeds_init = {{
    'wnd_size': {wnd_size},  # window size
    'method': 'rolling',
    'stp_size': {stp_size},  # step size
    'max_wnd': {max_wnd},    # maximum window (set based on session type)
    'diff_thres': {diff_thres}  # difference threshold
}}

param_denoise = {{"method": "median", "ksize": 5}}
param_background_removal = {{"method": "tophat", "wnd": 15}}
noise_freq = 0.01
sparse_penal = 0.01

param_pnr_refine = {{"noise_freq": noise_freq, "thres": {pnr_thresh} }}  # try e.g., 1.1 or "auto"
param_ks_refine = {{"sig": 0.05}}
param_seeds_merge = {{"thres_dist": 10, "thres_corr": 0.8, "noise_freq": 0.01}}
param_initialize = {{"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.01}}
param_init_merge = {{"thres_corr": 0.8}}

# CNMF Parameters#
param_get_noise = {{"noise_range": (0.01, 0.5)}}
param_first_spatial = {{
    "dl_wnd": 10,
    "sparse_penal": sparse_penal,
    "size_thres": (25, None),
}}

param_first_merge = {{"thres_corr": 0.8}}

minian_path = "/home/lq53/mir_repos/minian/minian"

dpath = os.path.dirname(pathlib.Path(__file__).resolve())
minian_ds_path = os.path.join(dpath, "minian")
intpath = os.path.join(dpath, "minian_intermediate")
nc_file_name = os.path.join(dpath, "minian_dataset_{unique_id}.nc")

subset = dict(frame=slice(0, None))
subset_mc = None
interactive = True
output_size = 100
n_workers = int(os.getenv("MINIAN_NWORKERS", 4))
param_save_minian = {{
    "dpath": minian_ds_path,
    "meta_dict": dict(session=-1, animal=-2),
    "overwrite": True,
}}

param_load_videos = {{
    "pattern": "[0-9]+\\.avi$",
    "dtype": np.uint8,
    "downsample": dict(frame=1, height=1, width=1),
    "downsample_strategy": "subset",
}}

subset_mc = None
param_estimate_motion = {{"dim": "frame"}}

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MINIAN_INTERMEDIATE"] = intpath
"""

def selective_rerun(session_dir, animal, wnd_size, stp_size, max_wnd, diff_thres, pnr_thresh):
    """
    Run the analysis for a manually specified parameter set.
    
    Parameters:
        session_dir (str): Path to the session directory.
        animal (str): Animal/subject identifier.
        wnd_size (int): Window size.
        stp_size (int): Step size.
        max_wnd (int): Maximum window.
        diff_thres (float): Difference threshold.
        pnr_thresh (str or float): PNR threshold (e.g., 1.1 or "auto").
    """
    # Determine a unique identifier for this run.
    combination_str = f"selective_wnd{wnd_size}_stp{stp_size}_max{max_wnd}_diff{diff_thres}_pnr{pnr_thresh}"
    print(f"Running selective re-run: {combination_str}")

    # Ensure pnr_thresh is formatted correctly.
    try:
        # If pnr_thresh can be cast to float, leave unquoted.
        float(pnr_thresh)
        pnr_thresh_str = pnr_thresh
    except ValueError:
        # Else, assume it is non-numeric and quote it.
        pnr_thresh_str = f'"{pnr_thresh}"'

    # Generate the parameter file content.
    param_content = param_template.format(
        wnd_size=wnd_size,
        stp_size=stp_size,
        max_wnd=max_wnd,
        diff_thres=diff_thres,
        pnr_thresh=pnr_thresh_str,
        unique_id=combination_str
    )

    # Write the parameter file.
    param_file_path = os.path.join(session_dir, "minian_param_mir.py")
    with open(param_file_path, "w") as f:
        f.write(param_content)
    print(f"Parameter file written to {param_file_path}")

    # Run the main analysis script (assumed to be 'minian_mirpy_set.py') in the session directory.
    result = subprocess.run(["python", "minian_mirpy_set.py"],
                            cwd=session_dir, capture_output=True, text=True)
    if result.stdout:
        print("[STDOUT]", result.stdout)
    if result.stderr:
        print("[STDERR]", result.stderr)

    # Determine the expected output file name.
    output_nc = os.path.join(session_dir, f"minian_dataset_{combination_str}.nc")
    if os.path.exists(output_nc):
        print(f"Output file generated: {output_nc}")
    else:
        print(f"Warning: Output file not found for combination {combination_str}.")
        output_nc = "Not produced"

    # Append details to a log file.
    log_file = os.path.join(session_dir, "selective_rerun_log.csv")
    file_exists = os.path.exists(log_file)
    with open(log_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "combination", "wnd_size", "stp_size", "max_wnd",
                             "diff_thres", "pnr_thresh", "result_file"])
        timestamp = datetime.now().isoformat()
        writer.writerow([timestamp, combination_str, wnd_size, stp_size, max_wnd,
                         diff_thres, pnr_thresh, output_nc])
    print("Selective re-run logged.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run selective re-run with specified parameters for a session.")
    parser.add_argument("--session_dir", required=True, help="Path to the session directory.")
    parser.add_argument("--animal", default="UnknownAnimal", help="Animal ID (optional).")
    parser.add_argument("--wnd_size", type=int, required=True, help="Window size parameter (e.g., 700).")
    parser.add_argument("--stp_size", type=int, required=True, help="Step size parameter (e.g., 700).")
    parser.add_argument("--max_wnd", type=int, required=True, help="Maximum window parameter (e.g., 25).")
    parser.add_argument("--diff_thres", type=float, required=True, help="Difference threshold (e.g., 4.0).")
    parser.add_argument("--pnr_thresh", type=str, required=True, help='PNR threshold (e.g., 1.1 or "auto").')

    args = parser.parse_args()

    selective_rerun(args.session_dir, args.animal,
                    args.wnd_size, args.stp_size,
                    args.max_wnd, args.diff_thres,
                    args.pnr_thresh)
