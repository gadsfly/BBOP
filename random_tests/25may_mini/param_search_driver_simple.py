import os
import subprocess
import itertools
import csv
from datetime import datetime
import argparse

def main(session_dir, animal):
    """
    session_dir: path to the folder where data/analysis should be done
    animal:      an identifier for the subject/animal
    """
    # === Parameter Grid Setup ===
    wnd_size_vals    = [1500] #700, 
    stp_size_vals    = [700]  # Only one value
    diff_thres_vals  = [3.5, 4.0, 5.0] #3.5 4.0,  3.0, 
    pnr_thresh_vals  = [1.1, "auto"]

    # Decide max_wnd based on session_dir content.
    # (Adjust these conditions as you see fit.)
    if "pmc" in session_dir.lower():
        max_wnd_vals = [15]
    elif "v1" in session_dir.lower():
        max_wnd_vals = [25]
    else:
        max_wnd_vals = [25]

    # === Log File Setup ===
    log_file = os.path.join(session_dir, "param_search_log.csv")
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp",
                             "combination",
                             "wnd_size",
                             "stp_size",
                             "max_wnd",
                             "diff_thres",
                             "pnr_thresh",
                             "result_file"])

    # === Parameter File Template ===
    param_template = r"""
import os
import numpy as np
import pathlib
# import datetime

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

param_pnr_refine = {{"noise_freq": noise_freq, "thres": {pnr_thresh} }}  # try 1.1 or "auto"
param_ks_refine = {{"sig": 0.05}}
param_seeds_merge = {{"thres_dist": 10, "thres_corr": 0.8, "noise_freq": 0.01}}
param_initialize = {{"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.01}}
param_init_merge = {{"thres_corr": 0.8}}

# CNMF Parameters#
param_get_noise = {{"noise_range": (0.01, 0.5)}}
param_first_spatial = {{
    "dl_wnd": 10,
    "sparse_penal": sparse_penal,  # remains at 0.01
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

    # === Grid Search Loop ===
    for wnd_size, stp_size, max_wnd, diff_thres, pnr_thresh in itertools.product(
        wnd_size_vals, stp_size_vals, max_wnd_vals, diff_thres_vals, pnr_thresh_vals
    ):
        # Properly format pnr_thresh for insertion into the Python code
        if isinstance(pnr_thresh, str):
            pnr_thresh_str = f'"{pnr_thresh}"'
        else:
            pnr_thresh_str = pnr_thresh

        combination_str = f"wnd{wnd_size}_stp{stp_size}_max{max_wnd}_diff{diff_thres}_pnr{pnr_thresh}"
        print(f"Running combination: {combination_str}")

        # Generate the text of the param file
        param_content = param_template.format(
            wnd_size=wnd_size,
            stp_size=stp_size,
            max_wnd=max_wnd,
            diff_thres=diff_thres,
            pnr_thresh=pnr_thresh_str,
            unique_id=combination_str
        )

        # Write the param file in the session_dir
        param_file_path = os.path.join(session_dir, "minian_param_mir.py")
        with open(param_file_path, "w") as f:
            f.write(param_content)

        # Run the main analysis script (e.g. minian_mirpy_set.py) in session_dir
        # Adjust as needed for your environment. 
        result = subprocess.run(["python", "minian_mirpy_set.py"],
                                cwd=session_dir, capture_output=True, text=True)
        # Capture output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)

        # Check if output file was generated
        output_nc = os.path.join(session_dir, f"minian_dataset_{combination_str}.nc")
        if not os.path.exists(output_nc):
            print(f"Warning: Output file not found for combination {combination_str}.")
            output_nc = "Not produced"

        # Write the attempt to the CSV log
        timestamp = datetime.now().isoformat()
        with open(log_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp,
                             combination_str,
                             wnd_size,
                             stp_size,
                             max_wnd,
                             diff_thres,
                             pnr_thresh,
                             output_nc])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parameter search for a given session.")
    parser.add_argument("--session_dir", required=True, help="Path to the session directory.")
    parser.add_argument("--animal", default="UnknownAnimal", help="Animal ID (optional).")

    args = parser.parse_args()

    main(args.session_dir, args.animal)
