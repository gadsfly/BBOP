import os
import subprocess
import argparse
import csv
import json
from datetime import datetime
import pathlib

def nested_update(d, key_str, value):
    """
    Update a nested dictionary d using a dot-separated key_str.
    For example, key_str "param_seeds_init.wnd_size" updates:
      d["param_seeds_init"]["wnd_size"] = value
    """
    keys = key_str.split('.')
    current = d
    for k in keys[:-1]:
        if k not in current or not isinstance(current[k], dict):
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value

def generate_param_file(session_dir, config, unique_id):
    """
    Write a Python parameter file based on the config dictionary.
    All keys in config will be written as assignments.
    Then, extra code (from your original template) is appended.
    """
    lines = []
    lines.append("import os")
    lines.append("import numpy as np")
    lines.append("import pathlib")
    lines.append("")
    # Write each key/value as a Python assignment.
    for key, value in config.items():
        lines.append(f"{key} = {repr(value)}")
    lines.append("")
    # Append extra code (the parts that were in your original template)
    lines.append("dpath = os.path.dirname(pathlib.Path(__file__).resolve())")
    lines.append('minian_ds_path = os.path.join(dpath, "minian")')
    lines.append('intpath = os.path.join(dpath, "minian_intermediate")')
    lines.append(f'nc_file_name = os.path.join(dpath, "minian_dataset_{unique_id}.nc")')
    lines.append("")
    lines.append('os.environ["OMP_NUM_THREADS"] = "1"')
    lines.append('os.environ["MKL_NUM_THREADS"] = "1"')
    lines.append('os.environ["OPENBLAS_NUM_THREADS"] = "1"')
    lines.append('os.environ["MINIAN_INTERMEDIATE"] = intpath')
    lines.append("")
    param_file_path = os.path.join(session_dir, "minian_param_mir.py")
    with open(param_file_path, "w") as f:
        f.write("\n".join(lines))
    return param_file_path

def selective_rerun(session_dir, animal, config_overrides):
    """
    Run the analysis with a dynamically adjustable parameter configuration.
    
    Parameters:
        session_dir (str): Path to the session directory.
        animal (str): Animal/subject identifier.
        config_overrides (dict): A dictionary of parameter overrides.
           (Nested keys can be updated using dot-notation via command-line.)
    """
    # --- Define a default configuration ---
    default_config = {
        "param_seeds_init": {
            "wnd_size": 1500,
            "stp_size": 700,
            "max_wnd": 25,
            "diff_thres": 3.0,
            "method": "rolling"
        },
        "param_denoise": {"method": "median", "ksize": 5},
        "param_background_removal": {"method": "tophat", "wnd": 15},
        "noise_freq": 0.01,
        "sparse_penal": 0.01,
        "param_pnr_refine": {"noise_freq": 0.01, "thres": "auto"},
        "param_ks_refine": {"sig": 0.05},
        "param_seeds_merge": {"thres_dist": 10, "thres_corr": 0.8, "noise_freq": 0.01},
        "param_initialize": {"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.01},
        "param_init_merge": {"thres_corr": 0.8},
        "param_get_noise": {"noise_range": (0.01, 0.5)},
        "param_first_spatial": {"dl_wnd": 10, "sparse_penal": 0.01, "size_thres": (25, None)},
        "param_first_merge": {"thres_corr": 0.8},
        "minian_path": "/home/lq53/mir_repos/minian/minian",
        "subset": {"frame": slice(0, None)},
        "subset_mc": None,
        "interactive": True,
        "output_size": 100,
        "n_workers": 4,
        "param_save_minian": {
            "dpath": "minian_ds_path",
            "meta_dict": {"session": -1, "animal": -2},
            "overwrite": True
        },
        "param_load_videos": {
            "pattern": "[0-9]+\\.avi$",
            "dtype": "np.uint8",
            "downsample": {"frame": 1, "height": 1, "width": 1},
            "downsample_strategy": "subset"
        },
        "param_estimate_motion": {"dim": "frame"}
    }
    
    # --- Apply file- and command-line-based overrides ---
    for key, value in config_overrides.items():
        if "." in key:
            nested_update(default_config, key, value)
        else:
            default_config[key] = value

    # --- Create a unique identifier for this run ---
    seeds = default_config.get("param_seeds_init", {})
    unique_id = f"selective_wnd{seeds.get('wnd_size','NA')}_diff{seeds.get('diff_thres','NA')}"
    print("Running selective re-run with configuration:")
    import json
    # print(json.dumps(default_config, indent=4))
    print(json.dumps(default_config, indent=4, default=str))

    
    # --- Generate the parameter file (with extra code appended) ---
    param_file = generate_param_file(session_dir, default_config, unique_id)
    print("Parameter file generated at:", param_file)
    
    # --- Run the main analysis script (assumed to be 'minian_mirpy_set.py') ---
    result = subprocess.run(["python", "minian_mirpy_set.py"],
                            cwd=session_dir, capture_output=True, text=True)
    if result.stdout:
        print("[STDOUT]", result.stdout)
    if result.stderr:
        print("[STDERR]", result.stderr)
    
    # --- Check for the output file ---
    output_nc = os.path.join(session_dir, f"minian_dataset_{unique_id}.nc")
    if os.path.exists(output_nc):
        print(f"Output file generated: {output_nc}")
    else:
        print(f"Warning: Output file not found for combination {unique_id}.")
        output_nc = "Not produced"

    # --- Log the run ---
    log_file = os.path.join(session_dir, "selective_rerun_log.csv")
    file_exists = os.path.exists(log_file)
    with open(log_file, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "unique_id", "config", "result_file"])
        timestamp = datetime.now().isoformat()
        # writer.writerow([timestamp, unique_id, json.dumps(default_config), output_nc])
        writer.writerow([timestamp, unique_id, json.dumps(default_config, default=str), output_nc])

    print("Selective re-run logged.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Selective re-run with dynamic parameter adjustments"
    )
    parser.add_argument("--session_dir", required=True, help="Path to the session directory.")
    parser.add_argument("--animal", default="UnknownAnimal", help="Animal ID (optional).")
    parser.add_argument("--config_file", help="Path to a JSON file with parameter overrides.")
    parser.add_argument("--override", action="append", help="Override a specific parameter (use dot-notation for nested keys, e.g., param_seeds_init.wnd_size=700). Can be used multiple times.")
    
    args = parser.parse_args()

    # Build the configuration overrides.
    config_overrides = {}
    if args.config_file:
        try:
            with open(args.config_file, "r") as f:
                file_overrides = json.load(f)
            for k, v in file_overrides.items():
                config_overrides[k] = v
        except Exception as e:
            print("Error reading config file:", e)

    if args.override:
        for override in args.override:
            if "=" not in override:
                print("Override must be in key=value format. Skipping:", override)
                continue
            key_str, value_str = override.split("=", 1)
            try:
                value_converted = int(value_str)
            except ValueError:
                try:
                    value_converted = float(value_str)
                except ValueError:
                    value_converted = value_str
            config_overrides[key_str] = value_converted

    selective_rerun(args.session_dir, args.animal, config_overrides)
