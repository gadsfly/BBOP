import yaml
from pathlib import Path
import subprocess
import itertools

# Path to the YAML configuration file
config_path = "/home/lq53/mir_repos/dappy_24_nov/byws_version/mir_1.yaml"

# Base output path
base_out_path = "/home/lq53/mir_repos/dappy_24_nov/byws_version/250109_opti_combinations/"

# Path to your main script
script_path = "/home/lq53/mir_repos/dappy_24_nov/byws_version/mir_runopti.py"  # Replace with the actual script path

# Feature combinations to test (adjust as needed)
features_to_test = [
    ["ego_pose", "angle", "velocity"],
    ["ego_pose", "angle", "velocity", "angular_velocity"],
    ["ego_pose", "angle", "velocity", "head_angular"],
    ["ego_pose", "angle", "velocity", "euler_angles"],
    ["ego_pose", "angle", "velocity", "angular_velocity", "head_angular"],
]

# Load the original configuration
with open(config_path, "r") as file:
    original_config = yaml.safe_load(file)

# Iterate over combinations of features
for i, features in enumerate(features_to_test):
    new_out_path = f"{base_out_path}combo_{i}/"
    original_config["out_path"] = new_out_path
    original_config["features_to_include"] = features  # Add this to your YAML

    # Ensure the output path exists
    Path(new_out_path).mkdir(parents=True, exist_ok=True)

    # Save the modified configuration back to the file
    with open(config_path, "w") as file:
        yaml.dump(original_config, file)

    # Run the main script
    subprocess.run(["python", script_path])

    print(f"Completed processing for feature combination {features}. Results saved to {new_out_path}.")

# Restore the original configuration
with open(config_path, "w") as file:
    yaml.dump(original_config, file)

print("Configuration restored to its original state.")
