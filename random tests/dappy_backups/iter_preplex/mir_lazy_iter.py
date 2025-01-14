import yaml
from pathlib import Path
import subprocess

# Path to the YAML configuration file
config_path = "/home/lq53/mir_repos/dappy_24_nov/byws_version/mir_1.yaml"

# Base output path
base_out_path = "/home/lq53/mir_repos/dappy_24_nov/byws_version/250109_opti_ang_vel/"

# List of perplexities to iterate over
perplexities = [50, 55] #[100, 200, 300, 400, 500] #[70, 80, 90, 40]

# Path to your main script
script_path = "/home/lq53/mir_repos/dappy_24_nov/byws_version/mir_runopti.py"  # Replace with the actual script path

# Load the original configuration
with open(config_path, "r") as file:
    original_config = yaml.safe_load(file)

for perplexity in perplexities:
    # Modify the configuration
    new_out_path = f"{base_out_path}{perplexity}_p_velo/"
    original_config["out_path"] = new_out_path
    original_config["single_embed"]["perplexity"] = perplexity

    # Ensure the output path exists
    Path(new_out_path).mkdir(parents=True, exist_ok=True)

    # Save the modified configuration back to the original file
    with open(config_path, "w") as file:
        yaml.dump(original_config, file)

    # Run the script
    subprocess.run(["python", script_path])

    print(f"Completed processing for perplexity {perplexity}. Results saved to {new_out_path}.")

# Restore the original configuration
with open(config_path, "w") as file:
    yaml.dump(original_config, file)

print("Configuration restored to its original state.")
