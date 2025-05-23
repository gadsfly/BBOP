import subprocess
from manual_config import sessions

for sess in sessions:
    session_dir = sess["session_dir"]

    # Determine which script to run. Default to the dynamic selective re-run script.
    script_to_run = sess.get("script", "selective_rerun_dynamic.py")

    # For the dynamic selective re-run, we use dot‑notation overrides.
    if script_to_run == "selective_rerun_dynamic.py":
        # Use "animal" from the config if provided; otherwise, default.
        animal = sess.get("animal", "UnknownAnimal")
        overrides = []
        for key, value in sess.items():
            if key in ["session_dir", "animal", "script"]:
                continue
            # If the key does not include a dot, assume it belongs to "param_seeds_init."
            if "." not in key:
                override_key = f"param_seeds_init.{key}"
            else:
                override_key = key
            overrides.extend(["--override", f"{override_key}={value}"])
        cmd = [
            "conda", "run", "-n", "minian", "python", script_to_run,
            "--session_dir", session_dir,
            "--animal", animal,
        ] + overrides

    else:
        # For other scripts, simply pass each key (except session_dir and script) as its own command-line flag.
        cmd = [
            "conda", "run", "-n", "minian", "python", script_to_run,
            "--session_dir", session_dir,
        ]
        # For non-dynamic scripts, we add any extra keys directly.
        
        for key, value in sess.items():
            if key in ["session_dir", "animal", "script"]:
                continue
            cmd.extend([f"--{key}", str(value)])
        # Optionally, add animal if provided.
        if "animal" in sess:
            cmd.extend(["--animal", sess["animal"]])

    print("---------------------------------------------------")
    print(f"Running for session: {session_dir}")
    print("Command:", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print("[STDOUT]", result.stdout)
    if result.stderr:
        print("[STDERR]", result.stderr)
