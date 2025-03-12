import subprocess
import os

def sync_entire_folder(base_dir, cluster_base_dir, dry_run=True):
    """Sync the entire folder structure and then process."""
    rsync_command = [
        "rsync", "--update", "-av", "--ignore-existing", base_dir + '/', cluster_base_dir, #--mkpath
        # "--exclude=auto_failed.py",
        # "--exclude=/*/"
        # "--include=/*/*/*/metaData.json",
        "--exclude=/*/*/*/*/My_V4_Miniscope/__pycache__",
        "--exclude=/*/*/*/*/My_V4_Miniscope/dask-worker-space",
        "--exclude=/*/*/*/*/My_V4_Miniscope/minian",
        "--exclude=/*/*/*/*/My_V4_Miniscope/minian_intermediate",
        "--exclude=/*/*/*/*/My_V4_Miniscope/.ipynb_checkpoints",
        "--exclude=/*/*/*/*/My_V4_Miniscope/minian_mc.mp4",
        "--exclude=/*/*/*/*/My_V4_Miniscope/minian_param_mir.py", # can uncomment and use later if needed
        "--exclude=/*/*/*/*/My_V4_Miniscope/minian_mirpy_set.py",
        # "--exclude=/*/*/*/My_V4_Miniscope/__pycache__",
        # "--exclude=/*/*/*/My_V4_Miniscope/__pycache__",
        # "--include=/*/*/*/My_First_WebCam/",
        # "--exclude=/*/*/*/My_First_WebCam/*[1-9]*.avi",
        # # "--include=/*/*/*/My_First_WebCam/timeStamps.csv",
        # # "--include=/*/*/*/My_First_WebCam/metaData.json",
        # "--exclude=/*/*/*/behaviorTracker/",  # Exclude entire behaviorTracker folder
        # "--exclude=/*/*/*/experiment/",  # Exclude entire experiment folder
        # "--exclude=/*/*/*/My_V4_Miniscope/headOrientation.csv",  # Exclude specific file
        # "--exclude=/*/*/*/notes.csv",  # Exclude specific file
        # "--exclude=/*/*/*/*"  # Exclude all other files and folders at this level
    ]



    if dry_run:
        rsync_command.append("--dry-run")

    try:
        print(f"Syncing {base_dir} to {cluster_base_dir}")
        subprocess.run(rsync_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error syncing: {e}")

if __name__ == "__main__":
    base_dir = "/data/big_rim/rsync_dcc_sum/Oct3V1mini_sorted"
    #"/home/lq53/mir_data/Oct3V1mini/Oct3V1mini_sorted" #"/mnt/h/241010_mir_auto_sync_test"
    cluster_base_dir = "lq53@dcc-login.oit.duke.edu:/hpc/group/tdunn/Bryan_Rigs/BigOpenField/Oct3V1mini_sorted"
    
    dry_run = True  #False # Set to False to perform the actual sync
    sync_entire_folder(base_dir, cluster_base_dir, dry_run)
