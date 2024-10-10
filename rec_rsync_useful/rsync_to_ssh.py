import subprocess
import os

def sync_entire_folder(base_dir, cluster_base_dir, dry_run=True):
    """Sync the entire folder structure and then process."""
    rsync_command = [
        "rsync", "--mkpath", "-av", "--ignore-existing", base_dir + '/', cluster_base_dir,
        # "--exclude=auto_failed.py",
        # "--exclude=/*/"
        "--include=/*/*/*/metaData.json",
        "--include=/*/*/*/My_V4_Miniscope/",
        "--include=/*/*/*/My_First_WebCam/0.avi",
        "--exclude=/*/*/*/My_First_WebCam/timeStamps.csv",
        "--exclude=/*/*/*/My_First_WebCam/metaData.json",
        "--exclude=/*/*/*/behaviorTracker/",  # Exclude entire behaviorTracker folder
        "--exclude=/*/*/*/experiment/",  # Exclude entire experiment folder
        "--exclude=/*/*/*/My_V4_Miniscope/headOrientation.csv",  # Exclude specific file
        "--exclude=/*/*/*/notes.csv",  # Exclude specific file
        
        "--exclude=/*/*/*/*"  # Exclude all other files and folders at this level
    ]



    if dry_run:
        rsync_command.append("--dry-run")

    try:
        print(f"Syncing {base_dir} to {cluster_base_dir}")
        subprocess.run(rsync_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error syncing: {e}")

if __name__ == "__main__":
    base_dir = "/mnt/h/241010_mir_auto_sync_test"
    cluster_base_dir = "lq53@dcc-login.oit.duke.edu:/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24oct_bbop_test/bryan_rec_miniscopo_test"
    
    dry_run = True  # Set to False to perform the actual sync
    sync_entire_folder(base_dir, cluster_base_dir, dry_run)
