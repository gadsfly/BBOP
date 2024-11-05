import glob
import subprocess

# Find directories containing COM and DANNCE
paths = glob.glob('/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/*_*_*/*/*COM*/') + \
        glob.glob('/hpc/group/tdunn/Bryan_Rigs/BigOpenField/24summ/*_*_*/*/*DANNCE*/')

# Save to a file for review or use with rsync
with open('/hpc/group/tdunn/lq53/BBOP/rec_rsync_useful/paths_to_copy.txt', 'w') as f:
    for path in paths:
        f.write(f"{path}\n")

# Run rsync for each path with --relative to keep the path structure
# for path in paths:
#     subprocess.run(['rsync', '-av', '--dry-run', '--relative', '--remove-source-files', 
#                     path, '/datacommons/tdunn/lq53'])

# then run in termianl of below:
# while read -r dir; do     rsync -av --dry-run --relative --remove-source-files "$dir" /datacommons/tdunn/lq53; done < /hpc/group/tdunn/lq53/BBOP/rec_rsync_useful/paths_to_copy.txt