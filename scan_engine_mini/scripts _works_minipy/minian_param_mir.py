import os
import numpy as np
import pathlib
# import datetime

# param_seeds_init['max_wnd']= 25 #25 good for V1. default 15, the maxium size of a neuron, which should zoom in and briefly determine
param_seeds_init = {'wnd_size': 700, #1500, smaller windows would potential generate more seeds, so maybe would have included some potential neurons.
 'method': 'rolling',
 'stp_size': 700, #700
 'max_wnd': 25, #25 works well for V1. now trying 15 for PMC
 'diff_thres': 3.5} #3.5, 6

param_denoise = {"method": "median", "ksize": 5}
param_background_removal = {"method": "tophat", "wnd": 15}
noise_freq = 0.005
sparse_penal = 0.01



param_pnr_refine = {"noise_freq": noise_freq, "thres": 0.9}
param_ks_refine = {"sig": 0.05}
param_seeds_merge = {"thres_dist": 10, "thres_corr": 0.8, "noise_freq": noise_freq}
param_initialize = {"thres_corr": 0.8, "wnd": 10, "noise_freq": noise_freq}
param_init_merge = {"thres_corr": 0.8}

# CNMF Parameters#
param_get_noise = {"noise_range": (noise_freq, 0.5)}
param_first_spatial = {
    "dl_wnd": 10,
    "sparse_penal": sparse_penal,
    "size_thres": (25, None),
}

param_first_merge = {"thres_corr": 0.8}

# Set up Initial Basic Parameters#
minian_path = "/home/lq53/mir_repos/minian/minian"


dpath =  os.path.dirname(pathlib.Path(__file__).resolve()) #"./" ./ will return terminal path, interesting
minian_ds_path = os.path.join(dpath, "minian")
intpath = os.path.join(dpath, "minian_intermediate") # same thing here"./minian_intermediate"
nc_file_name = os.path.join(dpath,"minian_dataset.nc")

subset = dict(frame=slice(0, None))
subset_mc = None
interactive = True
output_size = 100
n_workers = int(os.getenv("MINIAN_NWORKERS", 4)) #4
param_save_minian = {
    "dpath": minian_ds_path,
    "meta_dict": dict(session=-1, animal=-2),
    "overwrite": True,
}

# Pre-processing Parameters#
param_load_videos = {
    "pattern": "[0-9]+\.avi$",
    "dtype": np.uint8,
    "downsample": dict(frame=1, height=1, width=1),
    "downsample_strategy": "subset",
}


# Motion Correction Parameters#
subset_mc = None
param_estimate_motion = {"dim": "frame"}

# # Initialization Parameters#
# param_seeds_init = {
#     "wnd_size": 1000,
#     "method": "rolling",
#     "stp_size": 500,
#     "max_wnd": 15,
#     "diff_thres": 3,
# }
# param_pnr_refine = {"noise_freq": 0.04, "thres": 0.9}
# param_ks_refine = {"sig": 0.05}
# param_seeds_merge = {"thres_dist": 10, "thres_corr": 0.8, "noise_freq": 0.06}
# param_initialize = {"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.06}
# param_init_merge = {"thres_corr": 0.8}

# # CNMF Parameters#
# param_get_noise = {"noise_range": (0.06, 0.5)}
# param_first_spatial = {
#     "dl_wnd": 10,
#     "sparse_penal": 0.01,
#     "size_thres": (25, None),
# }
# param_first_temporal = {
#     "noise_freq": 0.06,
#     "sparse_penal": 1,
#     "p": 1,
#     "add_lag": 20,
#     "jac_thres": 0.2,
# }
# param_first_merge = {"thres_corr": 0.8}
# param_second_spatial = {
#     "dl_wnd": 10,
#     "sparse_penal": 0.01,
#     "size_thres": (25, None),
# }
# param_second_temporal = {
#     "noise_freq": 0.06,
#     "sparse_penal": 1,
#     "p": 1,
#     "add_lag": 20,
#     "jac_thres": 0.4,
# }

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MINIAN_INTERMEDIATE"] = intpath