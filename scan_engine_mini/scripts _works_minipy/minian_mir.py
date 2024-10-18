# %%capture
# %load_ext autoreload
# %autoreload 2
import warnings
import asyncio

from minian_param_mir import *

import itertools as itt
import os
import sys

# import holoviews as hv
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster
# from holoviews.operation.datashader import datashade, regrid
# from holoviews.util import Dynamic
# from IPython.core.display import display
# import matplotlib.pyplot as plt


sys.path.append(minian_path)
from minian.cnmf import (
    compute_AtC,
    compute_trace,
    get_noise_fft,
    smooth_sig,
    unit_merge,
    update_spatial,
    # update_temporal,
    update_background,
)
from minian.initialization import (
    gmm_refine,
    initA,
    initC,
    intensity_refine,
    ks_refine,
    pnr_refine,
    seeds_init,
    seeds_merge,
)
from minian.motion_correction import apply_transform, estimate_motion
from minian.preprocessing import denoise, remove_background
from minian.utilities import (
    TaskAnnotation,
    get_optimal_chk,
    load_videos,
    open_minian,
    save_minian,
)
from minian.visualization import (
    # CNMFViewer,
    # VArrayViewer,
    # generate_videos,
    # visualize_gmm_fit,
    # visualize_motion,
    # visualize_preprocess,
    # visualize_seeds,
    # visualize_spatial_update,
    # visualize_temporal_update,
    write_video,
)

dpath = os.path.abspath(dpath)
# hv.notebook_extension("bokeh", width=100)
import time
# client.close()
# cluster.close()
# warnings.filterwarnings("ignore", category=RuntimeWarning, module="distributed.client")

# print("===========================Starting Cluster=============================")

def start_cluster_1():
    # cluster = LocalCluster(
    #     n_workers=n_workers,
    #     memory_limit="2GB",
    #     resources={"MEM": 1},
    #     threads_per_worker=2,
    #     dashboard_address=":8789",
    # )
    # time.sleep(1)
    # annt_plugin = TaskAnnotation()
    # cluster.scheduler.add_plugin(annt_plugin)
    # client = Client(cluster)


    cluster = LocalCluster(
        local_directory=dpath,
        n_workers=n_workers,
        memory_limit="25GB",
        resources={"MEM": 6},
        threads_per_worker=1,
    #     memory_spill=0.6,
    #     lifetime='2h',
    #     lifetime_stagger = '10m',
        dashboard_address=":8787",
    )
    time.sleep(1)
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)

    # await client.close()

    return client, cluster


def start_cluster_2():
    cluster = LocalCluster(
        local_directory=dpath,
        n_workers=n_workers,
        memory_limit="25GB",
        resources={"MEM": 6},
        threads_per_worker=2,
    #     memory_spill=0.6,
    #     lifetime='2h',
    #     lifetime_stagger = '10m',
        dashboard_address=":8787",
    )
    time.sleep(1)
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)
    client = Client(cluster)

    # await client.close()

    return client, cluster


# cluster, client = start_cluster()

# print("===========================Cluster Started=============================")

#below is mir's modification, to optimize speed and time, stupid>>>
# cluster = LocalCluster(
#     n_workers=n_workers,
#     memory_limit="25GB",
#     resources={"MEM": 6},
#     threads_per_worker=2,
# #     memory_spill=0.6,
# #     lifetime='2h',
# #     lifetime_stagger = '10m',
#     # dashboard_address=":8787",
# # )
# annt_plugin = TaskAnnotation()
# cluster.scheduler.add_plugin(annt_plugin)
# client = Client(cluster)

# varr = load_videos(dpath, **param_load_videos)
# chk, _ = get_optimal_chk(varr, dtype=float)


# varr = save_minian(
#     varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
#     intpath,
#     overwrite=True,
# )

# varr_ref = varr.sel(subset) #just to lazy to change anything, usually no subset

# varr_min = varr_ref.min("frame").compute()
# varr_ref = varr_ref - varr_min


# varr_ref = denoise(varr_ref, **param_denoise)

# varr_ref = remove_background(varr_ref, **param_background_removal)

# varr_ref = save_minian(varr_ref.rename("varr_ref"), dpath=intpath, overwrite=True)

# motion = estimate_motion(varr_ref.sel(subset_mc), **param_estimate_motion)

# param_save_minian

# motion = save_minian(
#     motion.rename("motion").chunk({"frame": chk["frame"]}), **param_save_minian
# )

# Y = apply_transform(varr_ref, motion, fill=0)

# Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
# Y_hw_chk = save_minian(
#     Y_fm_chk.rename("Y_hw_chk"),
#     intpath,
#     overwrite=True,
#     chunks={"frame": -1, "height": chk["height"], "width": chk["width"]},
# )

# vid_arr = xr.concat([varr_ref, Y_fm_chk], "width").chunk({"width": -1})
# write_video(vid_arr,"minian_mc.mp4", dpath)


# max_proj = save_minian(
#     Y_fm_chk.max("frame").rename("max_proj"), **param_save_minian
# ).compute()


# param_seeds_init

# seeds = seeds_init(Y_fm_chk, **param_seeds_init)

# seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **param_pnr_refine)

# param_ks_refine

# seeds = ks_refine(Y_hw_chk, seeds, **param_ks_refine)

# seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
# seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **param_seeds_merge)


# # below is to deal with killed workers. but probably would not work for the cluster i doubted. it was optimized for better performence specifically to cluster.....
# # client.close()
# # cluster.close()
# # cluster = LocalCluster(
# #     n_workers=n_workers,
# #     memory_limit="25GB",
# #     resources={"MEM": 6},
# #     threads_per_worker=1,
# # #     memory_spill=0.6,
# # #     lifetime='2h',
# # #     lifetime_stagger = '10m',
# #     dashboard_address=":8787",
# # )
# # annt_plugin = TaskAnnotation()
# # cluster.scheduler.add_plugin(annt_plugin)
# # client = Client(cluster)

# A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **param_initialize)
# A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)

# C_init = initC(Y_fm_chk, A_init)
# C_init = save_minian(
#     C_init.rename("C_init"), intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1}
# )

# param_init_merge

# A, C = unit_merge(A_init, C_init, **param_init_merge)
# A = save_minian(A.rename("A"), intpath, overwrite=True)
# C = save_minian(C.rename("C"), intpath, overwrite=True)
# C_chk = save_minian(
#     C.rename("C_chk"),
#     intpath,
#     overwrite=True,
#     chunks={"unit_id": -1, "frame": chk["frame"]},
# )

# b, f = update_background(Y_fm_chk, A, C_chk)
# f = save_minian(f.rename("f"), intpath, overwrite=True)
# b = save_minian(b.rename("b"), intpath, overwrite=True)

# # client.close()
# # cluster.close()
# # cluster = LocalCluster(
# #     n_workers=n_workers,
# #     memory_limit="25GB",
# #     resources={"MEM": 6},
# #     threads_per_worker=1,
# # #     memory_spill=0.6,
# # #     lifetime='2h',
# # #     lifetime_stagger = '10m',
# #     dashboard_address=":8787",
# # )
# # annt_plugin = TaskAnnotation()
# # cluster.scheduler.add_plugin(annt_plugin)
# # client = Client(cluster)

# sn_spatial = get_noise_fft(Y_hw_chk, **param_get_noise)
# sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)

# # client.close()
# # cluster.close()
# # cluster = LocalCluster(
# #     n_workers=n_workers,
# #     memory_limit="25GB",
# #     resources={"MEM": 6},
# #     threads_per_worker=2,
# # #     memory_spill=0.6,
# # #     lifetime='2h',
# # #     lifetime_stagger = '10m',
# #     dashboard_address=":8787",
# # )
# # annt_plugin = TaskAnnotation()
# # cluster.scheduler.add_plugin(annt_plugin)
# # client = Client(cluster)


# A_new, mask, norm_fac = update_spatial(
#     Y_hw_chk, A, C, sn_spatial, **param_first_spatial
# )
# C_new = save_minian(
#     (C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True
# )
# C_chk_new = save_minian(
#     (C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True
# )

# b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)

# A = save_minian(
#     A_new.rename("A"),
#     intpath,
#     overwrite=True,
#     chunks={"unit_id": 1, "height": -1, "width": -1},
# )
# b = save_minian(b_new.rename("b"), intpath, overwrite=True)
# f = save_minian(
#     f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True
# )
# C = save_minian(C_new.rename("C"), intpath, overwrite=True)
# C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)

# A = save_minian(
#     A_new.rename("A"),  
#     minian_ds_path,
#     overwrite=True,
#     chunks={"unit_id": 1, "height": -1, "width": -1},
# )
# b = save_minian(b_new.rename("b"), minian_ds_path, overwrite=True)
# f = save_minian(
#     f_new.chunk({"frame": chk["frame"]}).rename("f"), minian_ds_path, overwrite=True
# )
# C = save_minian(C_new.rename("C"), minian_ds_path, overwrite=True)
# C_chk = save_minian(C_chk_new.rename("C_chk"), minian_ds_path, overwrite=True)

# minian_ds = open_minian(minian_ds_path)
# minian_ds.to_netcdf("minian_dataset_241010_denoised.nc")

# async def close_cluster_and_client(client, cluster):
#     await client.close()  # Await the client close method
#     await cluster.close()  # Await the cluster close method
#     print("cluster closed")

if __name__ == "__main__":
    # freeze_support()
    # print("dpath = ", dpath)
    cluster, client = start_cluster_2()

    

    #Do Stuff
    varr = load_videos(dpath, **param_load_videos)
    chk, _ = get_optimal_chk(varr, dtype=float)


    varr = save_minian(
        varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
        intpath,
        overwrite=True,
    )

    varr_ref = varr.sel(subset) #just to lazy to change anything, usually no subset

    varr_min = varr_ref.min("frame").compute()
    varr_ref = varr_ref - varr_min


    varr_ref = denoise(varr_ref, **param_denoise)

    varr_ref = remove_background(varr_ref, **param_background_removal)

    varr_ref = save_minian(varr_ref.rename("varr_ref"), dpath=intpath, overwrite=True)

    motion = estimate_motion(varr_ref.sel(subset_mc), **param_estimate_motion)

    param_save_minian

    motion = save_minian(
        motion.rename("motion").chunk({"frame": chk["frame"]}), **param_save_minian
    )

    Y = apply_transform(varr_ref, motion, fill=0)

    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    Y_hw_chk = save_minian(
        Y_fm_chk.rename("Y_hw_chk"),
        intpath,
        overwrite=True,
        chunks={"frame": -1, "height": chk["height"], "width": chk["width"]},
    )

    vid_arr = xr.concat([varr_ref, Y_fm_chk], "width").chunk({"width": -1})
    write_video(vid_arr,"minian_mc.mp4", dpath)


    max_proj = save_minian(
        Y_fm_chk.max("frame").rename("max_proj"), **param_save_minian
    ).compute()


    param_seeds_init

    seeds = seeds_init(Y_fm_chk, **param_seeds_init)

    seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **param_pnr_refine)

    param_ks_refine

    seeds = ks_refine(Y_hw_chk, seeds, **param_ks_refine)

    seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
    seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **param_seeds_merge)


    # below is to deal with killed workers. but probably would not work for the cluster i doubted. it was optimized for better performence specifically to cluster.....
    client.close()
    cluster.close()
    cluster, client = start_cluster_1()
    # cluster = LocalCluster(
    #     n_workers=n_workers,
    #     memory_limit="25GB",
    #     resources={"MEM": 6},
    #     threads_per_worker=1,
    # #     memory_spill=0.6,
    # #     lifetime='2h',
    # #     lifetime_stagger = '10m',
    #     dashboard_address=":8787",
    # )
    # annt_plugin = TaskAnnotation()
    # cluster.scheduler.add_plugin(annt_plugin)
    # client = Client(cluster)

    A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **param_initialize)
    A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)

    C_init = initC(Y_fm_chk, A_init)
    C_init = save_minian(
        C_init.rename("C_init"), intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1}
    )

    param_init_merge

    A, C = unit_merge(A_init, C_init, **param_init_merge)
    A = save_minian(A.rename("A"), intpath, overwrite=True)
    C = save_minian(C.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(
        C.rename("C_chk"),
        intpath,
        overwrite=True,
        chunks={"unit_id": -1, "frame": chk["frame"]},
    )

    b, f = update_background(Y_fm_chk, A, C_chk)
    f = save_minian(f.rename("f"), intpath, overwrite=True)
    b = save_minian(b.rename("b"), intpath, overwrite=True)

    client.close()
    cluster.close()
    cluster, client = start_cluster_1()

    # client.close()
    # cluster.close()
    # cluster = LocalCluster(
    #     n_workers=n_workers,
    #     memory_limit="25GB",
    #     resources={"MEM": 6},
    #     threads_per_worker=1,
    # #     memory_spill=0.6,
    # #     lifetime='2h',
    # #     lifetime_stagger = '10m',
    #     dashboard_address=":8787",
    # )
    # annt_plugin = TaskAnnotation()
    # cluster.scheduler.add_plugin(annt_plugin)
    # client = Client(cluster)

    sn_spatial = get_noise_fft(Y_hw_chk, **param_get_noise)
    sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)

    client.close()
    cluster.close()
    # time.sleep(1)
    cluster, client = start_cluster_2()
    # client.close()
    # cluster.close()
    # cluster = LocalCluster(
    #     n_workers=n_workers,
    #     memory_limit="25GB",
    #     resources={"MEM": 6},
    #     threads_per_worker=2,
    # #     memory_spill=0.6,
    # #     lifetime='2h',
    # #     lifetime_stagger = '10m',
    #     dashboard_address=":8787",
    # )
    # annt_plugin = TaskAnnotation()
    # cluster.scheduler.add_plugin(annt_plugin)
    # client = Client(cluster)


    A_new, mask, norm_fac = update_spatial(
        Y_hw_chk, A, C, sn_spatial, **param_first_spatial
    )
    C_new = save_minian(
        (C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True
    )
    C_chk_new = save_minian(
        (C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True
    )

    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)

    A = save_minian(
        A_new.rename("A"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(
        f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True
    )
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)

    A = save_minian(
        A_new.rename("A"),  
        minian_ds_path,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), minian_ds_path, overwrite=True)
    f = save_minian(
        f_new.chunk({"frame": chk["frame"]}).rename("f"), minian_ds_path, overwrite=True
    )
    C = save_minian(C_new.rename("C"), minian_ds_path, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), minian_ds_path, overwrite=True)

    minian_ds = open_minian(minian_ds_path)
    minian_ds.to_netcdf(nc_file_name)
    print("data saved to .nc")
    # await client.close()
    time.sleep(1)
    

    # client.close()
    # cluster.close()
    print("cluster closed")