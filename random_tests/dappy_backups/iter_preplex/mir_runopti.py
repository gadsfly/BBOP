from neuroposelib import read
from neuroposelib import vis
import numpy as np
import time
from IPython.display import Video
from pathlib import Path
import matplotlib.pyplot as plt
# %matplotlib inline
from neuroposelib import preprocess
from neuroposelib import write
from neuroposelib import features
from neuroposelib import DataStruct as ds
from neuroposelib.embed import Embed
from neuroposelib import analysis
import pandas as pd
from neuroposelib.embed import Watershed

# analysis_key = "tutorial"
# config = read.config("../configs/" + analysis_key + ".yaml")
config = read.config('/home/lq53/mir_repos/dappy_24_nov/byws_version/mir_1.yaml')
# pose, ids = read.pose_h5(config["data_path"] + "demo_mouse.h5")


connectivity = read.connectivity(
    path=config["skeleton_path"], skeleton_name=config["skeleton_name"]
)

# Make out_path
Path(config["out_path"]).mkdir(parents=True, exist_ok=True)

# meta, meta_by_frame = read.meta(config["data_path"] + "demo_meta.csv", id=ids)
pose, ids, meta, meta_by_frame = read.pose_from_meta(
    path=config["meta_path"], connectivity=connectivity, key="Prediction_path", file_type="dannce"
)

Path(config["out_path"]).mkdir(parents=True, exist_ok=True)




# # write.pose_h5(pose_aligned, ids, config["data_path"] + "pose_aligned.h5")
write.pose_h5(pose,ids, config['out_path'] + 'pose_merged_newcol.h5')
# # Read pose_merged.h if already saved
# # pose, ids = read.pose_h5(config["data_path"] + "pose_merged.h5")


vis.pose.arena3D(
    pose,
    connectivity,
    frames=[1000, 500000],
    N_FRAMES=150,
    dpi=100,
    VID_NAME="raw.mp4",
    SAVE_ROOT=config["out_path"],
)

# Provide the mid-spine and the mid-spine -> front-spine indices.
# pose = preprocess.rotate_spine(preprocess.center_spine(pose_aligned, keypt_idx=4), keypt_idx=[4, 3])
pose = preprocess.rotate_spine(preprocess.center_spine(pose, keypt_idx=4), keypt_idx=[4, 3])

vis.pose.arena3D(
    pose,
    connectivity,
    frames=[50000],
    N_FRAMES=150,
    dpi=100,
    VID_NAME="centered.mp4",
    SAVE_ROOT=config["out_path"],
)

# # Getting relative velocities
rel_vel, rel_vel_labels = features.get_velocities(
    pose,
    ids,
    connectivity.joint_names,
    joints=np.delete(np.arange(18), 4),
    widths=[5, 11, 51],
    f_s = 30
)



# Video(url=config["out_path"] + "vis_centered.mp4", width=600, height=600)
# Calculating joint angles
angles, angle_labels = features.get_angles(pose, connectivity.angles)

# Angular Velocities
# angular_vel, angular_vel_labels = features.get_angular_vel(
#         angles,
#         angle_labels,
#         ids,
#         widths=[5, 11, 51],
#     )


# Reshape pose to get egocentric pose features
ego_pose, labels = features.get_ego_pose(pose, connectivity.joint_names)

# Write
write.features_h5(angles, labels, path=config["out_path"] + "angels.h5")
write.features_h5(ego_pose, labels, path=config["out_path"] + "ego_pose.h5")
print("saved")

combined_features = np.hstack((ego_pose, angles, rel_vel))
combined_labels = labels + angle_labels + rel_vel_labels

# write.features_h5(angles, angle_labels, path=config["out_path"] + "angles_calc.h5")

# Read
# features, labels = read.features_h5(path=config["out_path"] + "postural_feats.h5")

t = time.time()

# pc_feats, pc_labels = features.pca(
#     angles, angle_labels, categories=["ang"], n_pcs=5, method="fbpca"
# )
pc_feats, pc_labels = features.pca(
    combined_features,
    combined_labels,
    categories=["ego_euc", "angle", "velocity"],
    n_pcs=10,  #10 just in case, in case 5 will, for instnace, only cover 70% of the data
    method="fbpca",
)

write.features_h5(pc_feats, pc_labels, path=config["out_path"] + "_pca.h5")
print("PCA time: " + str(time.time() - t))
del angles, angle_labels
del ego_pose, labels

# from scipy import signal
# M = 100
# w0 = 5
# s = w0*90/(2*np.pi*25) #30, 90?
# morlet_wavelet = signal.morlet2(M, s, w0)
# plt.plot(morlet_wavelet.imag, label='Imaginary')
# plt.plot(morlet_wavelet.real, label='Real')
# plt.legend()
# plt.show()


wlet_feats, wlet_labels = features.wavelet(
    pc_feats, pc_labels, ids, f_s=30, freq=np.linspace(1, 25, 25), w0=5
)

write.features_h5(wlet_feats, wlet_labels, path=config["out_path"] + "_wavelets.h5")


# PCA on wavelet features
pc_wlet, pc_wlet_labels = features.pca(
    wlet_feats,
    wlet_labels,
    # categories=["wlet_ego_euc"],
    categories=["wlet_ang"],
    n_pcs=5,
    method="fbpca",
)

del wlet_feats, wlet_labels
pc_feats = np.hstack((pc_feats, pc_wlet))
pc_labels += pc_wlet_labels
del pc_wlet, pc_wlet_labels

write.features_h5(
    pc_feats, pc_labels, path="".join([config["out_path"], "_pca_on_wavelets.h5"])
)




data_obj = ds.DataStruct(
    pose=pose,
    id=ids,
    meta=meta,
    meta_by_frame=meta_by_frame,
    connectivity=connectivity,
)

data_obj.features = pc_feats
# When using high framerate data, downsampling may be necessary in order to 
# discover granular structure in embedding
data_obj = data_obj[:: config["downsample"], :]


embedder = Embed(
    embed_method=config["single_embed"]["method"],
    perplexity=config["single_embed"]["perplexity"],
    lr=config["single_embed"]["lr"],
)
data_obj.embed_vals = embedder.embed(data_obj.features, save_self=True)

# Watershed clustering
data_obj.ws = Watershed(
    sigma=config["single_embed"]["sigma"], max_clip=1, log_out=True, pad_factor=0.05
)
data_obj.data["Cluster"] = data_obj.ws.fit_predict(data=data_obj.embed_vals)
print("Writing Data Object to pickle")
data_obj.write_pickle(''.join([config['out_path'],'/']))



freq, combined_keys = analysis.cluster_freq_by_cat(
                        data_obj.data["Cluster"].values, cat=data_obj.id
                    )
freq_df = pd.DataFrame(freq.T, columns=combined_keys)
freq_df.to_csv(''.join([config['out_path'],'/cluster_occupancy.csv']))

vis.pose.sample_grid3D(
    pose-pose.mean(axis=-2, keepdims=True),
    connectivity=connectivity,
    labels=data_obj.data["Cluster"],
    n_samples=9,
    centered=True,
    N_FRAMES=100,
    fps=30,
    dpi=100,
    watershed=data_obj.ws,
    embed_vals=None,
    VID_NAME = "cluster",
    filepath=config["out_path"],
)