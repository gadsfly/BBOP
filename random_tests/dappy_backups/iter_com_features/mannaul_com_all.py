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

config = read.config('/home/lq53/mir_repos/BBOP/random_tests/dappy_backups/iter_com_features/mir_1.yaml')

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

# write.pose_h5(pose,ids, config['out_path'] + 'pose_merged_newcol.h5')


pose = preprocess.rotate_spine(preprocess.center_spine(pose, keypt_idx=4), keypt_idx=[4, 3])


rel_vel, rel_vel_labels = features.get_velocities(
    pose,
    ids,
    connectivity.joint_names,
    joints=np.delete(np.arange(18), 4),
    widths=[5, 11, 51],
    f_s = 30
)

angles, angle_labels = features.get_angles(pose, connectivity.angles)

ego_pose, labels = features.get_ego_pose(pose, connectivity.joint_names)
# # Ensure all feature datasets have the same number of samples

# Angular Velocities
angular_vel, angular_vel_labels = features.get_angular_vel(
        angles,
        angle_labels,
        ids,
        widths=[5, 11, 51],
    )
# write.features_h5(
#     angular_vel, angular_vel_labels, path=config["out_path"] + "angular_velocity.h5"
# )

# Compute and save Euler angles (if applicable)
euler_angles, euler_labels = features.get_euler_angles(pose, connectivity.angles)
# write.features_h5(
#     euler_angles, euler_labels, path=config["out_path"] + "euler_angles.h5"
# )

# Compute and save head angular velocities
head_angular = features.get_head_angular(pose, ids, widths=[5, 10, 50])
# write.features_h5(
#     head_angular, ["head_angular"], path=config["out_path"] + "head_angular.h5"
# )
# assert rel_vel.shape[0] == angles.shape[0] == ego_pose.shape[0]

# # Combine features
# combined_features = np.hstack([rel_vel, angles, ego_pose])

# # Concatenate labels
# combined_labels = rel_vel_labels + angle_labels + labels

assert head_angular.shape[0] == euler_angles.shape[0] == angular_vel.shape[0] == rel_vel.shape[0]

# Combine all features
combined_features = np.hstack([rel_vel, angles, ego_pose, head_angular, euler_angles, angular_vel])
combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=0.0, neginf=0.0)

# Combine all labels
combined_labels = rel_vel_labels + angle_labels + labels + ["head_angular"] + euler_labels + angular_vel_labels



# Perform PCA on the combined features
pca_features, pca_labels = features.pca(
    features=combined_features,
    labels=combined_labels,
    categories=["vel", "ang", "ego_euc", "head_ang", "euler_ang", "avel"], #["vel", "ego_euc", "ang", "avel"],
    n_pcs=8,
    downsample=1,
    method="fbpca",
)

write.features_h5(pca_features, pca_labels, path=config["out_path"] + "_combined_pca.h5")


pc_feats, pc_labels = pca_features, pca_labels

wlet_feats, wlet_labels = features.wavelet(
    pc_feats, pc_labels, ids, f_s=30, freq=np.geomspace(1,25,25), w0=5
)

write.features_h5(wlet_feats, wlet_labels, path=config["out_path"] + "_wavelets.h5")

# PCA on wavelet features
pc_wlet, pc_wlet_labels = features.pca(
    wlet_feats,
    wlet_labels,
    # categories=["wlet_ego_euc"],
    # categories=["wlet_ang"],
    categories=["wlet_vel", "wlet_ang", "wlet_ego_euc", "wlet_head_ang", "wlet_euler_ang", "wlet_avel"], #["vel", "ego_euc", "ang", "avel"],
    n_pcs=8,
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
