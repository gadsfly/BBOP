import numpy as np
import pandas as pd


def get_body_rotations_df_new(df, spineF_kp=4, tailB_kp=6, spineM_kp=5):
    print('Processing to get body related rotation matrix ......')
    
    nf = df.shape[0]
    r_roots = np.zeros((nf, 3, 3))  
    r_root_inv = np.zeros((nf, 3, 3))  
    r_root_inv_oriented = np.zeros((nf, 3, 3))  
    dir_backs = np.zeros((nf, 3))  

    r_roots[:] = np.nan
    r_root_inv[:] = np.nan
    r_root_inv_oriented[:] = np.nan
    dir_backs[:] = np.nan

    # Extract x, y, and z coordinates for relevant keypoints
    spineF_x, spineF_y, spineF_z = df[f"kp{spineF_kp}_x"], df[f"kp{spineF_kp}_y"], df[f"kp{spineF_kp}_z"]
    tailB_x, tailB_y, tailB_z = df[f"kp{tailB_kp}_x"], df[f"kp{tailB_kp}_y"], df[f"kp{tailB_kp}_z"]
    spineM_x, spineM_y, spineM_z = df[f"kp{spineM_kp}_x"], df[f"kp{spineM_kp}_y"], df[f"kp{spineM_kp}_z"]

    # Precompute valid mask
    valid_mask = ~np.isnan(spineF_x) & ~np.isnan(tailB_x) 

    # 1. NaN handling when computing xdir (body direction from tail to neck)
    xdir = np.column_stack((spineF_x - tailB_x, spineF_y - tailB_y, np.zeros_like(spineF_z)))
    xdir[~valid_mask] = np.nan  # Explicitly set NaN where mask is invalid
    ll = np.linalg.norm(xdir, axis=1, keepdims=True)
    valid_mask &= (ll[:, 0] >= 0.001)
    xdir[~valid_mask] = np.nan  # Propagate NaNs after length check
    xdir[valid_mask] /= ll[valid_mask]

    # Compute y-direction using right-hand rule
    zdir = np.array([0, 0, 1])
    ydir = np.cross(zdir, xdir)
    ydir[~valid_mask] = np.nan  # Ensure ydir is NaN where xdir is NaN

    # Construct the transformation matrix for body-to-global mapping
    r_roots = np.stack([xdir, ydir, np.tile(zdir, (nf, 1))], axis=1)
    r_roots[~valid_mask] = np.nan  # Set entire matrix to NaN for invalid frames
    r_roots = np.transpose(r_roots, (0, 2, 1))

    # 2. NaN handling for non-projected body rotation
    xdir_no_proj = np.column_stack((spineF_x - tailB_x, spineF_y - tailB_y, spineF_z - tailB_z))
    xdir_no_proj[~valid_mask] = np.nan  # Set NaN where invalid
    ll_no_proj = np.linalg.norm(xdir_no_proj, axis=1, keepdims=True)
    valid_mask_no_proj = (ll_no_proj[:, 0] >= 0.001)
    xdir_no_proj[~valid_mask_no_proj] = np.nan  # Propagate NaN if length is too short
    xdir_no_proj[valid_mask_no_proj] /= ll_no_proj[valid_mask_no_proj]

    ydir_no_proj = np.column_stack((-xdir_no_proj[:, 1], xdir_no_proj[:, 0], np.zeros(nf)))
    ydir_no_proj[~valid_mask_no_proj] = np.nan  # Ensure NaN in ydir
    ydir_no_proj /= np.linalg.norm(ydir_no_proj, axis=1, keepdims=True)

    zdir_no_proj = np.cross(xdir_no_proj, ydir_no_proj)
    zdir_no_proj[~valid_mask_no_proj] = np.nan  # Ensure zdir gets NaNs
    zdir_no_proj /= np.linalg.norm(zdir_no_proj, axis=1, keepdims=True)
    r_root_inv = np.stack([xdir_no_proj, ydir_no_proj, zdir_no_proj], axis=1)
    r_root_inv[~valid_mask_no_proj] = np.nan

    # 3. NaN handling for planar body direction
    xdir_planar = np.column_stack((spineF_x - tailB_x, spineF_y - tailB_y, np.zeros(nf)))
    xdir_planar[~valid_mask] = np.nan
    ll_planar = np.linalg.norm(xdir_planar, axis=1, keepdims=True)
    valid_mask_planar = (ll_planar[:, 0] >= 0.001)
    xdir_planar[~valid_mask_planar] = np.nan  # Propagate NaNs where length is too short
    xdir_planar[valid_mask_planar] /= ll_planar[valid_mask_planar]

    ydir_planar = np.column_stack((-xdir_planar[:, 1], xdir_planar[:, 0], np.zeros(nf)))
    ydir_planar[~valid_mask_planar] = np.nan
    ydir_planar /= np.linalg.norm(ydir_planar, axis=1, keepdims=True)

    zdir_fixed = np.tile([0, 0, 1], (nf, 1))
    r_root_inv_oriented = np.stack([xdir_planar, ydir_planar, zdir_fixed], axis=1)
    r_root_inv_oriented[~valid_mask_planar] = np.nan

    # 4. NaN handling for direction to mid-spine
    valid_mid_spine = ~np.isnan(spineM_x)
    dir_to_butt = np.column_stack((spineF_x - spineM_x, spineF_y - spineM_y, spineF_z - spineM_z))
    dir_to_butt[~valid_mid_spine] = np.nan
    ll_butt = np.linalg.norm(dir_to_butt, axis=1, keepdims=True)
    valid_butt_mask = (ll_butt[:, 0] >= 0.001)
    dir_to_butt[~valid_butt_mask] = np.nan  # Propagate NaNs if length is too short
    dir_to_butt[valid_butt_mask] /= ll_butt[valid_butt_mask]

    dir_to_butt = np.einsum('ijk,ik->ij', r_root_inv, dir_to_butt)
    dir_backs = dir_to_butt

    return r_roots, r_root_inv_oriented, r_root_inv, dir_backs

