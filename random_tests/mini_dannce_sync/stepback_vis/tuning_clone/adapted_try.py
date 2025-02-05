"""
Performs rotations on tracking data.
@author: Benjamin Adric Dunn
@modified: JingyiGF
@modified: Mir Qi (Gadsfly), 2025 Jan
"""
import numpy as np
import pandas as pd
import math
from scipy.optimize import minimize

###############################################################################
# 1) Basic Rotation Matrices
###############################################################################

def rotation_x(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    return np.array([[1, 0,   0],
                     [0, ct, -st],
                     [0, st,  ct]])

def rotation_y(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    return np.array([[ ct, 0, st],
                     [  0, 1,  0],
                     [-st, 0, ct]])

def rotation_z(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    return np.array([[ ct, -st, 0],
                     [ st,  ct, 0],
                     [  0,   0, 1]])

###############################################################################
# 2) Utility for building rotation from Euler angles (optional)
###############################################################################

def eul2rot(ang_vec, order=5):
    """
    Build rotation matrix from euler angles (x,y,z).
    'order' is a placeholder that you can adapt to your convention.
    Example usage is optional if you rely on rot2euler, etc. 
    """
    if np.any(np.isnan(ang_vec)):
        rm = np.zeros((3, 3))
        rm[:] = np.nan
        return rm

    x, y, z = ang_vec
    rz = rotation_z(z)
    ry = rotation_y(y)
    rx = rotation_x(x)

    # You can adapt the logic to your specific Euler angle order:
    # For instance, order == 5 might be 'xyz' as an example
    if order == 5:  # e.g. xyz
        rm = rz @ (ry @ rx)
    else:
        raise ValueError("Unhandled rotation order. Extend if needed.")
    return rm

###############################################################################
# 3) Converting rotation matrices to angles (various helpers)
###############################################################################

def check_rot_angs(angs):
    """Ensure angles lie in [-pi, pi] range, etc."""
    for i in range(3):
        if angs[i] >  math.pi:
            angs[i] -= 2. * math.pi
        if angs[i] < -math.pi:
            angs[i] += 2. * math.pi
    count_big = np.sum(np.abs(angs) > (math.pi / 2.0))
    return angs, count_big

def rot2expmap(rot_mat):
    """Convert a 3x3 rotation matrix to exponential map representation."""
    d = rot_mat - rot_mat.T
    r = np.array([-d[1, 2], d[0, 2], -d[0, 1]])
    sintheta = np.linalg.norm(r) / 2.0
    r0 = r / (np.linalg.norm(r) + np.finfo(float).eps)
    costheta = (np.trace(rot_mat) - 1.0) / 2.0
    theta = math.atan2(sintheta, costheta)
    # half-angle stuff
    theta = 2.0 * math.atan2(np.sin(theta / 2.0), np.cos(theta / 2.0))
    theta = np.fmod(theta + 2 * math.pi, 2 * math.pi)
    if theta > math.pi:
        theta = 2 * math.pi - theta
        r0 = -r0
    return r0 * theta

def rot2euler_xzy(rot_mat):
    """
    Example alternative for XZY euler extraction.
    """
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = rot_mat.flatten()
    temp = math.sqrt(r22**2 + r32**2)
    if temp > 1e-7:
        x = np.arctan2(r32, r22)
        z = np.arctan2(-r12, temp)
        y = np.arctan2(r13, r11)
    else:
        # degenerate case
        x = 0.0
        z = np.arctan2(-r12, temp)
        y = np.arctan2(r13, r11)
    return np.array([x, y, z])

def rot2euler(rot_mat, use_solution_with_least_tilt=False):
    """
    Default XYZ euler extraction, matching original logic for 'rot2euler'.
    Adjust as needed for your use case.
    """
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = rot_mat.flatten()
    temp = math.sqrt(r33**2 + r23**2)
    angs = np.zeros(3)
    if temp > 1e-4:
        angs[0] = np.arctan2(-r23, r33)   # roll
        angs[1] = np.arctan2(r13, temp)  # pitch
        angs[2] = np.arctan2(-r12, r11)  # yaw
    else:
        angs[0] = 0.0
        angs[1] = np.arctan2(r13, temp)
        angs[2] = np.arctan2(r21, r22)

    angs, _ = check_rot_angs(angs)
    if use_solution_with_least_tilt:
        # Example alternative solution
        angs_o = np.array([angs[0] + math.pi,
                           math.pi - angs[1],
                           angs[2] + math.pi])
        angs_o, _ = check_rot_angs(angs_o)
        # pick whichever has smaller pitch, for instance
        if abs(angs_o[1]) < abs(angs[1]):
            angs = angs_o
    return angs

def rot2angles(rot_mat, use_expmap=False, use_xzy_order=False, use_solution_with_least_tilt=False):
    """Wrapper function that returns either exponential map or Euler angles."""
    if np.isnan(rot_mat[0, 0]):
        return np.array([np.nan, np.nan, np.nan])

    if use_expmap:
        return rot2expmap(rot_mat)
    if use_xzy_order:
        return rot2euler_xzy(rot_mat)
    return rot2euler(rot_mat, use_solution_with_least_tilt)

###############################################################################
# 4) Example: Revising get_body_rotations to use DataFrame
###############################################################################

def get_body_rotations_df(
    df,
    spineF_kp=4,    # front spine keypoint index
    tailB_kp=6,     # tail base keypoint index
    spineM_kp=5     # middle spine keypoint index
):
    """
    Example revised function that calculates body rotation matrices from
    columns in a DataFrame (df).

    Returns:
      - r_roots                : top-view rotation matrix (body->global)
      - r_root_inv_oriented    : planar body rotation, ignoring Z
      - r_root_inv             : full 3D body rotation
      - dir_backs              : direction from spineM to spineF in body coords
    """
    print('Processing to get body-related rotation matrices from DataFrame ...')

    nf = df.shape[0]
    # Prepare output arrays
    r_roots             = np.full((nf, 3, 3), np.nan)
    r_root_inv          = np.full((nf, 3, 3), np.nan)
    r_root_inv_oriented = np.full((nf, 3, 3), np.nan)
    dir_backs           = np.full((nf, 3), np.nan)

    # Extract x, y, z for relevant keypoints
    # e.g. "kp4_x", "kp4_y", "kp4_z", etc.
    sf_x, sf_y, sf_z = df[f"kp{spineF_kp}_x"], df[f"kp{spineF_kp}_y"], df[f"kp{spineF_kp}_z"]
    tb_x, tb_y, tb_z = df[f"kp{tailB_kp}_x"],  df[f"kp{tailB_kp}_y"],  df[f"kp{tailB_kp}_z"]
    sm_x, sm_y, sm_z = df[f"kp{spineM_kp}_x"], df[f"kp{spineM_kp}_y"], df[f"kp{spineM_kp}_z"]

    #---------------------------------------------------------------------------
    # 1) r_roots (top-view rotation)
    #    xdir = (spineFront - tailBase), with z=0
    #---------------------------------------------------------------------------
    valid_mask = ~np.isnan(sf_x) & ~np.isnan(tb_x)  # minimal check
    xdir_top = np.column_stack((sf_x - tb_x, sf_y - tb_y, np.zeros(nf)))
    xdir_len = np.linalg.norm(xdir_top, axis=1, keepdims=True)
    # update valid_mask
    valid_mask = valid_mask & (xdir_len[:, 0] > 1e-3)
    # normalize
    xdir_top[valid_mask] /= xdir_len[valid_mask]

    zdir_top = np.array([0, 0, 1])
    ydir_top = np.cross(zdir_top, xdir_top)
    # stack into a 3x3
    # shape (nf, 3, 3) = [ (xdir, ydir, zdir) for each frame ]
    top_mats = np.stack([xdir_top, ydir_top, np.tile(zdir_top, (nf,1))], axis=1)
    top_mats[~valid_mask] = np.nan

    # We want r_roots to be (body->global), so we might transpose:
    # The original code had: r_roots[t,:,:] = np.transpose(shitinv)
    r_roots = np.transpose(top_mats, (0, 2, 1))
    r_roots[~valid_mask] = np.nan

    #---------------------------------------------------------------------------
    # 2) r_root_inv (full 3D body rotation)
    #    xdir = (spineFront - tailBase) in 3D
    #---------------------------------------------------------------------------
    xdir_3d = np.column_stack((sf_x - tb_x, sf_y - tb_y, sf_z - tb_z))
    xdir_len3d = np.linalg.norm(xdir_3d, axis=1, keepdims=True)
    valid_mask_3d = (xdir_len3d[:,0] > 1e-3)
    xdir_3d[~valid_mask_3d] = np.nan
    xdir_3d[valid_mask_3d] /= xdir_len3d[valid_mask_3d]

    # Build ydir in XY-plane
    ydir_3d = np.column_stack((-xdir_3d[:,1], xdir_3d[:,0], np.zeros(nf)))
    # normalize
    ydir_len = np.linalg.norm(ydir_3d, axis=1, keepdims=True)
    valid_mask_3d = valid_mask_3d & (ydir_len[:,0] > 1e-3)
    ydir_3d[~valid_mask_3d] = np.nan
    ydir_3d[valid_mask_3d] /= ydir_len[valid_mask_3d]

    zdir_3d = np.cross(xdir_3d, ydir_3d)
    zdir_len = np.linalg.norm(zdir_3d, axis=1, keepdims=True)
    valid_mask_3d = valid_mask_3d & (zdir_len[:,0] > 1e-3)
    zdir_3d[valid_mask_3d] /= zdir_len[valid_mask_3d]
    zdir_3d[~valid_mask_3d] = np.nan

    full_mats = np.stack([xdir_3d, ydir_3d, zdir_3d], axis=1)
    full_mats[~valid_mask_3d] = np.nan
    r_root_inv = full_mats

    #---------------------------------------------------------------------------
    # 3) r_root_inv_oriented (planar body rotation, ignoring Z)
    #---------------------------------------------------------------------------
    xdir_planar = np.column_stack((sf_x - tb_x, sf_y - tb_y, np.zeros(nf)))
    xdir_len_p = np.linalg.norm(xdir_planar, axis=1, keepdims=True)
    valid_mask_p = (xdir_len_p[:,0] > 1e-3)
    xdir_planar[~valid_mask_p] = np.nan
    xdir_planar[valid_mask_p] /= xdir_len_p[valid_mask_p]

    ydir_planar = np.column_stack((-xdir_planar[:,1], xdir_planar[:,0], np.zeros(nf)))
    ydir_len_p = np.linalg.norm(ydir_planar, axis=1, keepdims=True)
    valid_mask_p = valid_mask_p & (ydir_len_p[:,0] > 1e-3)
    ydir_planar[~valid_mask_p] = np.nan
    ydir_planar[valid_mask_p] /= ydir_len_p[valid_mask_p]

    zdir_fixed = np.tile([0,0,1], (nf,1))
    oriented_mats = np.stack([xdir_planar, ydir_planar, zdir_fixed], axis=1)
    oriented_mats[~valid_mask_p] = np.nan
    r_root_inv_oriented = oriented_mats

    #---------------------------------------------------------------------------
    # 4) dir_backs: direction from spineF to spineM in the local "ego3" coords
    #---------------------------------------------------------------------------
    # if spineM is valid
    valid_mid_spine = ~np.isnan(sm_x) & valid_mask_3d
    dir_to_butt = np.column_stack((sf_x - sm_x, sf_y - sm_y, sf_z - sm_z))
    dist_butt = np.linalg.norm(dir_to_butt, axis=1, keepdims=True)
    good_butt = (dist_butt[:,0] > 1e-3) & valid_mid_spine
    dir_to_butt[~good_butt] = np.nan
    dir_to_butt[good_butt] /= dist_butt[good_butt]

    # multiply each row by that row's 3x3 matrix: (nf x 3 x 3) dot (nf x 3)
    # easiest approach: np.einsum
    dir_backs = np.einsum('ijk,ik->ij', r_root_inv, dir_to_butt)

    return r_roots, r_root_inv_oriented, r_root_inv, dir_backs

###############################################################################
# 5) Example: head rotation from DataFrame (optional)
###############################################################################

def get_head_rotations_df(
    df,
    headF_kp=0,  # e.g. if keypoint 0 is the 'headFront'
    headX_kp=1,  # optional: if you store 'head_x' or something differently
    headZ_kp=2,
    r_root_inv=None,
    r_root_inv_oriented=None
):
    """
    Example function to compute global head rotation and head rotation in body coords
    from columns in DataFrame. 
    (You can adapt the keypoint logic to your exact usage.)
    """
    print("Processing head rotations from DataFrame...")

    nf = df.shape[0]
    global_head_rm = np.full((nf, 3, 3), np.nan)
    r_heads        = np.full((nf, 3, 3), np.nan)
    body_turned    = np.full((nf, 3, 3), np.nan)
    head_ups       = np.full(nf, np.nan)

    # For demonstration, let's say you have columns:
    hx_x, hx_y, hx_z = df[f"kp{headX_kp}_x"], df[f"kp{headX_kp}_y"], df[f"kp{headX_kp}_z"]
    hz_x, hz_y, hz_z = df[f"kp{headZ_kp}_x"], df[f"kp{headZ_kp}_y"], df[f"kp{headZ_kp}_z"]

    # We'll treat hx = unit vector, hz = unit vector, etc.
    for t in range(nf):
        if np.isnan(hx_x[t]):
            continue
        # build hx, hz
        hx = np.array([hx_x[t], hx_y[t], hx_z[t]], dtype=float)
        hz = np.array([hz_x[t], hz_y[t], hz_z[t]], dtype=float)

        # normalize
        if np.linalg.norm(hx) < 1e-7 or np.linalg.norm(hz) < 1e-7:
            continue
        hx /= np.linalg.norm(hx)
        hz /= np.linalg.norm(hz)

        hy = np.cross(hz, hx)
        # global_head_rm => a matrix that transforms from HEAD coords to GLOBAL
        # (similar to old code: [hx; hy; hz])
        mat_inv = np.vstack([hx, hy, hz])
        global_head_rm[t,:,:] = mat_inv

        # "head_ups" just an example if needed:
        head_ups[t] = np.dot(hx, [0,0,1])

        # If you want the local rotation in "ego3" or "ego2":
        if r_root_inv is not None:
            # r_root_inv[t,:,:] is shape (3,3)
            rhx = r_root_inv[t,:,:] @ hx
            rhy = r_root_inv[t,:,:] @ hy
            rhz = r_root_inv[t,:,:] @ hz
            r_heads[t,:,:] = np.vstack([rhx, rhy, rhz])

        if r_root_inv_oriented is not None:
            rhx2 = r_root_inv_oriented[t,:,:] @ hx
            rhy2 = r_root_inv_oriented[t,:,:] @ hy
            rhz2 = r_root_inv_oriented[t,:,:] @ hz
            body_turned[t,:,:] = np.vstack([rhx2, rhy2, rhz2])

    return global_head_rm, r_heads, body_turned, head_ups

###############################################################################
# 6) Angle calculation helpers
###############################################################################

def get_back_angles(p):
    """
    Compute two angles (Ry, Rz) s.t. Rz@Ry@ [1,0,0] = p.
    p is assumed normalized. 
    """
    p = p / np.linalg.norm(p)
    ry = -np.arctan2(p[2], np.sqrt(p[0]**2 + p[1]**2))
    rz = np.arctan2(p[1], p[0])
    return ry, rz

def get_angles(
    r_roots,
    r_heads,
    body_turned_heads,
    global_head_rm,
    dir_backs,
    head_angle_thresh=None,
    use_expmap=False,
    use_xzy_order=False,
    use_solution_with_least_tilt=False
):
    """
    Example wrapper to compute Euler angles from the above rotation matrices.
    """
    print("Computing Euler angles from the rotation matrices...")

    nf = r_heads.shape[0]

    # Prepare output
    allo_head_ang = np.full((nf, 3), np.nan)
    root_ang      = np.full((nf, 3), np.nan)
    ego3_head_ang = np.full((nf, 3), np.nan)
    ego2_head_ang = np.full((nf, 3), np.nan)
    back_ang      = np.full((nf, 2), np.nan)

    for t in range(nf):
        # Body root angles
        root_ang[t,:]      = rot2angles(r_roots[t,:,:], use_expmap, use_xzy_order, use_solution_with_least_tilt)
        # Ego3 (head in body coords)
        ego3_head_ang[t,:] = rot2angles(r_heads[t,:,:], use_expmap, use_xzy_order, use_solution_with_least_tilt)
        # Ego2 (head with planar orientation)
        ego2_head_ang[t,:] = rot2angles(body_turned_heads[t,:,:], use_expmap, use_xzy_order, use_solution_with_least_tilt)
        # Allo head angles (global)
        allo_head_ang[t,:] = rot2angles(global_head_rm[t,:,:], use_expmap, use_xzy_order, use_solution_with_least_tilt)

        # Back angles in 2D if needed
        if not np.any(np.isnan(dir_backs[t,:])):
            ry, rz = get_back_angles(dir_backs[t,:])
            back_ang[t,:] = [ry, rz]

    # If you want to filter angles by some threshold:
    if head_angle_thresh is not None:
        # e.g. head_angle_thresh = (ego2_up, ego2_down, ego3_up, ego3_down)
        pass  # implement your filtering as you see fit

    # Example: invert sign of certain angle dimension if needed
    # (like the original code did: es_head_ang[:,0] = -es_head_ang[:,0], etc.)
    # ...
    return allo_head_ang, root_ang, ego3_head_ang, ego2_head_ang, back_ang

###############################################################################
# 7) Optional: Minimizing or "opt rotate" logic
###############################################################################

def optrotate(avec):
    """
    Example of re-implementing the 'optrotate' step. 
    Takes an array of vectors 'avec' (shape: nFrames x 3) 
    and attempts to find a single global rotation that 
    lines them up with the +X axis.
    """
    print('processing to get opt rotation ...')
    nframe = len(avec)

    def get_rot(tryang):
        # If out-of-bounds, just skip
        if abs(tryang[0]) > math.pi*0.5 or abs(tryang[1]) > math.pi*0.5:
            return None, None

        rmat_x = rotation_x(tryang[0])
        rmat_y = rotation_y(tryang[1])
        rmat_z = rotation_z(tryang[2])
        rot_mat = rmat_x @ (rmat_y @ rmat_z)

        new_vec = np.dot(avec, rot_mat.T)  # shape (nframe, 3)
        return new_vec, rot_mat

    def distance_to_xaxis(params):
        # e.g. we only vary around y, z or so
        # Here we interpret params: [tiltY, tiltZ]
        # build big rotate
        chk, rot_mat = get_rot([0.0, params[0], params[1]])
        if rot_mat is None or chk is None:
            return 1e24
        # average direction
        mean_dir = np.nanmean(chk, axis=0)
        mean_dir /= (np.linalg.norm(mean_dir) + 1e-9)
        # angle to +X
        angle = np.arctan2(np.linalg.norm(np.cross(mean_dir, [1,0,0])), np.dot(mean_dir, [1,0,0]))
        return abs(angle)

    init_guess = np.array([0.0, 0.0])
    res = minimize(distance_to_xaxis, init_guess, method='nelder-mead', options={'xtol':1e-6, 'disp':True})
    best_ang = res.x
    print("opt rotation angles (tiltY, tiltZ) =", best_ang)

    rotated, rot_mat = get_rot([0., best_ang[0], best_ang[1]])
    return rotated, rot_mat

###############################################################################
# 8) Example: Derivatives
###############################################################################

def calc_der(values, frame_rate, bins_1st=1, bins_2nd=1, is_angle=False, der_2nd=False):
    """
    Central difference derivative. 'values' is shape (nFrames, nCols) or (nFrames,).
    'frame_rate' can be scalar or an array (if multi-session). 
    We keep it simple here; adapt to your needs.
    """
    values = np.asarray(values)
    if values.ndim == 1:
        values = values.reshape(-1,1)

    nframe, ncol = values.shape
    val_1st = np.full((nframe, ncol), np.nan)
    val_2nd = np.full((nframe, ncol), np.nan)

    for c in range(ncol):
        for t in range(nframe):
            ts = t - bins_1st
            te = t + bins_1st
            if (ts < 0 or te >= nframe or 
                np.isnan(values[ts,c]) or np.isnan(values[te,c])):
                continue
            diff_1 = values[te,c] - values[ts,c]
            if is_angle:
                # Wrap angle differences
                if diff_1 > 180:
                    diff_1 -= 360
                elif diff_1 < -180:
                    diff_1 += 360

            # dt in time
            dt = 2.*bins_1st / frame_rate if np.isscalar(frame_rate) else 2.*bins_1st / frame_rate[t]
            val_1st[t,c] = diff_1 / dt

        # second derivative
        if der_2nd:
            for t in range(nframe):
                ts = t - bins_2nd
                te = t + bins_2nd
                if (ts < 0 or te >= nframe or 
                    np.isnan(val_1st[ts,c]) or np.isnan(val_1st[te,c])):
                    continue
                diff_2 = val_1st[te,c] - val_1st[ts,c]
                dt2 = 2.*bins_2nd / frame_rate if np.isscalar(frame_rate) else 2.*bins_2nd / frame_rate[t]
                val_2nd[t,c] = diff_2 / dt2

    if der_2nd:
        return val_1st, val_2nd
    else:
        return val_1st, None

###############################################################################
# 9) Example: Self-motion from data in a DataFrame
###############################################################################

def get_selfmotion_df(
    df,
    kp_for_x=4,  # e.g. use spineFront x,y
    body_dir_deg_col='body_dir_deg', 
    frame_rate=30., 
    speed_def='jump',
    window_ms=150
):
    """
    Another small function for self-motion. 
    - df: input DataFrame
    - kp_for_x: which keypoint to use for X/Y
    - body_dir_deg_col: which column in df is the body direction in deg
    - speed_def: 'jump' or 'cum'
    - window_ms: how large a time offset in ms

    Returns dx, dy, speeds as arrays of shape (nFrame,)
    """
    n = df.shape[0]
    dx = np.full(n, np.nan)
    dy = np.full(n, np.nan)
    speeds = np.full(n, np.nan)

    # e.g. columns named "kp4_x", "kp4_y"
    X = df[f"kp{kp_for_x}_x"].values
    Y = df[f"kp{kp_for_x}_y"].values

    # Convert to radians:
    body_dir = np.deg2rad(df[body_dir_deg_col].values)

    # frames offset
    frame_offset = int(round(window_ms / 1000. * frame_rate))
    half_offset  = frame_offset // 2

    for i in range(n):
        ii = i - half_offset
        jj = i + (frame_offset - half_offset)
        if ii < 0 or jj >= n:
            continue
        if np.isnan(body_dir[ii]) or np.isnan(body_dir[jj]) or np.isnan(X[ii]) or np.isnan(Y[jj]):
            continue

        ang_diff = body_dir[jj] - body_dir[ii]

        if speed_def == 'cum':
            # sum up small steps from ii to jj
            speed = 0.
            for k in range(ii+1, min(jj, n)):
                if np.isnan(X[k]) or np.isnan(X[k-1]):
                    speed = np.nan
                    break
                speed += 100. * np.sqrt((X[k]-X[k-1])**2 + (Y[k]-Y[k-1])**2) / (window_ms/1000.)
        elif speed_def == 'jump':
            # direct jump
            dist = np.sqrt( (X[jj]-X[ii])**2 + (Y[jj]-Y[ii])**2 )
            speed = 100. * dist / (window_ms/1000.)
        else:
            raise ValueError("speed_def must be 'cum' or 'jump'")

        speeds[i] = speed
        dx[i] = speed * np.sin(ang_diff)
        dy[i] = speed * np.cos(ang_diff)

    return dx, dy, speeds

###############################################################################
# 10) Neuro-data placeholders (if needed)
###############################################################################

def load_neuro_data_from_df(neuro_df):
    """
    Placeholder for your new neuro data logic, if needed.
    For example, your DataFrame might have columns like 'cell_id', 'spike_times', etc.
    """
    # Implement your logic
    cell_names = []       # or read from columns
    cell_spikes = []      # ...
    return cell_names, cell_spikes

###############################################################################
# 11) The main pipeline (example)
###############################################################################

def process_tracking_data_df(
    df,
    spineF_kp=4,
    tailB_kp=6,
    spineM_kp=5,
    headX_kp=0,
    headZ_kp=2,
    frame_rate=30.,
    use_expmap=False,
    use_xzy_order=False
):
    """
    Example top-level function to compute:
      - body rotations
      - head rotations
      - angles, derivatives
      - self-motion, etc.
    from a single DataFrame that has all the necessary columns.

    Returns a dictionary of computed results.
    """
    # 1) Body rotations
    r_roots, r_root_inv_oriented, r_root_inv, dir_backs = get_body_rotations_df(df, spineF_kp, tailB_kp, spineM_kp)

    # 2) Head rotations
    global_head_rm, r_heads, body_turned, head_ups = get_head_rotations_df(
        df, headX_kp, headX_kp, headZ_kp,
        r_root_inv=r_root_inv,
        r_root_inv_oriented=r_root_inv_oriented
    )

    # 3) Euler angles or exponential maps
    allo_head_ang, root_ang, ego3_head_ang, ego2_head_ang, back_ang = get_angles(
        r_roots,
        r_heads,
        body_turned,
        global_head_rm,
        dir_backs,
        head_angle_thresh=None,
        use_expmap=use_expmap,
        use_xzy_order=use_xzy_order
    )

    # 4) Convert some angles to degrees, e.g. body direction as -root_ang[:,2]
    body_direction_deg = -root_ang[:,2] * 180.0 / math.pi
    # 5) If desired, compute self-motion using the new body dir col
    #    First, let's put the body direction in the DataFrame:
    df = df.copy()
    df['body_dir_deg'] = body_direction_deg

    dx_jump, dy_jump, speeds_jump = get_selfmotion_df(df, kp_for_x=spineF_kp, 
                                                      body_dir_deg_col='body_dir_deg', 
                                                      frame_rate=frame_rate,
                                                      speed_def='jump',
                                                      window_ms=150)
    dx_cum, dy_cum, speeds_cum   = get_selfmotion_df(df, kp_for_x=spineF_kp, 
                                                     body_dir_deg_col='body_dir_deg', 
                                                     frame_rate=frame_rate,
                                                     speed_def='cum',
                                                     window_ms=150)

    # 6) Example derivative calculations
    #    Suppose we want derivative of body_direction_deg:
    body_dir_1st, body_dir_2nd = calc_der(body_direction_deg, frame_rate, bins_1st=5, bins_2nd=5, is_angle=True, der_2nd=True)

    # 7) Organize results
    results = {
        'r_roots'           : r_roots,
        'r_root_inv'        : r_root_inv,
        'r_root_inv_oriented': r_root_inv_oriented,
        'dir_backs'         : dir_backs,
        'global_head_rm'    : global_head_rm,
        'r_heads'           : r_heads,
        'body_turned'       : body_turned,
        'head_ups'          : head_ups,
        'allo_head_ang'     : allo_head_ang,
        'root_ang'          : root_ang,
        'ego3_head_ang'     : ego3_head_ang,
        'ego2_head_ang'     : ego2_head_ang,
        'back_ang'          : back_ang,
        'body_direction_deg': body_direction_deg,
        'dx_jump'           : dx_jump,
        'dy_jump'           : dy_jump,
        'speeds_jump'       : speeds_jump,
        'dx_cum'            : dx_cum,
        'dy_cum'            : dy_cum,
        'speeds_cum'        : speeds_cum,
        'bodydir_1st'       : body_dir_1st,
        'bodydir_2nd'       : body_dir_2nd
    }

    return results

###############################################################################
# End of Script
###############################################################################
