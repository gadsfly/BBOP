"""
Performs rotations on tracking data.
@author: Benjamin Adric Dunn
@modified: JingyiGF
@modified: Mir Qi
"""

# import library
import os
from scipy import *
import scipy.ndimage.filters
import scipy.io
import scipy.stats
import scipy.linalg as linalg
import numpy as np
import math
import pickle
import pandas as pd
from scipy.optimize import minimize
import time


rand = np.random.rand

def rotation_x(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    return rx


def rotation_y(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    return ry


def rotation_z(theta):
    ct = np.cos(theta)
    st = np.sin(theta)
    rz = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
    return rz


def eul2rot(ang_vec, order):
    # order is a integer range from 1 to 12, 1-6 are Tait–Bryan angles and 7-12 are Proper(classic) Euler angles
    # 1:x-y-z, 2:y-z-x, 3:z-x-y, 4:x-z-y, 5:z-y-x, 6:y-x-z, note that tracking system use order = 1: xyz
    # ang_vec is Euler angles, x, y, z
    if np.any(np.isnan(ang_vec)):
        rv = np.zeros((3,3))
        rv[:] = np.nan
        return(rv)
    x, y, z = np.ravel(ang_vec)
    rz = rotation_z(z)
    ry = rotation_y(y)
    rx = rotation_x(x)
    if order == 1:  # zyx
        rm = np.dot(rx, np.dot(ry, rz))
    elif order == 2:  # xzy
        rm = np.dot(ry, np.dot(rz, rx))
    elif order == 3:  # yxz
        rm = np.dot(rz, np.dot(rx, ry))
    elif order == 4:  # yzx
        rm = np.dot(rx, np.dot(rz, ry))
    elif order == 5:  # xyz
        rm = np.dot(rz, np.dot(ry, rx))
    elif order == 6:  # zxy
        rm = np.dot(ry, np.dot(rx, rz))
    else:
        raise Exception('Input wrong!')
    return rm


def reformat_data(mat_data):

    print('Processing re-format the original data ...... ')
    if 'pointdatadimensions' not in mat_data.keys():
        raise Exception('Check the mat. It should be a file after tracking system at least.')
    
    pdd = np.ravel(np.array(mat_data['pointdatadimensions'])).astype(int)
    nframes = pdd[2]
    
    head_origin = np.ravel(np.array(mat_data['headorigin']))
    head_origin[np.logical_or(head_origin < -100, head_origin > 100)] = np.nan
    head_origin = np.reshape(head_origin, (pdd[2], 3))
    
    head_x = np.ravel(np.array(mat_data['headX']))
    head_x[np.logical_or(head_x < -100, head_x > 100)] = np.nan
    # nn = head_x + 0
    head_x = np.reshape(head_x, (pdd[2], 3))
    
    head_z = np.ravel(np.array(mat_data['headZ']))
    head_z[np.logical_or(head_z < -100, head_z > 100)] = np.nan
    head_z = np.reshape(head_z, (pdd[2], 3))
    
    point_data = np.ravel(np.array(mat_data['pointdata']))
    point_data[np.logical_or(point_data < -100, point_data > 100)] = np.nan
    point_data = np.reshape(point_data, (pdd[0], pdd[1], pdd[2]))

    # NOTE THERE ARE MORE POINTS BUT CURRENTLY I DO NOT CARE ABOUT THEM
    # 0:3 head, 4 neck, 5 middle, 6 tail
    # start_time = time.time()
    npoints = 7
    sorted_point_data = np.empty((nframes, npoints, 3))
    sorted_point_data[:] = np.nan
    for t in np.arange(nframes):
        marker_lable = point_data[:, 3, t]
        pind = np.where(marker_lable < npoints)[0]
        if len(pind) > 1:
            marker_inuse = marker_lable[pind].astype(int)
            sorted_point_data[t, marker_inuse, :] = point_data[pind, :3, t]
    
        # for j in np.arange(pdd[0]):
        #     for k in np.arange(npoints):
        #         if point_data[j, :, t][3] == k:
        #             sorted_point_data[t, k, :] = point_data[j, :, t][0:3]
    # print("--- %s seconds ---" % (time.time() - start_time))
    return head_origin, head_x, head_z, sorted_point_data


def get_global_head_data(head_x, head_z):
    nframe = len(head_x)
    head_rot_mat = np.zeros((nframe, 3, 3))
    head_rot_mat_inv = np.zeros((nframe, 3, 3))
    head_eul_ang = np.zeros((nframe, 3))
    head_rot_mat[:] = np.nan
    head_rot_mat_inv[:] = np.nan
    head_eul_ang[:] = np.nan
    for t in range(nframe):
        if np.isnan(head_x[t, 0]):
            continue
        hx = head_x[t] / np.linalg.norm(head_x[t])
        hz = head_z[t] / np.linalg.norm(head_z[t])
        hy = np.cross(hz, hx)
        head_rot_mat_inv[t] = np.array([hx, hy, hz])
        head_eul_ang[t, :] = rot2euler(head_rot_mat_inv[t, :, :])
        head_rot_mat[t] = head_rot_mat_inv[t].transpose()
    global_head = {}
    global_head['head_rot_mat_inv'] = head_rot_mat_inv
    global_head['head_eul_ang'] = head_eul_ang
    global_head['head_rot_mat'] = head_rot_mat
    return head_rot_mat_inv, head_eul_ang, head_rot_mat


def get_body_rotations(sorted_point_data):
    print('Processing to get body related rotation matrix ...... ')
    nf = len(sorted_point_data)  # number of frames

    r_roots = np.zeros((nf, 3, 3))  #
    r_root_inv = np.zeros((nf, 3, 3))  #
    r_root_inv_oriented = np.zeros((nf, 3, 3))  #
    dir_backs = np.zeros((nf, 3))  #
    r_roots[:] = np.nan
    r_root_inv[:] = np.nan
    r_root_inv_oriented[:] = np.nan
    dir_backs[:] = np.nan

    timeofheadnans = []
    # DO IT ALL FOR THE NEW ROOT NODE (ASSUMING BUTT TO NECK!)
    for t in np.arange(nf):
        if np.isnan(sorted_point_data[t, 4, 0]) or np.isnan(sorted_point_data[t, 6, 0]):  # 4 is neck and 6 is tail
            continue
    
        # THIS IS A STUPID WAY OF CREATING RROOTS (ANGLE OF BODY RELATIVE TO A WALL)
        # It should just be xdir with z=0...
        xdir = sorted_point_data[t, 4, :] - sorted_point_data[t, 6, :]  # from tail to neck
        xdir[2] = 0.
        ll = np.linalg.norm(xdir)
        if ll < 0.001:
            continue
        xdir = xdir / ll  # x direction of the animal from the top view (chosen)
        zdir = np.ravel(np.array([0, 0, 1]))
        ydir = np.cross(zdir, xdir)  # y direction according to right-hand rule
        shitinv = (np.array([xdir, ydir, zdir])) + 0.  # mapping from global coordinates to animal coordinates
        r_roots[t, :, :] = np.transpose(shitinv)  # mapping from animal to global from the top view
    
        # HERE IS THE REAL ROOT ANGLE!!!
        # SAME AS WITHLOCKINZ = TRUE AND WITHROTATIONS = TRUE
        xdir = sorted_point_data[t, 4, :] - sorted_point_data[t, 6, :]
        ll = np.linalg.norm(xdir)
        if ll < 0.001:
            continue
        xdir = xdir / ll  # unit vector from tail to neck in global coordinates (not top view)
        ydir = np.zeros(3)
        ydir[0] = -xdir[1] + 0.
        ydir[1] = xdir[0] + 0.
        ydir[2] = 0.
        ydir = ydir / np.linalg.norm(ydir)  # an orthogonal vector to the x dir in xy plane
        zdir = np.cross(xdir, ydir)
        zdir = zdir / np.linalg.norm(zdir)
        r_root_inv[t, :, :] = (np.array([xdir, ydir, zdir])) + 0.  # for ego3
    
        # HERE IS THE OTHER OTHER OTHER ROOT ANGLE!!!
        xdir = sorted_point_data[t, 4, :] - sorted_point_data[t, 6, :]
        xdir[2] = 0  # so we just keep the planar direction from the body coordinates!!!
        ll = np.linalg.norm(xdir)
        if ll < 0.001:
            timeofheadnans.append(t)
            continue
        xdir = xdir / ll
        ydir = np.zeros(3)
        ydir[0] = -xdir[1] + 0.
        ydir[1] = xdir[0] + 0.
        ydir[2] = 0.
        ydir = ydir / np.linalg.norm(ydir)
        zdir = np.zeros(3)
        zdir[2] = 1.
        r_root_inv_oriented[t, :, :] = (np.array([xdir, ydir, zdir])) + 0.  # for ego2
    
        # if( WITHLOCKEDINZ == False and WITHROTATIONS == True ):
        # xdir = sorted_point_data[t,4,:] - sorted_point_data[t,6,:]
        # ll = np.linalg.norm(xdir)
        # if(ll<0.001):
        # continue
        # xdir[2] = 0.
        # xdir = xdir / ll
        # ydir = zeros(3)
        # ydir[0] = -xdir[1]+0.
        # ydir[1] = xdir[0]+0.
        # ydir[2] = 0.
        # ydir = ydir/np.linalg.norm(ydir)
        # zdir = zeros(3)
        # zdir[2] = 1.
        # RrootINV = ( np.array([xdir, ydir, zdir]) ) + 0.
        if ~np.isnan(sorted_point_data[t, 6, 0]):
            dir_to_butt = sorted_point_data[t, 4, :] - sorted_point_data[t, 5, :]
            dir_to_butt = dir_to_butt / np.linalg.norm(dir_to_butt)
            dir_to_butt = np.dot(r_root_inv[t, :, :], dir_to_butt)
            dir_backs[t, :] = dir_to_butt + 0.
    
    return r_roots, r_root_inv_oriented, r_root_inv, dir_backs


def get_head_rotations(head_x, head_z, r_root_inv, r_root_inv_oriented):
    print('Processing to get head related rotation matrix ...... ')
    nf = len(head_x)  # number of frames
    
    global_head_rm = np.zeros((nf, 3, 3))  # global head rotation matrix
    r_heads = np.zeros((nf, 3, 3))  #
    head_ups = np.zeros(nf)  #
    body_turned_heads = np.zeros((nf, 3, 3))  #
    
    body_turned_heads[:] = np.nan
    head_ups[:] = np.nan
    global_head_rm[:] = np.nan
    r_heads[:] = np.nan

    for t in np.arange(nf):
        if (r_root_inv is not None) and (r_root_inv_oriented is not None):
            if np.isnan(r_root_inv[t, 0, 0]):
                continue
        else:
            print('Only global head rotation matrix is generated.')

        if ~np.isnan(head_x[t, 0]):
            hx = head_x[t] / np.linalg.norm(head_x[t])
            hz = head_z[t] / np.linalg.norm(head_z[t])
            hy = np.cross(hz, hx)
            global_head_rm[t, :, :] = np.array([hx, hy, hz])  # global to head

            if (r_root_inv is not None) and (r_root_inv_oriented is not None):
                rhx = np.dot(r_root_inv[t, :, :], hx)
                rhz = np.dot(r_root_inv[t, :, :], hz)
                rhy = np.dot(r_root_inv[t, :, :], hy)
                r_heads[t, :, :] = np.array([rhx, rhy, rhz])  # ego3_head_rotm
                # hy = head_y[t]/np.linalg.norm(head_y[t])  ## these were screwed up at in head csys maker of gui (oops)
                # rhy = dot(RrootINV, hy)
                
                rhx2 = np.dot(r_root_inv_oriented[t, :, :], hx)
                rhz2 = np.dot(r_root_inv_oriented[t, :, :], hz)
                rhy2 = np.dot(r_root_inv_oriented[t, :, :], hy)
                body_turned_heads[t, :, :] = np.array([rhx2, rhy2, rhz2])  # ego2_head_rotm

            head_ups[t] = np.dot(hx, [0, 0, 1])
        
    return global_head_rm, r_heads, body_turned_heads, head_ups


def check_rot_angs(angs):
    
    for i in range(3):
        if angs[i] > math.pi:
            angs[i] -= 2. * math.pi
        if angs[i] < - math.pi:
            angs[i] += 2. * math.pi
    count_big = sum(abs(angs) > math.pi / 2.)
    return angs, count_big


def rot2expmap(rot_mat):
    # convert to quaternions
    d = rot_mat - np.transpose(rot_mat)
    r = np.zeros(3)
    r[0] = -d[1, 2]
    r[1] = d[0, 2]
    r[2] = -d[0, 1]
    sintheta = linalg.norm(r) / 2.
    r0 = r / (linalg.norm(r) + np.finfo(float).eps)  # get epsilon
    costheta = (np.trace(rot_mat) - 1.) / 2.
    theta = math.atan2(sintheta, costheta)
    coshalftheta = math.cos(theta / 2.)
    sinhalftheta = math.sin(theta / 2.)

    # now convert to 'exponential map'
    theta = 2 * math.atan2(sinhalftheta, coshalftheta)
    theta = np.fmod(theta + 2 * math.pi, 2 * math.pi)  # Remainder after division (modulo operation)
    if theta > math.pi:
        theta = 2 * math.pi - theta
        r0 = -r0
    r = r0 * theta
    # note, this code is adapted from Graham's stuff
    return r


def rot2euler_xzy(rot_mat):
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = np.ravel(rot_mat)
    temp = math.sqrt(r22 * r22 + r32 * r32)
    if temp > 0.0000001:
        x = np.arctan2(r32, r22)
        z = np.arctan2(-r12, temp)
        y = np.arctan2(r13, r11)
    else:
        x = 0.0
        z = np.arctan2(-r12, temp)
        y = np.arctan2(r13, r11)
    
    # CHECK FOR THE ALTERNATIVE SOLUTION DUDE!!!
    return np.ravel(np.array([x, y, z]))


def rot2euler(rot_matrix, use_solution_with_least_tilt=False):
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = np.ravel(rot_matrix)
    temp = math.sqrt(r33 * r33 + r23 * r23)
    angs = np.zeros(3)
    if temp > 0.0001:
        angs[0] = np.arctan2(-r23, r33)
        angs[1] = np.arctan2(r13, temp)
        angs[2] = np.arctan2(-r12, r11)
    else:
        angs[0] = 0.0
        angs[1] = np.arctan2(r13, temp)
        angs[2] = np.arctan2(r21, r22)
    angs, count = check_rot_angs(angs)
    
    if use_solution_with_least_tilt:
        angs_o = np.zeros(3)
        angs_o[0] = angs[0] + math.pi
        angs_o[1] = math.pi - angs[1]
        angs_o[2] = angs[2] + math.pi
        angs_o, count_o = check_rot_angs(angs_o)
        
        if abs(angs_o[1]) < abs(angs[1]):
            angs = angs_o
    
    return angs


def rot2angles(rotmat, useexpmap, use_xzy_order, usesolutionwithleasttilt=False):
    if np.isnan(rotmat[0, 0]):
        return np.nan
    if useexpmap:
        return rot2expmap(rotmat)
    if use_xzy_order:
        return rot2euler_xzy(rotmat)
    return rot2euler(rotmat, usesolutionwithleasttilt)


def get_back_angles(p):
    # get rotations around Z and Y axis such that dot(R, [1,0,0]) = point
    # this gets us ROT = dot(Zrot(Rz),Yrot(Ry))
    # note that if it is more than 90 degrees difference in Ry then it gets strange,
    # meaning that it finds equivalent euler angles that just go the other direction (suck)
    p = p / np.linalg.norm(p)
    ry = - np.arctan2(p[2], math.sqrt(p[0] ** 2 + p[1] ** 2))
    rz = np.arctan2(p[1], p[0])
    return ry, rz


def get_angles(r_roots, r_heads, body_turned_heads, global_head_rot_mat, dir_backs, opt_rotated_dir_back,
               head_angle_thresh, use_expmap, use_xzy_order,
               use_solution_with_least_tilt=False):
    print('Processing to get all Euler angles ...... ')
    nf = len(r_heads)
    
    a_head_ang = np.zeros((nf, 3))  # Euler angles of head
    root_ang = np.zeros((nf, 3))
    ef_head_ang = np.zeros((nf, 3))
    es_head_ang = np.zeros((nf, 3))
    angs_backs = np.zeros((nf, 2))
    opt_angs_backs = np.zeros((nf, 2))
    
    for t in np.arange(nf):
        # roots coordinate of the animal (body) (Rroot)
        root_ang[t, :] = rot2angles(r_roots[t, :, :], use_expmap, use_xzy_order, use_solution_with_least_tilt)
        # head relative to the body (ego_head_angels, facing) (EMroots) # ego3
        ef_head_ang[t, :] = rot2angles(r_heads[t, :, :], use_expmap, use_xzy_order, use_solution_with_least_tilt)
        # ego_head_angles (skelton) (EMroots2) # ego2
        es_head_ang[t, :] = rot2angles(body_turned_heads[t, :, :], use_expmap, use_xzy_order, use_solution_with_least_tilt)

        a_head_ang[t, :] = rot2angles(global_head_rot_mat[t, :, :], use_expmap, use_xzy_order, use_solution_with_least_tilt)
        opt_angs_backs[t, :] = get_back_angles(opt_rotated_dir_back[t, :])
        angs_backs[t, :] = get_back_angles(dir_backs[t, :])
        # if(False):
        #   yaxis = ravel(array([0.,1.,0.]))
        #   y = AngsBacks[t,0]
        #   z = AngsBacks[t,1]
        #   Rz = array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
        #   Ry = array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
        #   yaxis = dot(dot(Rz, Ry), yaxis)
    if head_angle_thresh is not None:
        ego2_up, ego2_down, ego3_up, ego3_down = np.ravel(head_angle_thresh)
        for t in np.arange(nf):
            if body_turned_heads[t, 0, 2] > ego2_up or body_turned_heads[t, 0, 2] < ego2_down:
                es_head_ang[t, 0] = np.nan
                es_head_ang[t, 2] = np.nan
                a_head_ang[t, 0] = np.nan
                a_head_ang[t, 2] = np.nan
            if r_heads[t, 0, 2] > ego3_up or r_heads[t, 0, 2] < ego3_down:
                ef_head_ang[t, 0] = np.nan
                ef_head_ang[t, 2] = np.nan

    es_head_ang[:, 0] = -es_head_ang[:, 0]
    a_head_ang[:, 0] = -a_head_ang[:, 0]
    ef_head_ang[:, 0] = -ef_head_ang[:, 0]
    
    return a_head_ang, root_ang, ef_head_ang, es_head_ang, angs_backs, opt_angs_backs


def optrotate(avec):
    print('processing to get opt rotation .....')
    n = len(avec[0])
    nframe = len(avec)
    
    def get_rot(tryang):
        # rotations about z (azimuth) are ok but bounds on the other two to avoid flipping
        if abs(tryang[0]) > math.pi * 0.5 or abs(tryang[1]) > math.pi * 0.5:
            return [-1], [-1]
        rmat_z = rotation_z(tryang[2])
        rmat_y = rotation_y(tryang[1])
        rmat_x = rotation_x(tryang[0])
        rot_mat = np.dot(rmat_x, np.dot(rmat_y, rmat_z))
        new_vec = np.zeros(np.shape(avec))
        new_vec[:] = np.nan
        for t in range(nframe):
            if np.isnan(avec[t, 0]):
                continue
            new_vec[t] = np.dot(rot_mat, avec[t])
        return new_vec, rot_mat
    
    def distance_to_xaxis(tryang):  # first just rotate around azimuth and tilt to center around zero
        chk, rot_mat = get_rot([0., tryang[0], tryang[1]])
        if len(np.ravel(chk)) < 3:
            return 1000000000000000000000000000.
        xvec = np.zeros(3)
        if n == 3:
            xvec[0] = np.nanmean(chk[:, 0])
            xvec[1] = np.nanmean(chk[:, 1])
            xvec[2] = np.nanmean(chk[:, 2])
        else:
            xvec[0] = np.nanmean(chk[:, 0, 0])
            xvec[1] = np.nanmean(chk[:, 0, 1])
            xvec[2] = np.nanmean(chk[:, 0, 2])
        
        ll = np.sqrt(np.sum(xvec ** 2))
        xvec = xvec / ll
        bvec = np.ravel(np.array([1, 0, 0]))
        angle = np.arctan2(linalg.norm(np.cross(xvec, bvec)), np.dot(xvec, bvec))
        # print tryang[0], tryang[1], angle
        return abs(angle)
    
    res = minimize(distance_to_xaxis, np.array([-0.25 * math.pi + rand() * 0.5 * math.pi, -0.25 * math.pi + rand() * 0.5 * math.pi]),
                   method='nelder-mead', options={'xtol': 1e-6, 'disp': True})
    best_ang = res.x
    print(('opt rotation', best_ang, 'max possible rotation', math.pi * 0.5))
    return get_rot([0., best_ang[0], best_ang[1]])


def calc_der(values, framerate, bins_1st, bins_2nd, is_angle, der_2nd, session_indicator=None):
    # central difference derivative
    nframe = len(values)
    dims_val = values.shape
    if len(dims_val) == 1:
        ncol_val = 1
    else:
        ncol_val = dims_val[1]
    value_2nd_der = np.zeros(dims_val)
    value_1st_der = np.zeros(dims_val)
    
    for i in range(ncol_val):
        first_der = np.zeros(nframe)
        second_der = np.zeros(nframe)
        if ncol_val == 1:
            val = values
        else:
            val = values[:,i]
        for t in range(nframe):
            ts = t - bins_1st
            te = t + bins_1st
            if ts < 0 or te > nframe - 1 or np.isnan(val[ts]) or np.isnan(val[te]):
                first_der[t] = np.nan
                continue
            first_der[t] = val[te] - val[ts]
            if is_angle:
                if first_der[t] > 180:
                    first_der[t] -= 360.
                if first_der[t] < -180:
                    first_der[t] += 360.
            if len(np.ravel(framerate)) > 1:
                first_der[t] /= (2. * bins_1st / framerate[session_indicator[t]])
            else:
                first_der[t] /= (2. * bins_1st / framerate)
        
        if der_2nd:
            for t in range(nframe):
                ts = t - bins_2nd
                te = t + bins_2nd
                if ts < 0 or te > nframe - 1 or np.isnan(first_der[ts]) or np.isnan(first_der[te]):
                    second_der[t] = np.nan
                    continue
                second_der[t] = first_der[te] - first_der[ts]
                if len(np.ravel(framerate)) > 1:
                    second_der[t] /= (2. * bins_1st / framerate[session_indicator[t]])
                else:
                    second_der[t] /= (2. * bins_1st / framerate)
        if ncol_val == 1:
            value_2nd_der = second_der + 0.
            value_1st_der = first_der + 0.
        else:
            value_2nd_der[:, i] = second_der + 0.
            value_1st_der[:, i] = first_der + 0.
        
    return value_1st_der, value_2nd_der


def get_selfmotion(xvals, yvals, body_dir, frame_rate, speed_def, movement_offset, session_indicator=None):

    dx = np.zeros(len(xvals))
    dy = np.zeros(len(xvals))
    speeds = np.zeros(len(xvals))
    
    if len(np.ravel(frame_rate)) > 1:
        for k in range(len(frame_rate)):
            da_frame_rate = frame_rate[k]
            da_index = np.where(session_indicator == k)[0]
            
            fsmo = float(movement_offset) * da_frame_rate / 1000.
            sm = -int(np.floor(fsmo / 2.))
            em = int(np.floor((fsmo + sm)))
            
            for i in da_index:
                ii = i + sm
                jj = i + em
                if ii < 0 or jj > len(xvals) - 1:
                    dx[i] = np.nan
                    dy[i] = np.nan
                    speeds[i] = np.nan
                    continue
                if np.isnan(body_dir[ii]) or np.isnan(body_dir[jj]) or np.isnan(xvals[ii]) or np.isnan(yvals[jj]):
                    dx[i] = np.nan
                    dy[i] = np.nan
                    speeds[i] = np.nan
                    continue
                ang_diff = body_dir[jj] - body_dir[ii]
                if speed_def == 'cum':
                    speed = 0.
                    for xi in np.arange(ii + 1, min([jj, len(body_dir) - 1]), 1):
                        speed += 100. * np.sqrt((xvals[xi] - xvals[xi - 1]) ** 2 + (yvals[xi] - yvals[xi - 1]) ** 2) / (
                                    movement_offset / 1000.)
                elif speed_def == 'jump':
                    speed = 100. * np.sqrt((xvals[jj] - xvals[ii]) ** 2 + (yvals[jj] - yvals[ii]) ** 2) / (movement_offset / 1000.)
                else:
                    raise Exception('Speed definition is not defined !!!')
                speeds[i] = speed + 0.
                dx[i] = speed * np.sin(ang_diff)
                dy[i] = speed * np.cos(ang_diff)
                if 10000 < i < 10015:
                    print('Time point, %d, movement vector (%f, %f)' % (i, dx[i], dy[i]))  # , 'angle diff', ang_diff, 'speed', speed[i]
    else:
        fsmo = float(movement_offset) * frame_rate / 1000.
        sm = -int(np.floor(fsmo / 2.))
        em = int(np.floor((fsmo + sm)))
    
        for i in range(len(body_dir)):
            ii = i + sm
            jj = i + em
            if ii < 0 or jj > len(xvals) - 1:
                dx[i] = np.nan
                dy[i] = np.nan
                speeds[i] = np.nan
                continue
            if np.isnan(body_dir[ii]) or np.isnan(body_dir[jj]) or np.isnan(xvals[ii]) or np.isnan(yvals[jj]):
                dx[i] = np.nan
                dy[i] = np.nan
                speeds[i] = np.nan
                continue
            ang_diff = body_dir[jj] - body_dir[ii]
            if speed_def == 'cum':
                speed = 0.
                for xi in np.arange(ii + 1, min([jj, len(body_dir) - 1]), 1):
                    speed += 100. * np.sqrt((xvals[xi] - xvals[xi - 1]) ** 2 + (yvals[xi] - yvals[xi - 1]) ** 2) / (
                            movement_offset / 1000.)
            elif speed_def == 'jump':
                speed = 100. * np.sqrt((xvals[jj] - xvals[ii]) ** 2 + (yvals[jj] - yvals[ii]) ** 2) / (
                            movement_offset / 1000.)
            else:
                raise Exception('Speed definition is not defined !!!')
            speeds[i] = speed + 0.
            dx[i] = speed * np.sin(ang_diff)
            dy[i] = speed * np.cos(ang_diff)
            if 10000 < i < 10015:
                print('Time point, %d, movement vector (%f, %f)' % (
                i, dx[i], dy[i]))  # , 'angle diff', ang_diff, 'speed', speed[i]
    return dx, dy, speeds


def calc_selfmotion(data, selfmotion_param=(150, 250), calc_derivatives=False, add_point=None):
    sorted_point_data = data['sorted_point_data']
    selfmotion_window_size = data['settings']['selfmotion_window_size']
    body_direction_radiance = data['body_direction'] / 180. * math.pi
    frame_rate = data['framerate']
    bins_der = data['settings']['bins_der']
    n_frame = len(sorted_point_data)
    
    if len(np.ravel(frame_rate)) > 1:
        session_indicator = data['session_indicator']
    else:
        session_indicator = np.zeros(n_frame)

    if selfmotion_param is None and add_point is None:
        return []
    
    if selfmotion_param is not None:
        add_point = 'neck'
        selfmotion_window_size = selfmotion_param
        print('recalculate selfmotion speeds using neck point.')
    
    if add_point == 'tail':
        self_x = sorted_point_data[:, 6, 0]
        self_y = sorted_point_data[:, 6, 1]
    elif add_point == 'back':
        self_x = sorted_point_data[:, 5, 0]
        self_y = sorted_point_data[:, 5, 1]
    else:
        self_x = sorted_point_data[:, 4, 0]
        self_y = sorted_point_data[:, 4, 1]
    
    if isinstance(selfmotion_window_size, int):
        dx_jump, dy_jump, speeds_jump = get_selfmotion(self_x, self_y, body_direction_radiance, frame_rate, 'jump',
                                                       selfmotion_window_size, session_indicator)
        dx_cums, dy_cums, speeds_cums = get_selfmotion(self_x, self_y, body_direction_radiance, frame_rate, 'cum',
                                                       selfmotion_window_size, session_indicator)
        speeds = np.column_stack([speeds_jump, speeds_cums])
        selfmotion_mat = np.column_stack([dx_jump, dy_jump, dx_cums, dy_cums])
    else:
        dx, dy, speeds0 = get_selfmotion(self_x, self_y, body_direction_radiance, frame_rate, 'jump',
                                         selfmotion_window_size[0], session_indicator)
        speeds = speeds0.copy()
        speeds = np.reshape(speeds, (n_frame, 1))
        selfmotion_mat = np.column_stack([dx, dy])
        for ws in selfmotion_window_size[1:]:
            dx, dy, speeds0 = get_selfmotion(self_x, self_y, body_direction_radiance, frame_rate, 'jump', ws, session_indicator)
            speeds = np.append(speeds, np.reshape(speeds0, (n_frame, 1)), 1)
            selfmotion_mat = np.append(selfmotion_mat, np.column_stack([dx, dy]), 1)
        for ws in selfmotion_window_size:
            dx, dy, speeds0 = get_selfmotion(self_x, self_y, body_direction_radiance, frame_rate, 'cum', ws, session_indicator)
            speeds = np.append(speeds, np.reshape(speeds0, (n_frame, 1)), 1)
            selfmotion_mat = np.append(selfmotion_mat, np.column_stack([dx, dy]), 1)
            
    speeds_1st_der = []
    if calc_derivatives:
        speeds_1st_der, speeds_2nd_der = calc_der(speeds, frame_rate, bins_der[0], bins_der[1], False, False, session_indicator)
    return speeds, selfmotion_mat, speeds_1st_der


def make_ang_continue(angs, degrees=True):
    if degrees:
        svec = np.array([-180, 0, 180])
    else:
        svec = np.array([-180, 0, 180]) / 180 * math.pi
    processed_ang = angs[~np.isnan(angs)].copy()
    n_ang = len(processed_ang)
    shift_vec = np.zeros((n_ang))
    final_ang = np.zeros((len(angs)))
    final_ang[:] = np.nan
    temp_ang = np.zeros((n_ang))
    temp_ang[:] = np.nan
    temp_ang[0] = processed_ang[0]
    # diff_ang = np.diff(processed_ang)
    for i in range(1, n_ang):
        diff_ang = processed_ang[i] - processed_ang[i-1]
        d_vec = diff_ang + svec
        idx = np.argmin(abs(d_vec))
        shift_vec[i] = shift_vec[i-1] + svec[idx]
        temp_ang[i] = processed_ang[i] + shift_vec[i]
    final_ang[~np.isnan(angs)] = temp_ang
    return final_ang


def data_loader(mat_file, imu_file=None, sync_file=None):
    if not os.path.exists(mat_file):
        raise Exception('mat file: %s does not exist !!! Please check the given path.' % mat_file)
    mat_data = scipy.io.loadmat(mat_file)
    
    mat_file_name = os.path.splitext(mat_file)[0]
    split_vec = mat_file_name.split(sep="/")
    file_name = split_vec[len(split_vec) - 1]
    mat_data['file_info'] = file_name
    
    if imu_file is not None:
        if not os.path.exists(imu_file):
            raise Exception('imu file: %s does not exist !!! Please check the given path.' % imu_file)
        imu_data = pd.read_pickle(imu_file)
        if sync_file is None:
            raise Exception('when imu file is given. the sync file must be given.')
        if not os.path.exists(sync_file):
            raise Exception('sync file: %s does not exist !!! Please check the given path.' % sync_file)
        sync_info = pd.read_pickle(sync_file)
        
        return mat_data, imu_data, sync_info
    
    return mat_data


def get_cell_data(mat_data):

    print('Processing to re-format spikes data ...... ')
    session_ts = np.ravel(mat_data['sessionTS'])
    
    kk = (list(mat_data.keys()))
    count = 0
    for i in np.arange(len(list(mat_data.keys()))):
        if 'cellname_' in (list(mat_data.keys()))[i]:
            count += 1
    
    cell_names = []
    cell_activities = []
    for i in np.arange(count):
        cn = 'cellname_%05d' % i
        if cn in mat_data:
            cell_names.append((mat_data[cn])[0])
        cn = 'cell_%05d' % i
        if cn in mat_data:  # read in spike times in time stamps
            cell_activities.append(np.ravel(np.array(mat_data[cn])) + session_ts[0])
    
    for i in np.arange(count):
        print((i, (cell_names[i], (cell_activities[i])[:10])))
    print('')
    print(count)
    if len(cell_names) < 1:
        print('There is no cell data in here!')
        return [], []
    
    return cell_names, cell_activities


def data_generator(data, head_angle_thresh=(0.9, -0.9, 0.9, -0.9), use_expmap=False, use_xzy_order=False,
                   use_solution_with_least_tilt=False,
                   selfmotion_window_size=(150, 250), bins_der=(10, 10)):

    settings = {'head_angle_thresh': head_angle_thresh, 'use_expmap': use_expmap, 'use_xzy_order': use_xzy_order,
                'use_solution_with_least_tilt': use_solution_with_least_tilt,
                'selfmotion_window_size': selfmotion_window_size, 'bins_der': bins_der}
    dt_keys = (list(data.keys()))
    if 'file_info' not in dt_keys:
        raise Exception('Please use the function data_loader() first.')
    
    bins_1st = bins_der[0]
    bins_2nd = bins_der[1]
    
    frame_rate = np.ravel(data['framerate'])
    if len(frame_rate) > 1:
        session_indicator = np.ravel(data['session_indicator'])
        frame_times = np.ravel(data['tracking_times'])
        full_tracking_ts = np.ravel(data['full_trackingTS'])
    else:
        frame_rate = frame_rate[0]
        session_indicator = np.zeros(len(data['headX']))
        frame_times = None
        full_tracking_ts = None
        
    cell_count = 0
    i = 0
    include_spikes = True
    while cell_count == 0 and i < len(dt_keys):
        if 'cellname_' in dt_keys[i]:
            cell_count += 1
        i += 1
    if cell_count == 0:
        print(
            'No spikes info in the data !!! This processing will not stop, but please make sure you do not need cell data !')
        print('IF you need cell data, please press Ctrl + C to stop this processing.')
        include_spikes = not include_spikes
    
    if include_spikes:
        cell_names, cell_activities = get_cell_data(data)
    else:
        cell_names = None
        cell_activities = None
    
    # if (not all(elem in np.arange(cell_count) for elem in cell_index)):
    #     raise Exception('In total %d cells in the data. Please give the correct index !!!' % cell_count)
    
    head_origin, head_x, head_z, sorted_point_data = reformat_data(data)
    print(('HEAD X is NON-NAN for', np.sum(~np.isnan(head_z[:, 0])), 'of', len(head_z[:, 0]), 'bins'))
    # Assume neck is where the animal is -- why not?
    # animal_location = sorted_point_data[:, 4, :]
    
    n_frame = len(head_x)
    
    r_roots, r_root_inv_oriented, r_root_inv, dir_backs = get_body_rotations(sorted_point_data)
    opt_rotated_dir_backs, back_rotation_rot_mat = optrotate(dir_backs)
    
    global_head_rot_mat, r_heads, body_turned_heads, head_ups = get_head_rotations(head_x, head_z, r_root_inv,
                                                                                   r_root_inv_oriented)
    
    # ego3 is EMheads
    # ego2 is EMheads2
    allo_head_ang, root_ang, ego3_head_ang, ego2_head_ang, back_ang, opt_back_ang = \
        get_angles(r_roots, r_heads, body_turned_heads, global_head_rot_mat, dir_backs, opt_rotated_dir_backs,
                   head_angle_thresh, use_expmap, use_xzy_order, use_solution_with_least_tilt)
    
    neck_elevation = sorted_point_data[:, 4, 2]
    self_x = sorted_point_data[:, 4, 0]  # animal_location[:,0]
    self_y = sorted_point_data[:, 4, 1]  # animal_location[:,1]
    
    if isinstance(selfmotion_window_size, int):
        dx_jump, dy_jump, speeds_jump = get_selfmotion(self_x, self_y, -root_ang[:, 2], frame_rate, 'jump',
                                                       selfmotion_window_size, session_indicator)
        dx_cums, dy_cums, speeds_cums = get_selfmotion(self_x, self_y, -root_ang[:, 2], frame_rate, 'cum',
                                                       selfmotion_window_size, session_indicator)
        speeds = np.column_stack([speeds_jump, speeds_cums])
        selfmotion_mat = np.column_stack([dx_jump, dy_jump, dx_cums, dy_cums])
    else:
        dx, dy, speeds0 = get_selfmotion(self_x, self_y, -root_ang[:, 2], frame_rate, 'jump', selfmotion_window_size[0], session_indicator)
        speeds = speeds0.copy()
        speeds = np.reshape(speeds, (n_frame, 1))
        selfmotion_mat = np.column_stack([dx, dy])
        for ws in selfmotion_window_size[1:]:
            dx, dy, speeds0 = get_selfmotion(self_x, self_y, -root_ang[:, 2], frame_rate, 'jump', ws, session_indicator)
            speeds = np.append(speeds, np.reshape(speeds0, (n_frame, 1)), 1)
            selfmotion_mat = np.append(selfmotion_mat, np.column_stack([dx, dy]), 1)
        for ws in selfmotion_window_size:
            dx, dy, speeds0 = get_selfmotion(self_x, self_y, -root_ang[:, 2], frame_rate, 'cum', ws, session_indicator)
            speeds = np.append(speeds, np.reshape(speeds0, (n_frame, 1)), 1)
            selfmotion_mat = np.append(selfmotion_mat, np.column_stack([dx, dy]), 1)
    
    speeds_1st_der, speeds_2nd_der = calc_der(speeds, frame_rate, bins_1st, bins_2nd, False, False, session_indicator)
    neck_1st_der, neck_2nd_der = calc_der(neck_elevation, frame_rate, bins_1st, bins_2nd, False, True, session_indicator)
    
    allo_head_ang = allo_head_ang / math.pi * 180.
    body_direction = -root_ang[:, 2] / math.pi * 180.
    ego3_head_ang = ego3_head_ang / math.pi * 180.
    ego2_head_ang = ego2_head_ang / math.pi * 180.
    back_ang = -back_ang / math.pi * 180.
    opt_back_ang = -opt_back_ang / math.pi * 180.
    
    allo_head_1st_der, allo_head_2nd_der = calc_der(allo_head_ang, frame_rate, bins_1st, bins_2nd, True, True, session_indicator)
    bodydir_1st_der, bodydir_2nd_der = calc_der(body_direction, frame_rate, bins_1st, bins_2nd, True, True, session_indicator)
    ego3_head_1st_der, ego3_head_2nd_der = calc_der(ego3_head_ang, frame_rate, bins_1st, bins_2nd, True, True, session_indicator)
    ego2_head_1st_der, ego2_head_2nd_der = calc_der(ego2_head_ang, frame_rate, bins_1st, bins_2nd, True, True, session_indicator)
    back_1st_der, back_2nd_der = calc_der(back_ang, frame_rate, bins_1st, bins_2nd, True, True, session_indicator)
    opt_back_1st_der, opt_back_2nd_der = calc_der(opt_back_ang, frame_rate, bins_1st, bins_2nd, True, True, session_indicator)
    
    file_name = data['file_info']
    if use_expmap:
        output_file_name = '%s_expmap' % file_name
    else:
        if use_xzy_order:
            output_file_name = '%s_XZYeuler' % file_name
        else:
            output_file_name = '%s_XYZeuler' % file_name
    if use_solution_with_least_tilt:
        output_file_name = '%s_leasttilt' % output_file_name
    else:
        output_file_name = '%s_notricks' % output_file_name
    
    new_data = {'file_info': output_file_name,
                'settings': settings,
                'session_indicator': session_indicator,
                'frame_times': frame_times,
                'full_tracking_ts': full_tracking_ts,
                'framerate': frame_rate,
                'point_data_dimensions': np.ravel(data['pointdatadimensions']),
                'tracking_ts': np.ravel(data['trackingTS']),
                'session_ts': np.ravel(data['sessionTS']),
                'ratcam_ts': np.ravel(data['ratcamTS']),
                'bbtrans_xy': np.ravel(data['bbtransXY']),
                'bbscale_xy': np.ravel(data['bbscaleXY']),
                'bbrot': np.ravel(data['bbrot']),
                'head_origin': head_origin,
                'head_x': head_x,
                'head_z': head_z,
                'sorted_point_data': sorted_point_data,
                'cell_names': cell_names,
                'cell_activities': cell_activities,
                'allo_head_rotm': global_head_rot_mat,
                'r_root_inv': r_root_inv,
                'r_root_inv_oriented': r_root_inv_oriented,
                'ego2_head_rotm': body_turned_heads,
                'ego3_head_rotm': r_heads,
                'allo_head_ang': allo_head_ang,
                'body_direction': body_direction,
                'ego3_head_ang': ego3_head_ang,
                'ego2_head_ang': ego2_head_ang,
                'back_ang': back_ang,
                'opt_back_ang': opt_back_ang,
                'speeds': speeds,
                'selfmotion': selfmotion_mat,
                'speeds_1st_der': speeds_1st_der,
                'neck_1st_der': neck_1st_der,
                'neck_2nd_der': neck_2nd_der,
                'allo_head_1st_der': allo_head_1st_der,
                'allo_head_2nd_der': allo_head_2nd_der,
                'bodydir_1st_der': bodydir_1st_der,
                'bodydir_2nd_der': bodydir_2nd_der,
                'ego3_head_1st_der': ego3_head_1st_der,
                'ego3_head_2nd_der': ego3_head_2nd_der,
                'ego2_head_1st_der': ego2_head_1st_der,
                'ego2_head_2nd_der': ego2_head_2nd_der,
                'back_1st_der': back_1st_der,
                'back_2nd_der': back_2nd_der,
                'opt_back_1st_der': opt_back_1st_der,
                'opt_back_2nd_der': opt_back_2nd_der}
    
    return new_data


def merge_comparing_data(data1, data2, file_info=None):
    keys1 = data1.keys()
    keys2 = data2.keys()
    if file_info is None:
        finfo1 = data1['file_info']
        finfo2 = data2['file_info']

        split_fi1 = finfo1.split(sep="_")
        split_fi2 = finfo2.split(sep="_")
        
        n_info = len(split_fi1)
        same_info_vec = [split_fi1[ind] for ind in range(n_info) if split_fi1[ind] in split_fi2]
        file_info = ''
        for i in range(n_info-1):
            file_info = file_info + same_info_vec[i]
            file_info = file_info + '_'
        file_info = file_info + same_info_vec[n_info-1]

    n_merged_sessions = len(np.ravel(data1['framerate']))
    if n_merged_sessions != 1:
        for i in range(n_merged_sessions):
            if abs(data1['framerate'][i] - data2['framerate'][i]) > 1e-6:
                raise Exception('merging files need to have same frame rate !!!')
    else:
        if abs(data1['framerate'] - data2['framerate']) > 1e-6:
            raise Exception('merging files need to have same frame rate !!!')
    
    if ~np.all(data1['tracking_ts'] == data2['tracking_ts']):
        raise Exception('merging files need to have same tracking time stamps !!!')

    if ~np.all(data1['session_ts'] == data2['session_ts']):
        raise Exception('merging files need to have same session time stamps !!!')
    
    if ~np.all(data1['ratcam_ts'] == data2['ratcam_ts']):
        raise Exception('merging files need to have same ratcam time stamps !!!')
    
    settings1 = data1['settings']
    settings2 = data2['settings']
    if len(settings1) != len(settings2):
        raise Exception('merging files need to have same settings !!!')
    keys_s1 = settings1.keys()
    keys_s2 = settings2.keys()
    for dakey in keys_s1:
        if dakey not in keys_s2:
            raise Exception('merging files need to have same settings !!!')
        else:
            if ~np.all(settings1[dakey] == settings2[dakey]):
                raise Exception('merging files need to have same %s settings !!!' % dakey)
    
    data = data1.copy()
    data['file_info'] = file_info
    data['cf_sorted_point_data'] = data2['sorted_point_data']
    data['cf_bbtrans_xy'] = data2['bbtrans_xy']
    data['cf_bbscale_xy'] = data2['bbscale_xy']
    data['cf_bbrot'] = data2['bbrot']
    data['cf_head_origin'] = data2['head_origin']
    data['cf_head_x'] = data2['head_x']
    data['cf_head_z'] = data2['head_z']
    data['cf_allo_head_rotm'] = data2['allo_head_rotm']
    data['cf_r_root_inv'] = data2['r_root_inv']
    data['cf_r_root_inv_oriented'] = data2['r_root_inv_oriented']
    data['cf_ego2_head_rotm'] = data2['ego2_head_rotm']
    data['cf_ego3_head_rotm'] = data2['ego3_head_rotm']
    data['cf_allo_head_ang'] = data2['allo_head_ang']
    data['cf_body_direction'] = data2['body_direction']
    data['cf_ego3_head_ang'] = data2['ego3_head_ang']
    data['cf_ego2_head_ang'] = data2['ego2_head_ang']
    data['cf_back_ang'] = data2['back_ang']
    data['cf_opt_back_ang'] = data2['opt_back_ang']
    data['cf_speeds'] = data2['speeds']
    data['cf_selfmotion'] = data2['selfmotion']
    data['cf_speeds_1st_der'] = data2['speeds_1st_der']
    data['cf_neck_1st_der'] = data2['neck_1st_der']
    data['cf_back_ang'] = data2['back_ang']
    data['cf_back_ang'] = data2['back_ang']
    
    return data
