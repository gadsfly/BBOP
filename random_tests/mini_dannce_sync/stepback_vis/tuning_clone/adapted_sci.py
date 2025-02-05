"""
Performs rotations on tracking data.
@author: Benjamin Adric Dunn
@modified: JingyiGF
@modified: Mir Qi (Gadsfly), 2025 Jan
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

def data_loader_h5(mat_file, imu_file=None, sync_file=None):
    if not os.path.exists(mat_file):
        raise Exception('mat file: %s does not exist !!! Please check the given path.' % mat_file)
    # mat_data = scipy.io.loadmat(mat_file)
    mat_data = pd.read_hdf(mat_file, key='df') #adapting to mir's h5 file
    
    mat_file_name = os.path.splitext(mat_file)[0]
    split_vec = mat_file_name.split(sep="/")
    file_name = split_vec[len(split_vec) - 1]
    mat_data['file_info'] = file_name
    
    return mat_data



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

