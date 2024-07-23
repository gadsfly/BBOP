"""
This should be run in the project folder that has com3d_used.mat and save_data_MAX.mat

Anshu: 07/18/24 save the smoothed preds in the predictions directory
"""
import scipy.io as sio
import numpy as np
import scipy.signal as signal
import os

path = "/home/lq53/mir_data/24summ/2024_06_26/1686940_left/DANNCE/predict_results/arnav_label2"
com_file = os.path.join(path,'com3d_used.mat')
com_data = sio.loadmat(com_file)

pred_file = os.path.join(path,'save_data_AVG0.mat')
# pred_file = 'DANNCE/predict_results/twd5/save_data_MAX0.mat'
pred_data = sio.loadmat(pred_file)

true_pred = np.zeros(pred_data['pred'].shape)
# shape: (30000, 3, 22)


for i in range(true_pred.shape[0]):  # range among the frames
    for j in range(22):
        true_pred[i,:,j] = pred_data['pred'][i,:,j] #+ com_data['com'][i,:]

#### check some random point
# print(com_data['com'][0,1])
# print(pred_data['pred'][0,1,2])
# print(true_pred[0,1,2])

#### Save true prediction #####
# true_pred_file = 'true_prediction.mat'
# sio.savemat(true_pred_file, {'pred':true_pred})
# print('True prediction saved')


################### Filter ##################
# filter parameters set up

smoothed_pred = true_pred
for i in range(22):
    for j in range(3):
        smoothed_pred[:,j,i] = signal.savgol_filter(np.squeeze(true_pred[:,j,i]), window_length=5, polyorder=3, deriv=0, delta=1.0, axis=- 1, mode='interp', cval=0.0) #window length 17 for 100 fps
        # smoothed_pred[:,j,i] = signal.medfilt(np.squeeze(smoothed_pred[:,j,i]), kernel_size=11)
        smoothed_pred[:,j,i] = signal.medfilt(np.squeeze(smoothed_pred[:,j,i]), kernel_size=1) # 5 for 100 fps

# print(smoothed_pred.shape)
smoothed_pred_file = os.path.join(os.path.dirname(com_file),
                                  'smoothed_prediction_AVG0.mat')
# smoothed_pred_file = 'smoothed_prediction_twd_withmetadata.mat'

mdict = {'pred':smoothed_pred,
        'data': pred_data['data'],
        # 'metadata': pred_data['metadata'],
        'p_max': pred_data['p_max'],
        'sampleID': pred_data['sampleID']}

sio.savemat(os.path.join(path, smoothed_pred_file), mdict)
# print('Smoothed prediction saved')


