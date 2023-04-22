# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:12:43 2020

@author: jaych
"""
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import database
import time
import scipy.io as sio

def merge(idxs):
    path = 'data/enum_train12/'
    xlist = list()
    ylist = list()
    radius = list()
    value = list()
    TPSF_ARRAY = np.zeros([301, 11, 11])
    lifetime = list()
    matfilename = path + 'tpsf' + str(idxs[0]) + '.mat'
    value_tmp = np.random.permutation(20)[0:1]/10 + 3
    value.append(value_tmp)
    # if idxs[0] <= (2400):
    #     radius.append(0.0025)
    # elif idxs[0] <= (4800):
    #     radius.append(0.0035)
    # else:
    #     radius.append(0.0045)

    with h5py.File(matfilename, 'r') as f:
        x = f['x_pos_target1'][()]
        y = f['y_pos_target1'][()]
        TPSF_ARRAY = TPSF_ARRAY + f['TPSF_array'][()][0:301, :, :] * value_tmp
        idxs3 = np.arange(30, 250, 1)
        datamax = np.amax(f['TPSF_array'][()][idxs3, :, :])
        lifetime.append(f['target1_inverselife'][()] / 1e9)
        radius.append(f['radius'][()])
        xlist.append(x)
        ylist.append(y)

    count = 1
    # 控制单目标data的数量 10%左右
    if np.random.rand(1) >= 0.15:
        for j in idxs[1:len(idxs)]:
            matfilename = path + 'tpsf' + str(j) + '.mat'
            if count >= 3:
                break
            # if j <= (2400):
            #     radius_tmp = 0.0025
            # elif j <= (4800):
            #     radius_tmp = 0.0035
            # else:
            #     radius_tmp = 0.0045

            with h5py.File(matfilename, 'r') as f:
                x = f['x_pos_target1'][()]
                y = f['y_pos_target1'][()]
                radius_tmp = f['radius'][()]
                dist = np.sqrt((np.array(xlist) - x) ** 2 + (np.array(ylist) - y) ** 2)
                dist = dist.reshape((len(dist)))
                if np.all(dist - radius_tmp - np.array(radius) >= 0.004):
                    if np.random.rand(1) <= 0.3 or count <= 1:
                        count = count + 1
                        xlist.append(x)
                        ylist.append(y)
                        radius.append(radius_tmp)
                        inverse_life_tmp = f['target1_inverselife'][()] / 1e9
                        # if inverse_life_tmp <= 1.3:
                        #     value_tmp_list = np.random.permutation(30)
                        #     value_tmp = value_tmp_list[0] / 10 + 3
                        # else:
                        # 产生数据，让深处目标更多出现产额更高的可能，深处的产额更高
                        if np.random.rand(1) <= 0.3:
                            value_tmp_list = np.random.permutation(20)
                            value_tmp = value_tmp_list[0] / 10 + 2
                        else:
                            value_tmp_list = np.random.permutation(20)
                            value_tmp = value_tmp_list[0] / 10 + 4
                        value.append(value_tmp)
                        lifetime.append(inverse_life_tmp)
                        TPSF_ARRAY = TPSF_ARRAY + f['TPSF_array'][()][0:301, :, :] * value_tmp

    distribution_true = np.zeros([5, 32, 72])
    lifetime_true = np.zeros([5, 32, 72])
    for i in range(0, len(xlist)):
        ypos = int(np.round(ylist[i] * 1000))
        xpos = int(np.round(xlist[i] * 1000))
        radius_mesh = radius[i] * 1000
        value_tmp = value[i]
        lifetime_tmp = lifetime[i]
        count = 0
        for j in range(ypos-5, ypos+6):
            for k in range(xpos-5, xpos+6):
                dist_tmp = (j-ypos)**2 + (k-xpos)**2
                if dist_tmp <= radius_mesh**2:
                    count = count+1
                    distribution_true[:, j, k] = value_tmp
                    lifetime_true[:, j, k] = lifetime_tmp
        # if radius[i] == 0.0025:
        #     distribution_true[:, ypos - 1:ypos + 2, xpos - 2] = value_tmp
        #     distribution_true[:, ypos - 2:ypos + 3, xpos - 1] = value_tmp
        #     distribution_true[:, ypos - 2:ypos + 3, xpos] = value_tmp
        #     distribution_true[:, ypos - 2:ypos + 3, xpos + 1] = value_tmp
        #     distribution_true[:, ypos - 1:ypos + 2, xpos + 2] = value_tmp
        #     lifetime_true[:, ypos - 1:ypos + 2, xpos - 2] = 1 / lifetime_tmp
        #     lifetime_true[:, ypos - 2:ypos + 3, xpos - 1] = 1 / lifetime_tmp
        #     lifetime_true[:, ypos - 2:ypos + 3, xpos] = 1 / lifetime_tmp
        #     lifetime_true[:, ypos - 2:ypos + 3, xpos + 1] = 1 / lifetime_tmp
        #     lifetime_true[:, ypos - 1:ypos + 2, xpos + 2] = 1 / lifetime_tmp
        # else:
        #     distribution_true[:, ypos - 1:ypos + 2, xpos - 3] = value_tmp
        #     distribution_true[:, ypos - 2:ypos + 3, xpos - 2] = value_tmp
        #     distribution_true[:, ypos - 3:ypos + 4, xpos - 1] = value_tmp
        #     distribution_true[:, ypos - 3:ypos + 4, xpos] = value_tmp
        #     distribution_true[:, ypos - 3:ypos + 4, xpos + 1] = value_tmp
        #     distribution_true[:, ypos - 2:ypos + 3, xpos + 2] = value_tmp
        #     distribution_true[:, ypos - 1:ypos + 2, xpos + 3] = value_tmp
        #     lifetime_true[:, ypos - 1:ypos + 2, xpos - 3] = 1 / lifetime_tmp
        #     lifetime_true[:, ypos - 2:ypos + 3, xpos - 2] = 1 / lifetime_tmp
        #     lifetime_true[:, ypos - 3:ypos + 4, xpos - 1] = 1 / lifetime_tmp
        #     lifetime_true[:, ypos - 3:ypos + 4, xpos] = 1 / lifetime_tmp
        #     lifetime_true[:, ypos - 3:ypos + 4, xpos + 1] = 1 / lifetime_tmp
        #     lifetime_true[:, ypos - 2:ypos + 3, xpos + 2] = 1 / lifetime_tmp
        #     lifetime_true[:, ypos - 1:ypos + 2, xpos + 3] = 1 / lifetime_tmp

    dt = 12.5 / 1024 * 2
    TPSF_ARRAY[TPSF_ARRAY < 0] = 0
    Tpsf_logtail = (np.sum(np.log(TPSF_ARRAY[180:200, :, :].reshape(20, 11 * 11) + 0.00000001), axis=0) - \
                    np.sum(np.log(TPSF_ARRAY[200:220, :, :].reshape(20, 11 * 11) + 0.00000001), axis=0)) / 20 / 20 / dt
    # print(idxs)
    idxs1 = [32, 40, 48, 56, 64, 72, 80, 112, 144, 176, 208]
    # idxs1 = [40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]
    # idxs1 = np.arange(40, 188, 12) + rand_dt2
    idxs2 = [40, 48, 56, 64, 72, 80, 112, 144, 176, 208, 240]
    # idxs2 = [48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
    # # idxs2 = np.arange(40, 212, 12) + rand_dt
    Tpsf_true = TPSF_ARRAY[idxs2, :, :].reshape(11, 11 * 11) - TPSF_ARRAY[idxs1, :, :].reshape(11, 11 * 11)
    # Tpsf_true = TPSF_ARRAY[40:204:16, :, :].reshape(11, 11*11)
    Tpsf_timeless = np.sum(TPSF_ARRAY[:, :, :], axis=0).reshape(11 * 11)
    for i in range(0, 11):
        for j in range(0, 11):
            if i == j:
                continue
            rand_dt = np.random.permutation(7)[0:1] - 3
            time_rand = np.arange(3, 296, 1) + rand_dt
            TPSF_ARRAY[3:296, i, j] = TPSF_ARRAY[time_rand, i, j]

    # plt.figure()
    # plt.subplot(221)
    # plt.plot(TPSF_ARRAY[:,2,1])
    TPSF_ARRAY = TPSF_ARRAY + np.random.normal(0, 6e4, TPSF_ARRAY.shape)
    TPSF_ARRAY[TPSF_ARRAY <= 0] = 0
    TPSF_ARRAY = TPSF_ARRAY * np.random.normal(1, 0.05, [11, 11])
    # plt.subplot(222)
    # plt.plot(TPSF_ARRAY[:, 2, 1])
    TPSF_ARRAY = np.multiply(TPSF_ARRAY, np.random.normal(1, 0.01, TPSF_ARRAY.shape))
    # plt.subplot(223)
    # plt.plot(TPSF_ARRAY[:, 2, 1])
    idxs3 = np.arange(30, 250, 1)
    # datamax = np.amax(TPSF_ARRAY[idxs3, :, :])
    # distri_max = np.amax(distribution_true)
    distri_max = 4
    TPSF_ARRAY = TPSF_ARRAY / datamax / 4
    distribution_true = distribution_true / distri_max
    # TPSF_ARRAY = TPSF_ARRAY + np.random.normal(0, 0.0001, TPSF_ARRAY.shape)

    idx_meas2 = list()
    weight_logtail = np.zeros([54], dtype=np.float32)
    count = 0
    for i in range(0, 11):
        weightbalance = 1
        for j in range(max(i - 3, 0), min(i + 4, 11)):
            if i == j:
                continue
            idx_meas2.append(i * 11 + j)
            maxtmp = np.amax(TPSF_ARRAY[:, i, j])
            if maxtmp >= 5e-4:
                weight_logtail[count] = 1
                count = count + 1
            else:
                weight_logtail[count] = 1
                count = count + 1

    # start2 = time.perf_counter()
    Tpsf_logtail2 = Tpsf_logtail[idx_meas2]
    tmp = np.matmul(model_matrix, distribution_true[0, :, :].reshape(1 * 32 * 72))
    logW = (np.log(tmp[:, 0]) - np.log(tmp[:, 1])) / 80 / dt
    tmp1 = np.exp(Tpsf_logtail2 * 150 * dt)
    tmp2 = -np.exp(Tpsf_logtail2 * 150 * dt - logW * 300 * dt) + np.exp(-logW * 150 * dt)
    Tpsf_logtail2 = -np.log((1 + np.sqrt(1 - 4 * tmp1 * tmp2)) / 2 / tmp1) / 150 / dt
    # end2 = time.perf_counter()
    # print("Time elapsed:", end2 - start2)

    if np.random.rand(1) <= 0.01:
        TPSF_ARRAY = TPSF_ARRAY * 0
        distribution_true = distribution_true * 0
        lifetime_true = lifetime_true * 0
        Tpsf_logtail2 = Tpsf_logtail2 * 0
        Tpsf_true = Tpsf_true * 0
        Tpsf_timeless = Tpsf_timeless * 0

    idx_meas = list()
    for i in range(0, 11):
        for j in range(i + 1, min(i + 7, 11)):
            idx_meas.append(i * 11 + j)

    # end = time.perf_counter()
    # print("Time elapsed:", end - start)
    return TPSF_ARRAY[idxs3, :, :], Tpsf_true[:, idx_meas], Tpsf_timeless[idx_meas], Tpsf_logtail2, \
           distribution_true, lifetime_true, weight_logtail


def get_batch(num_sam, idx_of_dataset):
    counter = idx_of_dataset-1
    depth_idx = 0
    horizon_idx = 0
    idxs = np.random.permutation(num_sam)[0:1] + 1
    start = time.perf_counter()
    TPSF_ARRAY_batch = np.zeros([220, 11, 11, 4])
    while 1 and counter % 4 == 0:  # 控制4组数据中最浅处目标是一样的位置
        idxs = np.random.permutation(num_sam)[0:1] + 1
        # if 4800 >= idxs[0] >= 2401:
        #     idxs[0] = idxs[0] - 2400
        radius_idx = (idxs[0] - 1) % (2400)
        depth_idx = np.floor((radius_idx % (100)) / 10)
        horizon_idx = np.floor(radius_idx / (100))
        if abs(horizon_idx - 11.5) >= 17 - depth_idx or abs(horizon_idx - 11.5) >= 11:
            continue
        if 7 >= depth_idx >= 0:
            break
    if counter % 1 == 0:
        idxs_temp = np.random.permutation(num_sam) + 1
        idxs_deeper = list()
        for i in idxs_temp:
            # if 4800 >= i >= 2401:
            #     i = i - 2400
            radius_idx2 = (i - 1) % (2400)
            depth_idx2 = np.floor((radius_idx2 % (100)) / 10)
            horizon_idx2 = np.floor(radius_idx2 / (100))
            randflag = np.random.rand(1)
            if (depth_idx + 7 >= depth_idx2 >= depth_idx and 5 <= np.abs(horizon_idx - horizon_idx2) <= 12
                and randflag < 0.25) \
                    or (depth_idx + 6 >= depth_idx2 >= depth_idx and 2 <= np.abs(horizon_idx - horizon_idx2) <= 5
                        and randflag >= 0.25):
                # or (depth_idx + 4 >= depth_idx2 >= depth_idx and np.abs(horizon_idx - horizon_idx2) <= 4):
                if abs(horizon_idx2 - 11.5) >= 17 - depth_idx2 or abs(horizon_idx2 - 11.5) >= 8:
                    continue
                if 1.9 * np.abs(horizon_idx - horizon_idx2) < np.abs(depth_idx - depth_idx2):
                    continue
                if (horizon_idx - horizon_idx2) ** 2 + (depth_idx - depth_idx2) ** 2 <= 9:
                    continue
                idxs_deeper.append(i)
                if len(idxs_deeper) == 4:
                    break

    idxs_temp = list()
    idxs_temp.append(idxs[0])
    idxs_temp.extend(idxs_deeper)
    idxs_temp = np.array(idxs_temp)
    # idxs_temp = np.array([701, 4917, 2150, 5184, 5444])
    TPSF_ARRAY, tpsftrue, tpsftimeless, tpsf_logtail, distribution_true, lifetime_true, weight_logtail = merge(idxs_temp)

    # tmp = TPSF_ARRAY
    # tmp[tmp < 1e-6] = 1e-6
    # TPSF_ARRAY_batch[:, :, :, 3] = np.log(tmp[:, :, :])
    # TPSF_ARRAY_batch[:, :, :, 0] = TPSF_ARRAY
    # valuemax = np.amax(TPSF_ARRAY)
    # TPSF_ARRAY = TPSF_ARRAY * 20
    # TPSF_ARRAY[TPSF_ARRAY >= valuemax] = valuemax
    # TPSF_ARRAY_batch[:, :, :, 1] = TPSF_ARRAY
    # TPSF_ARRAY = TPSF_ARRAY * 20
    # TPSF_ARRAY[TPSF_ARRAY >= valuemax] = valuemax
    # TPSF_ARRAY_batch[:, :, :, 2] = TPSF_ARRAY

    return TPSF_ARRAY, tpsftrue, tpsftimeless, tpsf_logtail, weight_logtail, distribution_true, lifetime_true

datapath_model = 'data/model/'
model_matrix = database.get_model(datapath_model)
idx_meas2 = list()
for i in range(0, 11):
    for j in range(max(i - 3, 0), min(i + 4, 11)):
        if i == j:
            continue
        idx_meas2.append(i * 11 + j)
model_matrix = model_matrix.swapaxes(0, 2)[idx_meas2, 40:61:20, :].reshape(54, 2, 5, 32*72)
model_matrix = np.sum(model_matrix, axis=2)
num_sam = 4800
start = time.perf_counter()
for idx_of_dataset in range(50001, 501000):
    if idx_of_dataset % 500 == 0:
        print(idx_of_dataset)
        end = time.perf_counter()
        print("Time elapsed_total:", end - start)
        start = time.perf_counter()
    TPSF_ARRAY, tpsftrue, tpsftimeless, tpsf_logtail, weight_logtail, distribution_true, lifetime_true = get_batch(num_sam, idx_of_dataset)

    TPSF_ARRAY_batch = np.zeros([220, 11, 11, 4])
    tmp = TPSF_ARRAY
    tmp[tmp < 1e-6] = 1e-6
    TPSF_ARRAY_batch[:, :, :, 3] = np.log(tmp[:, :, :])
    TPSF_ARRAY_batch[:, :, :, 0] = TPSF_ARRAY
    valuemax = np.amax(TPSF_ARRAY)
    TPSF_ARRAY = TPSF_ARRAY * 20
    TPSF_ARRAY[TPSF_ARRAY >= valuemax] = valuemax
    TPSF_ARRAY_batch[:, :, :, 1] = TPSF_ARRAY
    TPSF_ARRAY = TPSF_ARRAY * 20
    TPSF_ARRAY[TPSF_ARRAY >= valuemax] = valuemax
    TPSF_ARRAY_batch[:, :, :, 2] = TPSF_ARRAY

    sio.savemat('/data/cjj/data/generated_sets9/' + str(idx_of_dataset) + '.mat',
                {'TPSF_ARRAY_batch': TPSF_ARRAY_batch,
                 'tpsftrue': tpsftrue,
                 'tpsftimeless': tpsftimeless,
                 'tpsf_logtail': tpsf_logtail,
                 'weight_logtail': weight_logtail,
                 'distribution_true': distribution_true,
                 'lifetime_true': lifetime_true,
                 })
    # matfilename = '/data/cjj/reflective_FMT_reconstruction_tf2/data/generated_sets/' \
    #               + str(int(200)) + '.mat'
    # f = sio.loadmat(matfilename)
    # print(f['weight_logtail'][()])