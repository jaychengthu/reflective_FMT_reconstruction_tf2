# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:12:43 2020

@author: jaych
"""
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
import time
# matlab 文件名 'H:\\FILE\\phdwork\\PhDwork\\reflected-FMT\\data\\simulation_python\\tpsf'+str(i)+'.mat'

def get_test_data(path):
    realdatalist = os.listdir(path)
    realdatalist.sort()
    print(realdatalist)
    TPSF = np.zeros([len(realdatalist), 220, 11, 11, 4])
    Tpsf_true = np.zeros([len(realdatalist), 11, 45])
    Tpsf_timeless_true = np.zeros([len(realdatalist), 45])
    Tpsf_logtail = np.zeros([len(realdatalist), 54])
    weight_logtail = np.zeros([len(realdatalist), 54])
    for i in range(0, len(realdatalist)):
        matfilename = os.path.join(path, realdatalist[i])
        # print(realdatalist[i])
        with h5py.File(matfilename, 'r') as f:
            TPSF_ARRAY1 = f['TPSF_array'][()][:, :, :]
            # TPSF_ARRAY = np.multiply(TPSF_ARRAY, np.random.normal(1, 0.1, TPSF_ARRAY.shape))
            # TPSF_ARRAY1 = smooth_filter(TPSF_ARRAY1)
            TPSF_ARRAY1[TPSF_ARRAY1 < 0] = 0
            datamax = np.amax(TPSF_ARRAY1)
    #        print(datamax)
            TPSF_ARRAY1 = TPSF_ARRAY1 / datamax
            idxs1 = [32, 40, 48, 56, 64, 72, 80, 112, 144, 176, 208]
            # idxs1 = [32, 40, 48, 56, 64, 72, 80, 88, 112, 136, 160, 184]
            # idxs1 = [40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]
            idxs2 = [40, 48, 56, 64, 72, 80, 112, 144, 176, 208, 240]
            # idxs2 = [40, 48, 56, 64, 72, 80, 88, 112, 136, 160, 184, 208]
            # idxs2 = [48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
            Tpsf_true_tmp = TPSF_ARRAY1[idxs2, :, :].reshape(len(idxs2), 11 * 11) - TPSF_ARRAY1[idxs1, :, :].reshape(len(idxs1), 11 * 11)
            # Tpsf_true_tmp = TPSF_ARRAY1[40:204:16, :, :].reshape(11, 11 * 11)
            Tpsf_timeless = np.sum(TPSF_ARRAY1[0:260, :, :], axis=0).reshape(11 * 11)
            dt = 12.5 / 1024 * 2
            Tpsf_logtail_tmp = (np.mean(np.log(TPSF_ARRAY1[130:150, :, :].reshape(20, 11 * 11) + 0.000001), axis=0) - \
                                np.mean(np.log(TPSF_ARRAY1[150:170, :, :].reshape(20, 11 * 11) + 0.000001), axis=0)) \
                               / 20 / 1 / dt

            idx_meas = list()
            weightlogtail = list()
            for k in range(0, 11):
                for j in range(k + 1, min(k + 7, 11)):
                    idx_meas.append(k * 11 + j)
            idx_meas2 = list()
            for k in range(0, 11):
                weightbalance = 1
                for j in range(max(k - 3, 0), min(k + 4, 11)):
                    if k == j:
                        continue
                    idx_meas2.append(k * 11 + j)
                    if np.amax(TPSF_ARRAY1[:, k, j]) >= 1e-3:
                        weightlogtail.append(1)
                        # weightbalance = 1
                    else:
                        weightlogtail.append(0.5)

            Tpsf_true[i, :, :] = Tpsf_true_tmp[:, idx_meas]
            Tpsf_timeless_true[i, :] = Tpsf_timeless[idx_meas]
            # Tpsf_logtail_tmp = Tpsf_logtail_tmp[idx_meas]
            Tpsf_logtail[i, :] = Tpsf_logtail_tmp[idx_meas2]
            weight_logtail[i, :] = weightlogtail
            TPSF_ARRAY = TPSF_ARRAY1[30:250, :, :]
            datamax = np.amax(TPSF_ARRAY)
            #        print(datamax)
            TPSF_ARRAY = TPSF_ARRAY / datamax
            tmp = TPSF_ARRAY
            tmp[tmp < 1e-6] = 1e-6
            TPSF[i, :, :, :, 3] = np.log(tmp)
            # plt.figure()
            # plt.plot(np.log(TPSF_ARRAY[:, 4, 7]+0.001))
            TPSF[i, :, :, :, 0] = TPSF_ARRAY
            valuemax = np.amax(TPSF_ARRAY)
            TPSF_ARRAY = TPSF_ARRAY * 20
            TPSF_ARRAY[TPSF_ARRAY >= valuemax] = valuemax
            TPSF[i, :, :, :, 1] = TPSF_ARRAY
            valuemax = np.amax(TPSF_ARRAY)
            TPSF_ARRAY = TPSF_ARRAY * 20
            TPSF_ARRAY[TPSF_ARRAY >= valuemax] = valuemax
            TPSF[i, :, :, :, 2] = TPSF_ARRAY

    return TPSF, Tpsf_true, Tpsf_timeless_true, Tpsf_logtail, weight_logtail, realdatalist

def smooth_filter(x):
    y = x
    tl, _, _ = x.shape
    for i in range(tl):
        start_t = max(0, i-5)
        end_t = min(tl-1, i+5)
        y[i] = np.mean(x[start_t:end_t], axis=0)

    return y

def get_model(path):
    weightlist = os.listdir(path)
    matfilename = os.path.join(path, weightlist[0])
    with h5py.File(matfilename, 'r') as f:
        # dt = 12.5e-9/1024
        # dxyz = 2e-9
        # idx_meas = list()
        # for i in range(0, 11):
        #     for j in range(i+1, min(i+7, 11)):
        #         idx_meas.append(i*11+j)
        model_matrix = f['weight_matrix'][()][:, :, :, 0:301:4, :, :]
        model_matrix = model_matrix.reshape(5*32*72, 76, 11*11)
        # model_matrix = model_matrix[:, :, idx_meas]
        weightmax = np.amax(model_matrix)
        model_matrix = model_matrix / weightmax
        model_matrix = model_matrix.astype(np.float32)
        # tmp = model_matrix
        # print(tmp.shape)
        # for i in range(0, 80, 8):
        #     tmp2 = tmp[:, i, 6].reshape(5, 32, 72)
        #     plt.figure()
        #     plt.imshow(np.log(tmp2[2]+1e-9))
        #     plt.show()
    return model_matrix

# class database(object):
#     def __init__(self, database_path, num_samples):
#         self._index_in_epoch = 0
#         self.path = database_path
#         self.datalist = os.listdir(database_path)
#         self.num_sam = len(self.datalist)
# #        for i in range(num_samples):
# #            idxs = np.random.permutation(100)[0:5]+1
# #            for j in range(1,5):
# #                if np.random.rand(1)<=0.8:
# #                    idxs[j] = 0
# #            self.idxs_list.append(idxs)
#
#     def merge(self, idxs):
#         xlist = list()
#         ylist = list()
#         radius = list()
#         value = list()
#         value2 = list()
#         value3 = list()
#         value4 = list()
#         # TPSF_ARRAY_conbined = np.zeros([5, 201, 11, 11])
#         TPSF_ARRAY = np.zeros([201, 11, 11])
#         TPSF_ARRAY2 = np.zeros([3, 201, 11, 11])
#         valuemax_array = np.zeros([4, 1])
#         lifetime = list()
#         matfilename = self.path+'tpsf'+str(idxs[0])+'.mat'
#         value_tmp = np.random.permutation(60)[0:1] + 40
#         value_tmp2 = np.random.permutation(60)[0:1] + 40
#         value_tmp3 = np.random.permutation(60)[0:1] + 40
#         value_tmp4 = np.random.permutation(60)[0:1] + 40
#         value.append(value_tmp)
#         value2.append(value_tmp2)
#         value3.append(value_tmp3)
#         value4.append(value_tmp4)
#         if idxs[0] <= 2400:
#             radius.append(0.0025)
#         elif idxs[0] <= 4800:
#             radius.append(0.0035)
#         else:
#             radius.append(0.0045)
#
#         with h5py.File(matfilename, 'r') as f:
#             x = f['x_pos_target1'][()]
#             y = f['y_pos_target1'][()]
#             TPSF_ARRAY = TPSF_ARRAY + f['TPSF_array'][()][50:251, :, :] * value_tmp
#             TPSF_ARRAY2[0] = TPSF_ARRAY2[0] + f['TPSF_array'][()][50:251, :, :] * value_tmp2
#             TPSF_ARRAY2[1] = TPSF_ARRAY2[1] + f['TPSF_array'][()][50:251, :, :] * value_tmp3
#             TPSF_ARRAY2[2] = TPSF_ARRAY2[2] + f['TPSF_array'][()][50:251, :, :] * value_tmp4
#             lifetime.append(f['target1_life'][()]*1e9)
#             xlist.append(x)
#             ylist.append(y)
#
#         count = 1
#         for j in idxs[1:len(idxs)]:
#             if np.random.rand(1) <= 0.6:
#                 continue
#             matfilename = self.path+'tpsf'+str(j)+'.mat'
#             if j <= 2400:
#                 radius_tmp = 0.0025
#             elif j <= 4800:
#                 radius_tmp = 0.0035
#             else:
#                 radius_tmp = 0.0045
#
#             with h5py.File(matfilename, 'r') as f:
#                 x = f['x_pos_target1'][()]
#                 y = f['y_pos_target1'][()]
#                 dist = np.sqrt((np.array(xlist)-x)**2+(np.array(ylist)-y)**2)
#                 dist = dist.reshape((len(dist)))
#                 if np.all(dist - radius_tmp - np.array(radius) >= 0):
#                     if np.random.rand(1) <= 0.3 or count <= 1:
#                         count = count + 1
#                         xlist.append(x)
#                         ylist.append(y)
#                         radius.append(radius_tmp)
#                         value_tmp_list = np.random.permutation(60) + 40
#                         value_tmp = value_tmp_list[0]
#                         value.append(value_tmp)
#                         value_tmp2 = value_tmp_list[1]
#                         value2.append(value_tmp2)
#                         value_tmp3 = value_tmp_list[2]
#                         value3.append(value_tmp3)
#                         value_tmp4 = value_tmp_list[3]
#                         value4.append(value_tmp4)
#                         lifetime.append(f['target1_life'][()]*1e9)
#                         TPSF_ARRAY = TPSF_ARRAY + f['TPSF_array'][()][50:251, :, :] * value_tmp
#                         # TPSF_ARRAY2[0] = TPSF_ARRAY2[0] + f['TPSF_array'][()][50:251, :, :] * value_tmp2
#                         # TPSF_ARRAY2[1] = TPSF_ARRAY2[1] + f['TPSF_array'][()][50:251, :, :] * value_tmp3
#                         # TPSF_ARRAY2[2] = TPSF_ARRAY2[2] + f['TPSF_array'][()][50:251, :, :] * value_tmp4
#
#         distribution_true = np.zeros([1, 72, 32])
#         distribution_true2 = np.zeros([3, 72, 32])
#         lifetime_true = np.zeros([1, 72, 32])
#         for i in range(1, 73):
#             for j in range(1, 33):
#                 dist = np.sqrt((np.array(xlist)-i/1000)**2+(np.array(ylist)-j/1000)**2)
#                 dist = dist.reshape((len(dist)))
#                 if np.any(dist-np.array(radius) <= 0):
#                     value_idx = np.argmin(dist - np.array(radius))
#                     value_tmp = value[value_idx]
#                     value_tmp2 = value2[value_idx]
#                     value_tmp3 = value3[value_idx]
#                     value_tmp4 = value4[value_idx]
#                     lifetime_tmp = lifetime[value_idx]
#
#                     distribution_true[:, i, j] = value_tmp
#                     # distribution_true2[0, i, j] = value_tmp2
#                     # distribution_true2[1, i, j] = value_tmp3
#                     # distribution_true2[2, i, j] = value_tmp4
#                     lifetime_true[:, i, j] = 1/lifetime_tmp
#
#         datamax = np.amax(TPSF_ARRAY)
#         distri_max = np.amax(distribution_true)
#         TPSF_ARRAY = TPSF_ARRAY / datamax
#         valuemax_array[0] = datamax/1e9
#         distribution_true = distribution_true / distri_max
#
#         for i in range(0,3):
#             datamax2 = np.amax(TPSF_ARRAY2[i])
#             distri_max2 = np.amax(distribution_true2[i])
#             TPSF_ARRAY2[i] = TPSF_ARRAY2[i] / datamax2
#             # valuemax_array[i+1] = datamax2/1e9
#             # distribution_true2[i] = distribution_true2[i] / distri_max2
#
#         if np.random.rand(1) <= 0.04:
#             TPSF_ARRAY = TPSF_ARRAY * 0
#             # TPSF_ARRAY2 = TPSF_ARRAY2 * 0
#             distribution_true = distribution_true * 0
#             # distribution_true2 = distribution_true2 * 0
#             lifetime_true = lifetime_true * 0
#         # noise = np.random.normal(1, 0.1, TPSF_ARRAY.shape[1:3])
#         # noise_array = np.tile(noise, (201, 1, 1))
#         # TPSF_ARRAY = np.multiply(TPSF_ARRAY, noise_array)
#         TPSF_ARRAY = np.multiply(TPSF_ARRAY, np.random.normal(1, 0.1, TPSF_ARRAY.shape))
#         # TPSF_ARRAY2 = np.multiply(TPSF_ARRAY2, np.random.normal(1, 0.1, TPSF_ARRAY2.shape))
#         return TPSF_ARRAY, distribution_true, lifetime_true, TPSF_ARRAY2, distribution_true2, valuemax_array
#
#     def next_batch(self, batch_size):
#         TPSF_ARRAY_batch = np.zeros([batch_size, 201, 11, 11, 3])
#         value_array_batch = np.zeros([batch_size, 1])
#         distribution_true_batch = np.zeros([batch_size, 1, 72, 32])
#         lifetime_true_batch = np.zeros([batch_size, 1, 72, 32])
#         for i in range(0, batch_size, 4):
#             idxs = np.random.permutation(self.num_sam)[0:5] + 1
#             for j in range(0,4):
#                 TPSF_ARRAY, distribution_true, lifetime_true, _, _, value_array = self.merge(idxs[0:len(idxs) - j])
#
#                 TPSF_ARRAY_batch[i+j, :, :, :, 0] = TPSF_ARRAY
#                 valuemax = np.amax(TPSF_ARRAY)
#                 TPSF_ARRAY = TPSF_ARRAY * 10
#                 TPSF_ARRAY[TPSF_ARRAY >= valuemax] = valuemax
#                 TPSF_ARRAY_batch[i+j, :, :, :, 1] = TPSF_ARRAY
#                 TPSF_ARRAY = TPSF_ARRAY * 10
#                 TPSF_ARRAY[TPSF_ARRAY >= valuemax] = valuemax
#                 TPSF_ARRAY_batch[i+j, :, :, :, 2] = TPSF_ARRAY
#                 value_array_batch[i+j, :] = value_array[j]
#                 # TPSF_ARRAY[TPSF_ARRAY <= 1e-9] = 1e-9
#                 # TPSF_ARRAY_batch[i+j, :, :, :, 3] = np.log(TPSF_ARRAY)
#
#                 distribution_true_batch[i+j] = distribution_true
#                 lifetime_true_batch[i + j] = lifetime_true
#
#         return TPSF_ARRAY_batch, distribution_true_batch, lifetime_true_batch, value_array_batch
#
#     def get_special_batch(self, batch_size):
#         TPSF_ARRAY_batch = np.zeros([batch_size, 201, 11, 11, 3])
#         distribution_true_batch = np.zeros([batch_size, 5, 32, 72])
#         lifetime_true_batch = np.zeros([batch_size, 5, 32, 72])
#
#         for j in range(int(batch_size/10)):
#             idxs = np.random.permutation(self.num_sam)[0:1]+1
#             radius_idx = (idxs[0] - 1) % 2400
#             depth_idx = np.floor((radius_idx % 100) / 10)
#             # horizon_idx = np.floor(radius_idx / 70)
#             idxs_temp = np.random.permutation(self.num_sam) + 1
#             idxs_deeper = list()
#             for i in idxs_temp:
#                 radius_idx2 = (i - 1) % 2400
#                 depth_idx2 = np.floor((radius_idx2 % 100) / 10)
#                 # horizon_idx2 = np.floor(radius_idx2 / 70)
#                 if depth_idx2 >= depth_idx:
#                     idxs_deeper.append(i)
#                     if len(idxs_deeper) == 10 + 4:
#                         break
#             for i in range(10):
#                 idxs_temp = list()
#                 idxs_temp.append(idxs[0])
#                 idxs_temp.extend(idxs_deeper[i:i+4])
#                 idxs_temp = np.array(idxs_temp)
#                 TPSF_ARRAY, distribution_true, lifetime_true, _, _,_ = self.merge(idxs_temp)
#
#                 TPSF_ARRAY_batch[j * 10 + i, :, :, :, 0] = TPSF_ARRAY
#                 valuemax = np.amax(TPSF_ARRAY)
#                 TPSF_ARRAY = TPSF_ARRAY * 10
#                 TPSF_ARRAY[TPSF_ARRAY >= valuemax] = valuemax
#                 TPSF_ARRAY_batch[j * 10 + i, :, :, :, 1] = TPSF_ARRAY
#                 TPSF_ARRAY = TPSF_ARRAY * 10
#                 TPSF_ARRAY[TPSF_ARRAY >= valuemax] = valuemax
#                 TPSF_ARRAY_batch[j * 10 + i, :, :, :, 2] = TPSF_ARRAY
#                 # TPSF_ARRAY[TPSF_ARRAY <= 1e-9] = 1e-9
#                 # TPSF_ARRAY_batch[j * 10 + i, :, :, :, 3] = np.log(TPSF_ARRAY)
#
#                 distribution_true_batch[j * 10 + i] = distribution_true
#                 lifetime_true_batch[j * 10 + i] = lifetime_true
#
#         return TPSF_ARRAY_batch, distribution_true_batch, lifetime_true_batch
#
#     def get_test_batch(self, batch_size):
#         TPSF_ARRAY_batch = np.zeros([batch_size, 201, 11, 11, 3])
#         valuemax_array_batch = np.zeros([batch_size, 1])
#         distribution_true_batch = np.zeros([batch_size, 5, 32, 72])
#         lifetime_true_batch = np.zeros([batch_size, 5, 32, 72])
#
#         for j in range(int(batch_size / 8)):
#             while 1:
#                 idxs = np.random.permutation(self.num_sam)[0:1] + 1
#                 radius_idx = (idxs[0] - 1) % 2400
#                 depth_idx = np.floor((radius_idx % 100) / 10)
#                 horizon_idx = np.floor(radius_idx / 100)
#                 if 7 >= depth_idx >= 0:
#                     break
#             idxs_temp = np.random.permutation(self.num_sam) + 1
#             idxs_deeper = list()
#             for i in idxs_temp:
#                 radius_idx2 = (i - 1) % 2400
#                 depth_idx2 = np.floor((radius_idx2 % 100) / 10)
#                 horizon_idx2 = np.floor(radius_idx2 / 100)
#                 if depth_idx + 6 >= depth_idx2 >= depth_idx and 2 <= np.abs(horizon_idx - horizon_idx2) <= 10:
#                     idxs_deeper.append(i)
#                     if len(idxs_deeper) == 16 + 4:
#                         break
#             for i in range(8):
#                 idxs_temp = list()
#                 idxs_temp.append(idxs[0])
#                 idxs_temp.extend(idxs_deeper[i*2:i*2 + 4])
#                 idxs_temp = np.array(idxs_temp)
#                 TPSF_ARRAY, distribution_true, lifetime_true, TPSF_ARRAY2, distribution_true2, valuemax_array = self.merge(idxs_temp)
#
#                 TPSF_ARRAY_batch[j * 8 + i * 1, :, :, :, 0] = TPSF_ARRAY
#                 valuemax = np.amax(TPSF_ARRAY)
#                 TPSF_ARRAY = TPSF_ARRAY * 10
#                 TPSF_ARRAY[TPSF_ARRAY >= valuemax] = valuemax
#                 TPSF_ARRAY_batch[j * 8 + i * 1, :, :, :, 1] = TPSF_ARRAY
#                 TPSF_ARRAY = TPSF_ARRAY * 10
#                 TPSF_ARRAY[TPSF_ARRAY >= valuemax] = valuemax
#                 TPSF_ARRAY_batch[j * 8 + i * 1, :, :, :, 2] = TPSF_ARRAY
#                 valuemax_array_batch[j * 8 + i * 1, :] = valuemax_array[0]
#
#                 distribution_true_batch[j * 8 + i * 1] = distribution_true
#                 lifetime_true_batch[j * 8 + i * 1] = lifetime_true
#
#                 # for k in range(0, 3):
#                 #     TPSF_ARRAY2_tmp = TPSF_ARRAY2[k]
#                 #     TPSF_ARRAY_batch[j * 16 + i * 4 + 1 + k, :, :, :, 0] = TPSF_ARRAY2_tmp
#                 #     valuemax = np.amax(TPSF_ARRAY2_tmp)
#                 #     TPSF_ARRAY2_tmp = TPSF_ARRAY2_tmp * 10
#                 #     TPSF_ARRAY2_tmp[TPSF_ARRAY2_tmp >= valuemax] = valuemax
#                 #     TPSF_ARRAY_batch[j * 16 + i * 2 + 1 + k, :, :, :, 1] = TPSF_ARRAY2_tmp
#                 #     TPSF_ARRAY2_tmp = TPSF_ARRAY2_tmp * 10
#                 #     TPSF_ARRAY2_tmp[TPSF_ARRAY2_tmp >= valuemax] = valuemax
#                 #     TPSF_ARRAY_batch[j * 16 + i * 4 + 1 + k, :, :, :, 2] = TPSF_ARRAY2_tmp
#                 #     valuemax_array_batch[j * 16 + i * 4 + 1 + k, :] = valuemax_array[k+1]
#                 #     distribution_true_batch[j * 16 + i * 4 + 1 + k] = distribution_true2[k]
#                 #     lifetime_true_batch[j * 16 + i * 4 + 1 + k] = lifetime_true
#                 # TPSF_ARRAY[TPSF_ARRAY <= 1e-9] = 1e-9
#                 # TPSF_ARRAY_batch[j * 5 + i, :, :, :, 3] = np.log(TPSF_ARRAY)
#
#         return TPSF_ARRAY_batch, distribution_true_batch, lifetime_true_batch, valuemax_array_batch
