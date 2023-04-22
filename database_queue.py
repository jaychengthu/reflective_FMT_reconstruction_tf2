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
import tensorflow.compat.v1 as tf
import scipy.io as sio


class database_queue(object):
    def __init__(self, database_path, batch_size):
        self.path = database_path
        self.datasetnum = 500000
        self.datapath_model = 'data/model/'
        model_matrix = database.get_model(self.datapath_model)
        self.idx_meas = list()
        for i in range(0, 11):
            for j in range(i + 1, min(i + 7, 11)):
                self.idx_meas.append(i * 11 + j)
        self.idx_meas2 = list()
        for i in range(0, 11):
            for j in range(max(i - 3, 0), min(i + 4, 11)):
                if i == j:
                    continue
                self.idx_meas2.append(i * 11 + j)
        self.radius1x = list()
        self.radius1y = list()
        for i in range(-2, 3):
            for j in range(-2, 3):
                if i ** 2 + j ** 2 <= 6:
                    self.radius1x.append(i)
                    self.radius1y.append(j)
        self.radius2x = list()
        self.radius2y = list()
        for i in range(-3, 4):
            for j in range(-3, 4):
                if i ** 2 + j ** 2 <= 12:
                    self.radius2x.append(i)
                    self.radius2y.append(j)
        model_matrix = model_matrix.swapaxes(0, 2)[self.idx_meas2, 40:61:20, :].reshape(54, 2, 5, 32 * 72)
        self.model_matrix = np.sum(model_matrix, axis=2)
        self.datalist = os.listdir(database_path)
        self.num_sam = 4800  # len(self.datalist)
        self.dataset = tf.data.Dataset.from_generator(self.get_batch, output_types=(tf.float32, tf.float32, tf.float32,
                                                                                    tf.float32, tf.float32, tf.float32,
                                                                                    tf.float32))
        # self.dataset = tf.data.Dataset.from_generator(self.get_batch, output_types=(tf.float32, tf.float32,
        #                                                                             tf.float32, tf.float32)). \
        #     map(self.add_noise, num_parallel_calls=8)
        self.dataset = self.dataset.prefetch(tf.data.experimental.AUTOTUNE).batch(batch_size)  #
        self.iterator = self.dataset.make_one_shot_iterator()

    def add_noise(self, TPSF_ARRAY_batch, TPSF_ARRAY, distribution_true, lifetime_true):

        dt = 12.5 / 1024 * 2
        Tpsf_logtail = (tf.reduce_sum(tf.log(tf.reshape(TPSF_ARRAY[180:200, :, :], [20, 11 * 11]) + 0.00000001),
                                      axis=0) - \
                        tf.reduce_sum(tf.log(tf.reshape(TPSF_ARRAY[200:220, :, :], [20, 11 * 11]) + 0.00000001),
                                      axis=0)) / 20 / 20 / dt
        # print(idxs)
        idxs1 = [32, 40, 48, 56, 64, 72, 80, 112, 144, 176, 208]
        idxs2 = [40, 48, 56, 64, 72, 80, 112, 144, 176, 208, 240]
        # idxs1 = [40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]
        # idxs1 = np.arange(40, 188, 12) + rand_dt2
        idxs2 = [40, 48, 56, 64, 72, 80, 112, 144, 176, 208, 240]
        # idxs2 = [48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
        # # idxs2 = np.arange(40, 212, 12) + rand_dt
        Tpsf_true = tf.reshape(tf.gather(TPSF_ARRAY, axis=0, indices=tf.constant(np.array(idxs2))), [11, 11 * 11]) \
                    - tf.reshape(tf.gather(TPSF_ARRAY, axis=0, indices=tf.constant(np.array(idxs1))), [11, 11 * 11])
        # Tpsf_true = TPSF_ARRAY[40:204:16, :, :].reshape(11, 11*11)
        Tpsf_timeless = tf.reshape(tf.reduce_sum(TPSF_ARRAY, axis=0), [11 * 11])
        for i in range(0, 11):
            for j in range(0, 11):
                if i == j:
                    continue
                rand_dt = np.random.permutation(7)[0:1] - 3
                time_rand = np.arange(3, 296, 1) + rand_dt
                TPSF_ARRAY[3:296, i, j] = tf.gather(TPSF_ARRAY, axis=0, indices=tf.constant(time_rand))

        # plt.figure()
        # plt.subplot(221)
        # plt.plot(TPSF_ARRAY[:,2,1])
        TPSF_ARRAY = TPSF_ARRAY + np.random.normal(0, 6e4, TPSF_ARRAY.shape)
        TPSF_ARRAY = tf.where(TPSF_ARRAY < 0, tf.zeros_like(t), TPSF_ARRAY)
        TPSF_ARRAY = TPSF_ARRAY * np.random.normal(1, 0.05, [11, 11])
        # plt.subplot(222)
        # plt.plot(TPSF_ARRAY[:, 2, 1])
        TPSF_ARRAY = tf.multiply(TPSF_ARRAY, np.random.normal(1, 0.01, TPSF_ARRAY.shape))
        # plt.subplot(223)
        # plt.plot(TPSF_ARRAY[:, 2, 1])
        idxs3 = np.arange(30, 250, 1)
        datamax = tf.max(tf.gather(TPSF_ARRAY, axis=0, indices=tf.constant(idxs3)))
        distri_max = tf.max(distribution_true)
        TPSF_ARRAY_batch = TPSF_ARRAY_batch / datamax
        distribution_true = distribution_true / distri_max

        Tpsf_logtail2 = tf.gather(Tpsf_logtail, indices=np.array(self.idx_meas2))
        tmp = tf.matmul(self.model_matrix, tf.reshape(distribution_true[2, :, :], [1 * 32 * 72]))
        logW = (tf.log(tmp[:, 0]) - tf.log(tmp[:, 1])) / 80 / dt
        tmp1 = tf.exp(Tpsf_logtail2 * 150 * dt)
        tmp2 = -tf.exp(Tpsf_logtail2 * 150 * dt - logW * 300 * dt) + tf.exp(-logW * 150 * dt)
        Tpsf_logtail2 = -tf.log((1 + np.sqrt(1 - 4 * tmp1 * tmp2)) / 2 / tmp1) / 150 / dt

        if np.random.rand(1) <= 0.01:
            TPSF_ARRAY_batch = TPSF_ARRAY_batch * 0
            distribution_true = distribution_true * 0
            lifetime_true = lifetime_true * 0
            Tpsf_logtail2 = Tpsf_logtail2 * 0
            Tpsf_true = Tpsf_true * 0
            Tpsf_timeless = Tpsf_timeless * 0

        weight_logtail = tf.constant(np.ones([54], dtype=np.float32), dtype=tf.float32)

        return TPSF_ARRAY_batch, tf.gather(Tpsf_true, axis=1, indices=tf.constant(np.array(self.idx_meas))), \
               tf.gather(Tpsf_timeless, axis=1, indices=tf.constant(np.array(self.idx_meas))), Tpsf_logtail2, \
               weight_logtail, distribution_true, lifetime_true

    def merge(self, idxs):
        start = time.perf_counter()

        xlist = list()
        ylist = list()
        radius = list()
        value = list()
        TPSF_ARRAY = np.zeros([301, 11, 11])
        # tpsftrue = np.zeros([301, 11, 11])
        lifetime = list()
        matfilename = self.path + 'tpsf' + str(idxs[0]) + '.mat'
        value_tmp = 4  # np.random.permutation(50)[0:1] + 50
        value.append(value_tmp)
        if idxs[0] <= (2400):
            radius.append(0.0025)
        elif idxs[0] <= (4800):
            radius.append(0.0035)
        else:
            radius.append(0.0045)

        with h5py.File(matfilename, 'r') as f:
            x = f['x_pos_target1'][()]
            y = f['y_pos_target1'][()]
            TPSF_ARRAY = TPSF_ARRAY + f['TPSF_array'][()][0:301, :, :] * value_tmp
            lifetime.append(f['target1_life'][()] * 1e9)
            xlist.append(x)
            ylist.append(y)
            # dist_min = 0
            # if y >= 0.018:
            #     dist_min = 0.002

        count = 1
        for j in idxs[1:len(idxs)]:
            if np.random.rand(1) <= 0.6:
                continue
            if j <= (2400):
                radius_tmp = 0.0025
            elif j <= (4800):
                radius_tmp = 0.0035
            else:
                radius_tmp = 0.0045
            radius_idx = (j - 1) % (2400)
            depth_idx = np.floor((radius_idx % (100)) / 10)
            y = depth_idx * 0.002 + 0.005
            horizon_idx = np.floor(radius_idx / (100))
            x = horizon_idx * 0.002 + 0.012
            dist = np.sqrt((np.array(xlist) - x) ** 2 + (np.array(ylist) - y) ** 2)
            dist = dist.reshape((len(dist)))
            if np.all(dist - radius_tmp - np.array(radius) >= 0.001):
                if np.random.rand(1) <= 0.3 or count <= 1:
                    matfilename = self.path + 'tpsf' + str(j) + '.mat'
                    with h5py.File(matfilename, 'r') as f:
                        x = f['x_pos_target1'][()]
                        y = f['y_pos_target1'][()]
                        count = count + 1
                        xlist.append(x)
                        ylist.append(y)
                        radius.append(radius_tmp)
                        lifetmp = f['target1_life'][()] * 1e9
                        if lifetmp >= 1.3:
                            value_tmp_list = np.random.permutation(30)
                            value_tmp = value_tmp_list[0] / 10 + 3
                        else:
                            value_tmp_list = np.random.permutation(40)
                            value_tmp = value_tmp_list[0] / 10 + 2

                        value.append(value_tmp)
                        lifetime.append(lifetmp)
                        TPSF_ARRAY = TPSF_ARRAY + f['TPSF_array'][()][0:301, :, :] * value_tmp

        distribution_true = np.zeros([5, 32, 72])
        lifetime_true = np.zeros([5, 32, 72])
        for i in range(0, len(xlist)):
            ypos = int(np.round(ylist[i] * 1000))
            xpos = int(np.round(xlist[i] * 1000))
            value_tmp = value[i]
            lifetime_tmp = lifetime[i]
            if radius[i] == 0.0025:
                distribution_true[:, [x + ypos for x in self.radius1y], [x + xpos for x in self.radius1x]] = value_tmp
                lifetime_true[:, [x + ypos for x in self.radius1y],
                [x + xpos for x in self.radius1x]] = 1 / lifetime_tmp
                # distribution_true[:, ypos - 1:ypos + 2, xpos - 2] = value_tmp
                # distribution_true[:, ypos - 2:ypos + 3, xpos - 1] = value_tmp
                # distribution_true[:, ypos - 2:ypos + 3, xpos] = value_tmp
                # distribution_true[:, ypos - 2:ypos + 3, xpos + 1] = value_tmp
                # distribution_true[:, ypos - 1:ypos + 2, xpos + 2] = value_tmp
                # lifetime_true[:, ypos - 1:ypos + 2, xpos - 2] = 1 / lifetime_tmp
                # lifetime_true[:, ypos - 2:ypos + 3, xpos - 1] = 1 / lifetime_tmp
                # lifetime_true[:, ypos - 2:ypos + 3, xpos] = 1 / lifetime_tmp
                # lifetime_true[:, ypos - 2:ypos + 3, xpos + 1] = 1 / lifetime_tmp
                # lifetime_true[:, ypos - 1:ypos + 2, xpos + 2] = 1 / lifetime_tmp
            else:
                distribution_true[:, [x + ypos for x in self.radius2y], [x + xpos for x in self.radius2x]] = value_tmp
                lifetime_true[:, [x + ypos for x in self.radius2y], [x + xpos for x in self.radius2x]] = 1 / lifetime_tmp
                # distribution_true[:, ypos - 1:ypos + 2, xpos - 3] = value_tmp
                # distribution_true[:, ypos - 2:ypos + 3, xpos - 2] = value_tmp
                # distribution_true[:, ypos - 3:ypos + 4, xpos - 1] = value_tmp
                # distribution_true[:, ypos - 3:ypos + 4, xpos] = value_tmp
                # distribution_true[:, ypos - 3:ypos + 4, xpos + 1] = value_tmp
                # distribution_true[:, ypos - 2:ypos + 3, xpos + 2] = value_tmp
                # distribution_true[:, ypos - 1:ypos + 2, xpos + 3] = value_tmp
                # lifetime_true[:, ypos - 1:ypos + 2, xpos - 3] = 1 / lifetime_tmp
                # lifetime_true[:, ypos - 2:ypos + 3, xpos - 2] = 1 / lifetime_tmp
                # lifetime_true[:, ypos - 3:ypos + 4, xpos - 1] = 1 / lifetime_tmp
                # lifetime_true[:, ypos - 3:ypos + 4, xpos] = 1 / lifetime_tmp
                # lifetime_true[:, ypos - 3:ypos + 4, xpos + 1] = 1 / lifetime_tmp
                # lifetime_true[:, ypos - 2:ypos + 3, xpos + 2] = 1 / lifetime_tmp
                # lifetime_true[:, ypos - 1:ypos + 2, xpos + 3] = 1 / lifetime_tmp
        # for i in range(1, 32):
        #     for j in range(1, 72):
        #         dist = np.sqrt((np.array(xlist) - j / 1000) ** 2 + (np.array(ylist) - i / 1000) ** 2)
        #         dist = dist.reshape((len(dist)))
        #         if np.any(dist - np.array(radius) <= 0):
        #             value_idx = np.argmin(dist - np.array(radius))
        #             value_tmp = value[value_idx]
        #             lifetime_tmp = lifetime[value_idx]
        #
        #             distribution_true[:, i, j] = value_tmp #/ lifetime_tmp
        #             lifetime_true[:, i, j] = 1 / lifetime_tmp
        #             print(i , j)

        dt = 12.5 / 1024 * 2
        Tpsf_logtail = (np.sum(np.log(TPSF_ARRAY[180:200, :, :].reshape(20, 11 * 11) + 0.00000001), axis=0) - \
                        np.sum(np.log(TPSF_ARRAY[200:220, :, :].reshape(20, 11 * 11) + 0.00000001), axis=0)) / 20 / 20 / dt
        # print(idxs)
        idxs1 = [32, 40, 48, 56, 64, 72, 80, 112, 144, 176, 208]
        # idxs1 = [32, 40, 48, 56, 64, 72, 80, 88, 112, 136, 160, 184]
        # idxs1 = [40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]
        # idxs1 = np.arange(40, 188, 12) + rand_dt2
        idxs2 = [40, 48, 56, 64, 72, 80, 112, 144, 176, 208, 240]
        # idxs2 = [40, 48, 56, 64, 72, 80, 88, 112, 136, 160, 184, 208]
        # idxs2 = [48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
        # # idxs2 = np.arange(40, 212, 12) + rand_dt
        Tpsf_true = TPSF_ARRAY[idxs2, :, :].reshape(len(idxs2), 11 * 11) - TPSF_ARRAY[idxs1, :, :].reshape(len(idxs1), 11 * 11)
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
        datamax = np.amax(TPSF_ARRAY[idxs3, :, :])
        distri_max = np.amax(distribution_true)
        TPSF_ARRAY = TPSF_ARRAY / datamax
        distribution_true = distribution_true / distri_max
        TPSF_ARRAY = TPSF_ARRAY + np.random.normal(0, 0.0001, TPSF_ARRAY.shape)

        # idx_meas2 = list()
        weight_logtail = np.ones([54], dtype=np.float32)
        # count = 0
        # for i in range(0, 11):
        #     weightbalance = 1
        #     for j in range(max(i - 3, 0), min(i + 4, 11)):
        #         if i == j:
        #             continue
        #         idx_meas2.append(i * 11 + j)
        #         maxtmp = np.amax(TPSF_ARRAY[:, i, j])
        #         if maxtmp >= 5e-4:
        #             weight_logtail[count] = 1
        #             count = count + 1
        #         else:
        #             weight_logtail[count] = 1
        #             count = count + 1

        # start2 = time.perf_counter()
        Tpsf_logtail2 = Tpsf_logtail[self.idx_meas2]
        tmp = np.matmul(self.model_matrix, distribution_true[2, :, :].reshape(1 * 32 * 72))
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

        return TPSF_ARRAY[idxs3, :, :], Tpsf_true[:, self.idx_meas], Tpsf_timeless[self.idx_meas], Tpsf_logtail2, \
               distribution_true, lifetime_true, weight_logtail
        # return TPSF_ARRAY, distribution_true, lifetime_true

    def get_batch(self):
        counter = 0
        idxs = np.random.permutation(self.num_sam)[0:1] + 1
        dataset_idxs = np.random.permutation(self.datasetnum) + 1
        while True:
            # TPSF_ARRAY_batch = np.zeros([220, 11, 11, 4])
            # while 1 and counter % 4 == 0:  # 控制4组数据中最浅处目标是一样的位置
            #     idxs = np.random.permutation(self.num_sam)[0:1] + 1
            #     # if 4800 >= idxs[0] >= 2401:
            #     #     idxs[0] = idxs[0] - 2400
            #     radius_idx = (idxs[0] - 1) % (2400)
            #     depth_idx = np.floor((radius_idx % (100)) / 10)
            #     horizon_idx = np.floor(radius_idx / (100))
            #     if abs(horizon_idx - 11.5) >= 17 - depth_idx or abs(horizon_idx - 11.5) >= 11:
            #         continue
            #     if 7 >= depth_idx >= 0:
            #         break
            # if counter % 1 == 0:
            #     idxs_temp = np.random.permutation(self.num_sam) + 1
            #     idxs_deeper = list()
            #     for i in idxs_temp:
            #         # if 4800 >= i >= 2401:
            #         #     i = i - 2400
            #         radius_idx2 = (i - 1) % (2400)
            #         depth_idx2 = np.floor((radius_idx2 % (100)) / 10)
            #         horizon_idx2 = np.floor(radius_idx2 / (100))
            #         randflag = np.random.rand(1)
            #         if (depth_idx + 7 >= depth_idx2 >= depth_idx and 5 <= np.abs(horizon_idx - horizon_idx2) <= 12
            #             and randflag < 0.5) \
            #                 or (
            #                 depth_idx + 6 >= depth_idx2 >= depth_idx and 2 <= np.abs(horizon_idx - horizon_idx2) <= 5
            #                 and randflag >= 0.5):
            #             # or (depth_idx + 4 >= depth_idx2 >= depth_idx and np.abs(horizon_idx - horizon_idx2) <= 4):
            #             if abs(horizon_idx2 - 11.5) >= 17 - depth_idx2 or abs(horizon_idx2 - 11.5) >= 8:
            #                 continue
            #             # if horizon_idx2 <= 1 and depth_idx2 >= 7:
            #             #     continue
            #             idxs_deeper.append(i)
            #             if len(idxs_deeper) == 4:
            #                 break
            #
            # idxs_temp = list()
            # idxs_temp.append(idxs[0])
            # idxs_temp.extend(idxs_deeper)
            # idxs_temp = np.array(idxs_temp)
            # # idxs_temp = np.array([701, 4917, 2150, 5184, 5444])
            # TPSF_ARRAY, tpsftrue, tpsftimeless, tpsf_logtail, distribution_true, lifetime_true, weight_logtail = self.merge(idxs_temp)
            #
            # # TPSF_ARRAY, distribution_true, lifetime_true = self.merge(idxs_temp)
            #
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
            #
            # yield TPSF_ARRAY_batch, tpsftrue, tpsftimeless, tpsf_logtail, weight_logtail, distribution_true, lifetime_true
            # # yield TPSF_ARRAY_batch, TPSF_ARRAY, distribution_true, lifetime_true
            # counter = counter + 1


            if counter >= self.datasetnum:
                counter = 0
                dataset_idxs = np.random.permutation(self.datasetnum) + 1
            # last version use generated_sets3
            matfilename = '/data/cjj/data/generated_sets8/' \
                          + str(int(dataset_idxs[counter])) + '.mat'
            # print(counter)
            f = sio.loadmat(matfilename)
            # TPSF_ARRAY = f['TPSF_ARRAY'][()]
            tpsftrue = f['tpsftrue'][()]
            tpsftimeless = f['tpsftimeless'][()].reshape(45)
            tpsf_logtail = f['tpsf_logtail'][()].reshape(54)
            weight_logtail = f['weight_logtail'][()].reshape(54)
            # distribution_true_slice = f['distribution_true'][()]
            # lifetime_true_slice = f['lifetime_true'][()]
            # direct store to improve training efficiency
            distribution_true = f['distribution_true'][()]
            lifetime_true = f['lifetime_true'][()]
            TPSF_ARRAY_batch = f['TPSF_ARRAY_batch'][()]
            # distribution_true = np.zeros([5, 32, 72])
            # lifetime_true = np.zeros([5, 32, 72])
            # for i in range(0, 5):
            #     distribution_true[i, :, :] = distribution_true_slice
            #     lifetime_true[i, :, :] = lifetime_true_slice
            #
            # TPSF_ARRAY_batch = np.zeros([220, 11, 11, 4])
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

            yield TPSF_ARRAY_batch, tpsftrue, tpsftimeless, tpsf_logtail, weight_logtail, distribution_true, lifetime_true

            counter = counter + 1
