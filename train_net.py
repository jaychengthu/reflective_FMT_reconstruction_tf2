# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:38:51 2020

@author: jaych
"""
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()
from tensorflow.python.framework import ops
import scipy.io as sio
import numpy as np
import database
import database_queue
import matplotlib.pyplot as plt
import os
import cv2
from imageio import imwrite as ims
import layers
import analysis
import time


class LatentAttention():
    def __init__(self):
        global model_logtail
        self.datapath_train = '/home/cjj/reflective_FMT_reconstruction_tf2/data/enum_train9/'
        self.datapath_test = '/home/cjj/reflective_FMT_reconstruction_tf2/data/enum_train9/'
        self.datapath_model = 'data/model/'
        self.n_samples_train = 51200
        self.training_batchsize = 64
        self.validation_batchsize = 500
        self.testing_batchsize = 1000
        self.database = database_queue.database_queue(self.datapath_train, self.training_batchsize)
        self.database_test = database_queue.database_queue(self.datapath_test, self.testing_batchsize)
        self.database_valid = database_queue.database_queue(self.datapath_train, self.validation_batchsize)
        model_matrix_all = database.get_model(self.datapath_model)
        idx_meas = list()
        for i in range(0, 11):
            for j in range(i + 1, min(i + 7, 11)):
                idx_meas.append(i * 11 + j)
        self.model_matrix = model_matrix_all[:, :, idx_meas]

        model_matrix_time_less = np.sum(self.model_matrix, axis=1)
        idx_meas2 = list()
        for i in range(0, 11):
            for j in range(max(i - 3, 0), min(i + 4, 11)):
                if i == j:
                    continue
                idx_meas2.append(i * 11 + j)

        model_logtail = model_matrix_all[:, 45, idx_meas2].reshape(5 * 32 * 72, 54)

        self.n_z = 128
        self.batchsize = tf.placeholder(tf.int32)
        self.training_stepsize = tf.placeholder(tf.float32)
        self.weightoflinear = tf.placeholder(tf.float32)

        self.tpsf_array = tf.placeholder(tf.float32, [None, 220, 11, 11, 4])
        self.tpsftrue = tf.placeholder(tf.float32, [None, 11, 45])
        self.tpsftrue_timeless = tf.placeholder(tf.float32, [None, 45])
        self.tpsf_logtail = tf.placeholder(tf.float32, [None, 54])
        self.weight_logtail = tf.placeholder(tf.float32, [None, 54])
        self.tpsf_array_reshaped = tf.reshape(self.tpsftrue, [-1, 11 * 45])
        self.distribution_true = tf.placeholder(tf.float32, [None, 5, 32, 72])
        self.lifetime_true = tf.placeholder(tf.float32, [None, 5, 32, 72])
        lifetime_true_flattened = tf.reshape(self.lifetime_true, [-1, 5 * 32 * 72])
        distribution_true_flattened = tf.reshape(self.distribution_true, [-1, 5 * 32 * 72])
        self.guessed_z = self.recognition(self.tpsf_array)
        self.given_z = tf.placeholder(tf.float32, [None, self.n_z])

        self.reconstructed_distribution, self.reconstructed_lifetime, self.mask = self.generation(self.guessed_z)


        self.flatten_distribution = tf.reshape(self.reconstructed_distribution, [-1, 5 * 32 * 72])
        flatten_lifetime = tf.reshape(self.reconstructed_lifetime, [-1, 5 * 32 * 72])
        # self.reconstructed_lifetime = tf.reshape(flatten_lifetime, [-1, 5, 32, 72])
        self.yita_loss = layers.get_cos_distance(distribution_true_flattened, self.flatten_distribution, 0.0001)
        # self.yita_loss = layers.get_pearson_cor(distribution_true_flattened, self.flatten_distribution)
        # self.yita_loss = tf.reduce_sum(tf.square(distribution_true_flattened - self.flatten_distribution), axis=1)

        self.Tpsf_logtail2 = self.tpsf_logtail

        # depth_reg = np.arange(0, 32)
        depth_reg = - np.arange(-15, 17).reshape(32, 1) * 30
        depth_reg[depth_reg <= 20] = 20
        depth_reg[depth_reg >= 0] = depth_reg[depth_reg >= 0] * 2
        depth_reg[depth_reg >= 300] = 300
        depth_reg = depth_reg / 20
        depth_cof = np.tile(depth_reg, (5, 1, 72))
        depth_cof = depth_cof.reshape(5*32*72)

        depth_reg = np.arange(0, 32)
        depth_reg = np.exp(-0.001 * (depth_reg - 12)).reshape(32, 1)
        depth_reg = depth_reg / np.amax(depth_reg) / 10
        depth_cof2 = np.tile(depth_reg, (5, 1, 72))
        depth_cof2 = depth_cof2.reshape(5 * 32 * 72)

        # depth_reg = np.arange(0, 32)
        depth_reg = - np.arange(-15, 17).reshape(32, 1) * 30
        depth_reg[depth_reg <= -200] = -200
        # depth_reg[depth_reg >= 0] = depth_reg[depth_reg >= 0] *
        depth_reg[depth_reg >= 200] = 200
        depth_reg = depth_reg / 20
        depth_cof3 = np.tile(depth_reg, (5, 1, 72))
        depth_cof3 = depth_cof.reshape(5 * 32 * 72)

        reconstructed_lifetime_tv = tf.transpose(tf.reshape(flatten_lifetime, [-1, 5, 32, 72]), [0, 2, 3, 1])#* (depth_cof2 + 0.1)
        reconstructed_distribution_tv = tf.transpose(tf.reshape(self.flatten_distribution * (depth_cof2 + 0.1), [-1, 5, 32, 72]), [0, 2, 3, 1])

        self.life_loss2 = tf.reduce_sum(tf.square(flatten_lifetime) * depth_cof, axis=1)
        self.life_loss = layers.get_cos_distance(lifetime_true_flattened, flatten_lifetime, 0.001)
        self.life_loss3 = tf.reduce_sum(tf.square(flatten_lifetime-lifetime_true_flattened), axis=1)
        self.yita_loss2 = tf.reduce_sum(tf.square(distribution_true_flattened) * depth_cof3, axis=1)
        self.yita_loss3 = tf.reduce_sum(tf.square(distribution_true_flattened - self.flatten_distribution), axis=1)
        self.tpsf_timeless = tf.matmul(self.flatten_distribution, model_matrix_time_less)
        self.tpsf_predict = layers.get_predicted_curve(
            self.flatten_distribution, self.model_matrix, flatten_lifetime)
        # self.tpsf_predict2 = layers.get_predicted_curve(
        #     distribution_true_flattened, self.model_matrix, lifetime_true_flattened)
        self.curve_loss = layers.get_cos_distance(self.tpsf_array_reshaped,
                                                  self.tpsf_predict, 1e-6)
        self.curve_loss2 = layers.get_cos_distance(self.tpsftrue_timeless,
                                                   self.tpsf_timeless, 1e-6)

        self.lifetimeweight = tf.reshape(self.flatten_distribution * tf.reduce_mean(model_logtail, axis=1), [-1, 5, 32, 72])
        self.tpsf_logtail_pred = tf.div(tf.matmul(self.flatten_distribution * flatten_lifetime, model_logtail),
                                        tf.matmul(self.flatten_distribution + 1e-10, model_logtail))

        self.curve_loss3 = tf.reduce_sum(tf.square((self.tpsf_logtail_pred - self.tpsf_logtail) * self.weight_logtail),
                                         axis=1)
        self.curve_loss4 = tf.reduce_sum(tf.square((self.tpsf_logtail_pred - self.Tpsf_logtail2) * self.weight_logtail),
                                         axis=1)
        # self.curve_loss = tf.reduce_sum(tf.abs(self.tpsf_predict/(tf.reduce_mean(self.tpsf_predict)+1e-5) - self.tpsf_array_reshaped/(tf.reduce_mean(self.tpsf_array_reshaped))+1e-5), axis=1)
        self.L1_loss = tf.reduce_mean(tf.reduce_sum(tf.reshape(self.mask, [-1, 5 * 32 * 72]) , axis=1))
        self.L1_loss2 = tf.reduce_mean(tf.reduce_sum(tf.square(self.flatten_distribution) * depth_cof2, axis=1))
        # self.L1_loss = tf.reduce_mean(tf.div(tf.reduce_sum(tf.square(self.flatten_distribution)*depth_cof, axis=1),
        #                                      tf.reduce_sum(tf.square(self.flatten_distribution) + 1e-10, axis=1)))
        # self.L1_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.flatten_distribution) * depth_cof, axis=1))
        self.cost_curve = 1 - tf.reduce_mean(self.curve_loss)
        self.cost_curve2 = 1 - tf.reduce_mean(self.curve_loss2)
        self.cost_curve3 = tf.reduce_mean(self.curve_loss3)
        self.cost_curve4 = tf.reduce_mean(self.curve_loss4)
        self.cost_yita = 1 - tf.reduce_mean(self.yita_loss)
        self.cost_yita2 = tf.reduce_mean(self.yita_loss2)
        self.cost_yita3 = tf.reduce_mean(self.yita_loss3)
        self.cost_life = 1 - tf.reduce_mean(self.life_loss)
        self.cost_life2 = tf.reduce_mean(self.life_loss2)
        self.cost_life3 = tf.reduce_mean(self.life_loss3)
        self.lifetime_tvloss = tf.reduce_mean(reconstructed_lifetime_tv)
        self.distribution_tvloss = tf.reduce_mean(reconstructed_distribution_tv)
        # self.cost = 2 * self.cost_curve + 10 * self.cost_curve2 + 2 * self.cost_yita + 2 * self.cost_life
        # self.cost_start = 2 * self.cost_curve + 10 * self.cost_curve2 + 2 * self.cost_yita + 2 * self.cost_life
        self.cost_start = 5 * self.cost_curve + 2 * self.cost_curve2 + 5 * self.weightoflinear * self.cost_curve4 + 2 * self.cost_yita \
                          + 2 * self.cost_life + self.weightoflinear * 1e-2 * (self.cost_life2) + self.weightoflinear * 1e-2 * (self.cost_yita2) \
                          + self.weightoflinear * 1e-2 * self.L1_loss + self.weightoflinear * 1e2 * (
                                      self.lifetime_tvloss + self.distribution_tvloss) \
                          + self.weightoflinear * 1e-1 * self.cost_yita3
        self.cost = 5 * self.cost_curve + 2 * self.cost_curve2 + 5 * self.weightoflinear * self.cost_curve4 + 2 * self.cost_yita \
                    + 2 * self.cost_life + self.weightoflinear * 1e-2 * (self.cost_life2) + self.weightoflinear * 1e-2 * (self.cost_yita2) \
                    + self.weightoflinear * 1e-2 * self.L1_loss + self.weightoflinear * 1e2 * (
                                      self.lifetime_tvloss + self.distribution_tvloss) \
                    + self.weightoflinear * 1e-1 * self.cost_yita3

        self.optimizer_start = tf.train.AdamOptimizer(self.training_stepsize).minimize(self.cost_start)
        self.optimizer = tf.train.AdamOptimizer(self.training_stepsize).minimize(self.cost)

    # encoder
    def recognition(self, input_tpsf_array):
        with tf.variable_scope("recognition"):
            h1 = layers.lrelu(
                layers.conv3d(input_tpsf_array, 4, 32, 15, 5, 5, "conv_h1", "SAME"))  # 220x11x11x3 -> 45x11x11x32
            h2 = layers.lrelu(layers.conv3d(h1, 32, 64, 11, 3, 3, "conv_h2", "VALID"))  # 45x11x11x32 -> 12x7x7x64
            h3 = layers.lrelu(layers.conv3d(h2, 64, 128, 11, 3, 3, "conv_h3", "VALID"))  # 12x7x7x64 -> 1x5x5x128
            # h4 = layers.lrelu(layers.conv3d(h3, 128, 256, 11, 3, 3, "conv_h4", "VALID"))  # 16x7x7x128 -> 3x5x5x256
            h4_flat = tf.reshape(h3, [-1, 1 * 7 * 7 * 128])
            h5 = layers.lrelu(layers.dense(h4_flat, 1 * 7 * 7 * 128, 1024, "h5"))
            h6 = layers.lrelu(layers.dense(h5, 1024, 256, "h6"))
            h7 = layers.lrelu(layers.dense(h6, 256, self.n_z, "w_mean"))

        return h7

    # decoder
    def generation(self, z):
        with tf.variable_scope("generation", reuse=tf.AUTO_REUSE):
            z_develop = tf.nn.relu(layers.dense(z, self.n_z, 1 * 4 * 9 * 8, scope='z_matrix'))
            z_develop2 = tf.nn.relu(layers.dense(z_develop, 1 * 4 * 9 * 8, 1 * 8 * 18 * 16, scope='z_develop2'))
            z_develop3 = tf.nn.relu(layers.dense(z_develop2, 1 * 8 * 18 * 16, 5 * 16 * 36 * 4, scope='z_develop3'))
            z_develop_life = tf.nn.relu(layers.dense(z, self.n_z, 1 * 4 * 9 * 8,
                                                     scope='z_matrix_life'))
            z_develop2_life = tf.nn.relu(layers.dense(z_develop_life, 1 * 4 * 9 * 8, 1 * 8 * 18 * 16,
                                                      scope='z_develop2_life'))
            z_develop3_life = tf.nn.relu(layers.dense(z_develop2_life, 1 * 8 * 18 * 16, 5 * 16 * 36 * 4,
                                                      scope='z_develop3_life'))
            z_matrix = tf.nn.relu(tf.reshape(z_develop3, [self.batchsize, 5, 16, 36, 4]))
            z_matrix_life = tf.nn.relu(tf.reshape(z_develop3_life, [self.batchsize, 5, 16, 36, 4]))
            h3 = tf.nn.relu(layers.conv3d_transpose(z_matrix, [self.batchsize, 5, 32, 72, 1], "g_h3"))
            h3_life = tf.nn.relu(layers.conv3d_transpose(z_matrix_life, [self.batchsize, 5, 32, 72, 1], "g_h3_life"))

            h3 = tf.reshape(h3, [self.batchsize, 5, 32, 72])
            h3_life = tf.reshape(h3_life, [self.batchsize, 5, 32, 72])
            mask1 = tf.sigmoid(150 * (h3 - 0.15))
            mask = mask1 * tf.sigmoid(150 * (h3_life - 0.5)) + 1e-3

        return h3 * mask, h3_life * mask, mask1

    def train(self):
        # train
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            module_file = tf.train.latest_checkpoint(os.getcwd() + "/training")
            saver = tf.train.Saver(max_to_keep=3)
            saver.restore(sess, module_file)

            training_step = 5e-5
            batch_tpsf_valid_tf, batch_tpsftrue_valid_tf, batch_tpsftimeless_valid_tf, batch_tpsflogtail_valid_tf, \
            batch_weightlogtail_valid_tf, batch_distribution_valid_tf, batch_lifetime_valid_tf = \
                self.database_valid.iterator.get_next()
            batch_tpsf_tf, batch_tpsftrue_tf, batch_tpsftimeless_tf, batch_tpsflogtail_tf, \
            batch_weightlogtail_tf, batch_distribution_tf, batch_lifetime_tf = \
                self.database.iterator.get_next()
            batch_tpsf_valid, batch_tpsftrue_valid, batch_tpsftimeless_valid, batch_tpsflogtail_valid, \
            batch_weightlogtail_valid, batch_distribution_valid, batch_lifetime_valid = \
                sess.run((batch_tpsf_valid_tf, batch_tpsftrue_valid_tf, batch_tpsftimeless_valid_tf,
                          batch_tpsflogtail_valid_tf, batch_weightlogtail_valid_tf, batch_distribution_valid_tf,
                          batch_lifetime_valid_tf))

            cost_line = list()
            cost_curve = list()
            cost_life = list()
            start = time.perf_counter()
            weightoflinear = 0.001#
            for epoch in range(0, 15):
                for idx in range(int(self.n_samples_train / self.training_batchsize)):

                    batch_tpsf, batch_tpsftrue, batch_tpsftrue_timeless, batch_tpsflogtail, \
                    batch_weightlogtail, batch_distribution, batch_lifetime = \
                        sess.run((batch_tpsf_tf, batch_tpsftrue_tf, batch_tpsftimeless_tf, batch_tpsflogtail_tf,
                                  batch_weightlogtail_tf, batch_distribution_tf, batch_lifetime_tf))


                    # plt.figure()
                    # plt.imshow(batch_distribution[0][0])
                    # plt.show()
                    # batch_tpsf, batch_distribution, batch_lifetime, _ = self.database_test.get_test_batch(64)
                    # start = time.perf_counter()
                    if epoch <= 2:
                        _ = sess.run((self.optimizer_start),
                                     feed_dict={self.tpsf_array: batch_tpsf,
                                                self.distribution_true: batch_distribution,
                                                self.lifetime_true: batch_lifetime,
                                                self.batchsize: self.training_batchsize,
                                                self.training_stepsize: training_step,
                                                self.tpsftrue: batch_tpsftrue,
                                                self.tpsftrue_timeless: batch_tpsftrue_timeless,
                                                self.tpsf_logtail: batch_tpsflogtail,
                                                self.weightoflinear: weightoflinear,
                                                self.weight_logtail: batch_weightlogtail
                                                })
                    else:
                        _ = sess.run((self.optimizer),
                                     feed_dict={self.tpsf_array: batch_tpsf,
                                                self.distribution_true: batch_distribution,
                                                self.lifetime_true: batch_lifetime,
                                                self.batchsize: self.training_batchsize,
                                                self.training_stepsize: training_step,
                                                self.tpsftrue: batch_tpsftrue,
                                                self.tpsftrue_timeless: batch_tpsftrue_timeless,
                                                self.tpsf_logtail: batch_tpsflogtail,
                                                self.weightoflinear: weightoflinear,
                                                self.weight_logtail: batch_weightlogtail
                                                })
                    # end = time.perf_counter()
                    # print("Training Time elapsed:", end - start)

                    # if epoch >= 2:
                    #     _ = sess.run((self.optimizer),
                    #                  feed_dict={self.tpsf_array: batch_tpsf,
                    #                             self.distribution_true: batch_distribution,
                    #                             self.lifetime_true: batch_lifetime,
                    #                             self.batchsize: self.training_batchsize,
                    #                             self.training_stepsize: training_step,
                    #                             self.tpsftrue: batch_tpsftrue
                    #                             })

                    if idx % 10 == 0:
                        valid_cost, L1, curve_loss2, curve_loss3, cost_life2, distribution_tvloss, lifetime_tvloss, \
                        lifeloss, yield_reconstructed, lifetime_reconstructed = \
                            sess.run((self.cost_yita, self.cost_life, self.cost_curve2, self.cost_curve3, self.cost_life2,
                                      self.distribution_tvloss, self.lifetime_tvloss, self.cost_life3,
                                      self.reconstructed_distribution, self.reconstructed_lifetime), \
                                     feed_dict={self.tpsf_array: batch_tpsf_valid,
                                                self.distribution_true: batch_distribution_valid,
                                                self.lifetime_true: batch_lifetime_valid,
                                                self.batchsize: self.validation_batchsize,
                                                self.tpsftrue: batch_tpsftrue_valid,
                                                self.tpsftrue_timeless: batch_tpsftimeless_valid,
                                                self.tpsf_logtail: batch_tpsflogtail_valid,
                                                self.weightoflinear: weightoflinear,
                                                self.weight_logtail: batch_weightlogtail_valid
                                                })
                        curve_loss, gen_loss = sess.run((self.cost_curve, self.cost), \
                                                        feed_dict={self.tpsf_array: batch_tpsf,
                                                                   self.distribution_true: batch_distribution,
                                                                   self.lifetime_true: batch_lifetime,
                                                                   self.batchsize: self.training_batchsize,
                                                                   self.training_stepsize: training_step,
                                                                   self.tpsftrue: batch_tpsftrue,
                                                                   self.tpsftrue_timeless: batch_tpsftrue_timeless,
                                                                   self.tpsf_logtail: batch_tpsflogtail,
                                                                   self.weightoflinear: weightoflinear,
                                                                   self.weight_logtail: batch_weightlogtail
                                                                   })
                        cost_line.append(valid_cost)
                        cost_curve.append(lifeloss)
                        cost_life.append(L1)
                        if idx % 100 == 0:
                            saver.save(sess, os.getcwd() + "/training/train", global_step=epoch)
                            length = len(cost_line)
                            plt.close("all")
                            if length >= 150:
                                plt.plot(cost_line[length - 150:length])
                            else:
                                plt.plot(cost_line)
                            # plt.plot(cost_life, color='red')
                            # plt.plot(cost_curve, color='green')
                            plt.show()
                        if idx % 5000 == 0 and 0:
                            sio.savemat(os.getcwd() + '/results/' + 'valid_row_data.mat',
                                        {'yita_true': batch_distribution_valid,
                                         'lifetime_true': batch_lifetime_valid,
                                         'yita_recon': yield_reconstructed,
                                         'lifetime_recon': lifetime_reconstructed,
                                         })
                        if idx % 100 == 0 and 1:
                            path = 'data/real_data/'
                            TPSF_ARRAY, tpsftrue_array_vis, tpsftimeless_array_vis, tpsflogtail_array_vis, weightlogtail_array_vis, realdatalist \
                                = database.get_test_data(path)
                            generated_test, generated_lifetime_test, lifetimeweight = \
                                sess.run(
                                    (self.reconstructed_distribution, self.reconstructed_lifetime, self.lifetimeweight), \
                                    feed_dict={self.tpsf_array: TPSF_ARRAY,
                                               self.tpsftrue: tpsftrue_array_vis,
                                               self.tpsftrue_timeless: tpsftimeless_array_vis,
                                               self.tpsf_logtail: tpsflogtail_array_vis,
                                               self.weight_logtail: weightlogtail_array_vis,
                                               self.weightoflinear: 0.1,
                                               self.batchsize: TPSF_ARRAY.shape[0]})

                            idxs2show = [21, 22, 24, 25, 34, 35, 45]
                            for i in idxs2show:  # range(0, generated_test.shape[0]):
                                generated_test1 = generated_test[i][2]
                                generated_lifetime_test1 = generated_lifetime_test[i][2]
                                weightoflifetime = lifetimeweight[i][2]

                                yield_value_recon_list = analysis.yita_recon_quality(generated_test1, realdatalist[i])
                                lifetime_value_recon_list = analysis.effective_lifetime(generated_lifetime_test1,
                                                                                        weightoflifetime,
                                                                                        realdatalist[i])

                        weightoflinear = 0.99 * weightoflinear + 0.01 * 0.01 * np.exp(10 * (0.2 - L1))
                        print("epoch %d: genloss %f" % (epoch, gen_loss))
                        print("epoch %d: curveloss %f" % (epoch, curve_loss))
                        print("epoch %d: curveloss2 %f" % (epoch, curve_loss2))
                        print("epoch %d: curveloss3 %f" % (epoch, curve_loss3))
                        print("epoch %d: loss of validation %f" % (epoch, valid_cost))
                        print("cost_life2:", cost_life2)
                        print("(life-true)**2:", lifeloss)
                        print(L1)
                        print(weightoflinear)
                        print("distribution TV:",distribution_tvloss)
                        print("Lifetime TV:", lifetime_tvloss)



                    if L1 <= 0.42:
                        training_step = 4e-5
                    if L1 <= 0.31:
                        training_step = 2e-5
                    if L1 <= 0.20:
                        training_step = 1e-5
                    if L1 <= 0.19:
                        training_step = 5e-6
                    if L1 <= 0.18:
                        training_step = 2e-6
                    if L1 <= 0.17:
                        training_step = 1e-6
            print("epoch %d: genloss %f " % (epoch, np.mean(gen_loss)))
            saver.save(sess, os.getcwd() + "/training/train", global_step=epoch)
            end = time.perf_counter()
            print("Time elapsed:", end - start)
            return 0

    def test(self):
        # gpu_options = tf.GPUOptions(allow_growth=True)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            module_file = tf.train.latest_checkpoint(os.getcwd() + "/training")
            saver = tf.train.Saver(max_to_keep=2)
            saver.restore(sess, module_file)

            tpsf_test_tf, tpsftrue_test_tf, tpsftimeless_test_tf, tpsflogtail_test_tf, \
            weightlogtail_test_tf, distribution_test_tf, lifetime_test_tf = \
                self.database_test.iterator.get_next()

            start = time.perf_counter()
            tpsf_array_vis, tpsftrue_array_vis, tpsftimeless_array_vis, tpsflogtail_array_vis, \
            weightlogtail_array_vis, distribution_true_vis, lifetime_true_vis = \
                sess.run((tpsf_test_tf, tpsftrue_test_tf, tpsftimeless_test_tf, tpsflogtail_test_tf,
                          weightlogtail_test_tf, distribution_test_tf, lifetime_test_tf))
            end = time.perf_counter()
            print("Time elapsed_total:", end - start)
            generated_test, generated_lifetime_test, test_cost, tpsf_predict, \
            curveloss, curveloss2, curveloss3, tpsf_logtail_pred, life_loss2, lifetimeweight = \
                sess.run((self.reconstructed_distribution, self.reconstructed_lifetime,
                          self.cost, self.tpsf_predict, self.curve_loss, self.curve_loss2,
                          self.curve_loss3, self.tpsf_logtail_pred, self.life_loss2, self.lifetimeweight),
                         feed_dict={self.tpsf_array: tpsf_array_vis,
                                    self.distribution_true: distribution_true_vis,
                                    self.lifetime_true: lifetime_true_vis,
                                    self.batchsize: tpsf_array_vis.shape[0],
                                    self.tpsftrue: tpsftrue_array_vis,
                                    self.tpsftrue_timeless: tpsftimeless_array_vis,
                                    self.tpsf_logtail: tpsflogtail_array_vis,
                                    self.weightoflinear: 0.1,
                                    self.weight_logtail: weightlogtail_array_vis
                                    })
            sio.savemat(os.getcwd() + '/results/' + 'test_row_data.mat', {'yita_true': distribution_true_vis,
                                                                     'lifetime_true': lifetime_true_vis,
                                                                     'yita_recon': generated_test,
                                                                     'lifetime_recon': generated_lifetime_test,
                                                                     })
            print("test_lost:", test_cost)
            # print(depth_cof)
            # plt.figure()
            # plt.imshow(distribution_true_vis[0, 0, :, :])
            # plt.show()
            # plt.figure()
            # plt.imshow(depth_cof[0])
            # plt.show()
            # for i in range(middle_result[0].shape[3]):
            #     plt.figure()
            #     plt.imshow(middle_result[0][0, :, :, i])
            #     plt.show()
            for i in range(0, 50, 1):
                print(curveloss[i])
                print(curveloss2[i])
                print(curveloss3[i])
                print(life_loss2[i])
                tmp = np.zeros([11, 121], dtype=np.float32)
                tmp2 = np.zeros([11, 121], dtype=np.float32)
                idx_meas = list()
                for k in range(0, 11):
                    for j in range(k + 1, min(k + 7, 11)):
                        idx_meas.append(k * 11 + j)
                tmp[:, idx_meas] = tpsf_predict[i].reshape(11, 45)
                curve_tmp = tpsf_predict[i].reshape(11, 45)
                tmp2[:, idx_meas] = tpsftrue_array_vis[i].reshape(11, 45)
                curve_tmp2 = tpsftrue_array_vis[i].reshape(11, 45)
                tmp = tmp.reshape(11, 11, 11)
                tmp2 = tmp2.reshape(11, 11, 11)
                plt.figure()
                for k in range(0, 11, 1):
                    plt.subplot(4, 6, 1 + k)
                    plt.imshow(tmp[k, :, :])
                    plt.subplot(4, 6, 1 + k + 12)
                    plt.imshow(tmp2[k, :, :])
                plt.show()
                plt.figure()
                for k in range(0, 45, 1):
                    plt.subplot(5, 9, 1 + k)
                    plt.plot(curve_tmp[:, k] / np.amax(curve_tmp))
                    plt.plot(curve_tmp2[:, k] / np.amax(curve_tmp2), color='red')
                plt.show()
                plt.figure()
                plt.plot(tpsf_logtail_pred[i])
                plt.plot(tpsflogtail_array_vis[i], color='red')
                plt.plot(weightlogtail_array_vis[i], color='green')
                plt.show()
                # plt.figure()
                # for k in range(0, 25, 1):
                #     plt.subplot(5, 5, 1 + k)
                #     plt.plot(np.log(curve_tmp[:, k] / np.amax(curve_tmp)))
                #     plt.plot(np.log(curve_tmp2[:, k] / np.amax(curve_tmp2)), color='red')
                #     plt.plot(np.log(curve_tmp3[:, k] / np.amax(curve_tmp3)), color='green')
                # plt.show()
                # dt = 12.5 / 1024 * 2
                # for k in range(0, 45, 1):
                #     print(idx_meas[k] % 11, np.floor(idx_meas[k]/11), ':',(np.log(curve_tmp[10, k]) - np.log(curve_tmp[8, k])) / dt / 15 / 2)
                #     print((np.log(curve_tmp2[10, k]) - np.log(curve_tmp2[7, k])) / dt / 15 / 3)
                #     print((np.log(curve_tmp3[10, k]) - np.log(curve_tmp3[7, k])) / dt / 15 / 3)
                # [r,c] = np.where(tmp2[2,:,:] >= np.amax(tmp2[2,:,:]))
                # print(r,c)
                weightoflife = lifetimeweight[i][2]
                generated_test1 = generated_test[i][2]
                generated_lifetime_test1 = generated_lifetime_test[i][2]
                distribution_true_vis_show = distribution_true_vis[i][2]
                lifetime_true_vis_show = lifetime_true_vis[i][2]
                yield_value_recon_list = analysis.yita_recon_quality(distribution_true_vis_show, str(i)+'true')
                yield_value_recon_list = analysis.yita_recon_quality(generated_test1, i)
                lifetime_value_true_list = analysis.effective_lifetime(lifetime_true_vis_show,
                                                                       distribution_true_vis_show, str(i)+'true')
                lifetime_value_recon_list = analysis.effective_lifetime(generated_lifetime_test1, weightoflife, i)

    def application(self):
        with tf.Session() as sess:
            #            print(os.getcwd()+"/training")
            module_file = tf.train.latest_checkpoint(os.getcwd() + '/training')
            saver = tf.train.Saver(max_to_keep=2)
            saver.restore(sess, module_file)
            path = 'data/real_data/'
            TPSF_ARRAY, tpsftrue_array_vis, tpsftimeless_array_vis, tpsflogtail_array_vis, weightlogtail_array_vis, realdatalist \
                = database.get_test_data(path)

            # training_step = 5e-5
            # for i in range(10):
            #     _ = sess.run((self.optimizer),
            #                  feed_dict={self.tpsf_array: TPSF_ARRAY,
            #                             self.batchsize: TPSF_ARRAY.shape[0],
            #                             self.training_stepsize: training_step,
            #                             self.tpsftrue: tpsftrue_array_vis
            #                             })
            #     curve_cost = sess.run(self.cost_curve,
            #                 feed_dict={self.tpsf_array: TPSF_ARRAY,
            #                            self.tpsftrue: tpsftrue_array_vis,
            #                            self.batchsize: TPSF_ARRAY.shape[0]})
            #     saver.save(sess, os.getcwd() + "/training/train", global_step=10)
            #     print(curve_cost)

            generated_test, generated_lifetime_test, tpsf_predict, tpsflogtail_pred, curve_cost, curve_cost2, \
            curve_cost3, tpsf_timeless, lifetimeweight = \
                sess.run((self.reconstructed_distribution, self.reconstructed_lifetime, self.tpsf_predict,
                          self.tpsf_logtail_pred, self.curve_loss, self.curve_loss2, self.curve_loss3,
                          self.tpsf_timeless, self.lifetimeweight), \
                         feed_dict={self.tpsf_array: TPSF_ARRAY,
                                    self.tpsftrue: tpsftrue_array_vis,
                                    self.tpsftrue_timeless: tpsftimeless_array_vis,
                                    self.tpsf_logtail: tpsflogtail_array_vis,
                                    self.weight_logtail: weightlogtail_array_vis,
                                    self.weightoflinear: 0.1,
                                    self.batchsize: TPSF_ARRAY.shape[0]})
            sio.savemat(os.getcwd() + '/results/' + 'real_row_data.mat', {'yita_recon': generated_test,
                                                                          'lifetime_recon': generated_lifetime_test,
                                                                          })
            idxs2show = np.arange(0, generated_test.shape[0])
            # idxs2show = np.arange(32, 38)
            # idxs2show = [28, 29, 32, 34, 35, 36]
            for i in idxs2show:  # range(0, generated_test.shape[0]):
                print(i, realdatalist[i])
                print(curve_cost[i])
                print(curve_cost2[i])
                plt.close("all")
                # plt.figure()
                # plt.plot(tpsf_timeless[i] / np.amax(tpsf_timeless[i]))
                # plt.plot(tpsftimeless_array_vis[i] / np.amax(tpsftimeless_array_vis[i]), color='red')
                # plt.show()
                tmp = np.zeros([11, 121], dtype=np.float32)
                tmp2 = np.zeros([11, 121], dtype=np.float32)
                idx_meas = list()
                for k in range(0, 11):
                    for j in range(k + 1, min(k + 7, 11)):
                        idx_meas.append(k * 11 + j)
                tmp[:, idx_meas] = tpsf_predict[i].reshape(11, 45)
                curve_tmp = tpsf_predict[i].reshape(11, 45)
                tmp2[:, idx_meas] = tpsftrue_array_vis[i].reshape(11, 45)
                curve_tmp2 = tpsftrue_array_vis[i].reshape(11, 45)
                tmp = tmp.reshape(11, 11, 11)
                tmp2 = tmp2.reshape(11, 11, 11)
                plt.figure()
                for k in range(0, 11, 1):
                    plt.subplot(4, 6, 1 + k)
                    plt.imshow(tmp[k, :, :])
                    plt.subplot(4, 6, 1 + k + 12)
                    plt.imshow(tmp2[k, :, :])
                # plt.savefig(os.getcwd() + '/results/' + realdatalist[i] + 'map.jpg')
                plt.show()
                plt.figure()
                for k in range(0, 45, 1):
                    plt.subplot(5, 9, 1 + k)
                    plt.plot(curve_tmp[:, k] / np.amax(curve_tmp))
                    plt.plot(curve_tmp2[:, k] / np.amax(curve_tmp2), color='red')
                    plt.yticks(size=4)
                    plt.xticks(size=5)
                plt.show()
                plt.figure()
                plt.plot(tpsflogtail_pred[i])
                plt.plot(tpsflogtail_array_vis[i], color='red')
                plt.plot(weightlogtail_array_vis[i], color='green')
                plt.savefig(os.getcwd() + '/results/' + realdatalist[i] + 'curve.jpg')
                plt.show()
                generated_test1 = generated_test[i][2]
                generated_lifetime_test1 = generated_lifetime_test[i][2]
                weightoflifetime = lifetimeweight[i][2]

                yield_value_recon_list = analysis.yita_recon_quality(generated_test1, realdatalist[i])
                lifetime_value_recon_list = analysis.effective_lifetime(generated_lifetime_test1, weightoflifetime, realdatalist[i])

            # writer = tf.summary.FileWriter('tensorboard/', sess.graph)


    def application_for_test(self):
        with tf.Session() as sess:
            module_file = tf.train.latest_checkpoint(os.getcwd() + '/training')
            saver = tf.train.Saver(max_to_keep=2)
            saver.restore(sess, module_file)
            path = 'data/test_data/'
            TPSF_ARRAY, tpsftrue_array_vis, tpsftimeless_array_vis, tpsflogtail_array_vis, weightlogtail_array_vis, realdatalist \
                = database.get_test_data(path)

            generated_test, generated_lifetime_test, tpsf_predict, tpsflogtail_pred, curve_cost, curve_cost2, \
            curve_cost3, tpsf_timeless, lifetimeweight = \
                sess.run((self.reconstructed_distribution, self.reconstructed_lifetime, self.tpsf_predict,
                          self.tpsf_logtail_pred, self.curve_loss, self.curve_loss2, self.curve_loss3,
                          self.tpsf_timeless, self.lifetimeweight), \
                         feed_dict={self.tpsf_array: TPSF_ARRAY,
                                    self.tpsftrue: tpsftrue_array_vis,
                                    self.tpsftrue_timeless: tpsftimeless_array_vis,
                                    self.tpsf_logtail: tpsflogtail_array_vis,
                                    self.weight_logtail: weightlogtail_array_vis,
                                    self.weightoflinear: 0.1,
                                    self.batchsize: TPSF_ARRAY.shape[0]})
            sio.savemat(os.getcwd() + '/results/' + 'simu_test_row_data.mat', {'yita_recon': generated_test,
                                                                          'lifetime_recon': generated_lifetime_test,
                                                                          })
            idxs2show = [0, 11, 16, 17, 18, 19]
            idxs2show = np.arange(0, 1)
            for i in idxs2show:
                print(i, realdatalist[i])
                print(curve_cost[i])
                print(curve_cost2[i])
                tmp = np.zeros([11, 121], dtype=np.float32)
                tmp2 = np.zeros([11, 121], dtype=np.float32)
                idx_meas = list()
                for k in range(0, 11):
                    for j in range(k + 1, min(k + 7, 11)):
                        idx_meas.append(k * 11 + j)
                tmp[:, idx_meas] = tpsf_predict[i].reshape(11, 45)
                curve_tmp = tpsf_predict[i].reshape(11, 45)
                tmp2[:, idx_meas] = tpsftrue_array_vis[i].reshape(11, 45)
                curve_tmp2 = tpsftrue_array_vis[i].reshape(11, 45)
                tmp = tmp.reshape(11, 11, 11)
                tmp2 = tmp2.reshape(11, 11, 11)
                # plt.figure()
                # for k in range(0, 11, 1):
                #     plt.subplot(4, 6, 1 + k)
                #     plt.imshow(tmp[k, :, :])
                #     plt.subplot(4, 6, 1 + k + 12)
                #     plt.imshow(tmp2[k, :, :])
                # plt.show()
                plt.figure()
                for k in range(0, 45, 1):
                    plt.subplot(5, 9, 1 + k)
                    plt.plot(curve_tmp[:, k] / np.amax(curve_tmp))
                    plt.plot(curve_tmp2[:, k] / np.amax(curve_tmp2), color='red')
                    plt.yticks(size=4)
                    plt.xticks(size=5)
                plt.show()
                plt.figure()
                plt.plot(tpsflogtail_pred[i])
                plt.plot(tpsflogtail_array_vis[i], color='red')
                plt.plot(weightlogtail_array_vis[i], color='green')
                plt.show()
                generated_test1 = generated_test[i][2]
                generated_lifetime_test1 = generated_lifetime_test[i][2]
                weightoflifetime = lifetimeweight[i][2]

                yield_value_recon_list = analysis.yita_recon_quality(generated_test1, realdatalist[i])
                lifetime_value_recon_list = analysis.effective_lifetime(generated_lifetime_test1, weightoflifetime, realdatalist[i])


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # use GPU with ID = 0
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # maximun alloc gpu50% of MEM
    # config.gpu_options.allow_growth = True  # allocate dynamically
    ops.reset_default_graph()
    model = LatentAttention()
    model.train()
    # model.test()
    model.application_for_test()
    # model.application()
