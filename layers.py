# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 14:32:18 2020

@author: jaych
"""
import numpy as np
import tensorflow.compat.v1 as tf
import time

class batch_norm(object):

    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                            initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                             initializer=tf.random_normal_initializer(1., 0.02))

                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
            x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed


# standard convolution layer
def conv2d(x, inputFeatures, outputFeatures, name):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [5, 5, inputFeatures, outputFeatures],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [outputFeatures], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding="SAME") + b
        return conv


def conv3d(x, inputFeatures, outputFeatures, kernal1, kernal2, kernal3, name, padding_type):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [kernal1, kernal2, kernal3, inputFeatures, outputFeatures],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [outputFeatures], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv3d(x, w, strides=[1, 4, 1, 1, 1], padding=padding_type) + b
        return conv


def conv_transpose(x, outputShape, name):
    with tf.variable_scope(name):
        # h, w, out, in
        w = tf.get_variable("w", [5, 5, outputShape[-1], x.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [outputShape[-1]], initializer=tf.constant_initializer(0.0))
        convt = tf.nn.conv2d_transpose(x, w, output_shape=outputShape, strides=[1, 2, 2, 1])
        return convt


def conv3d_transpose(x, outputShape, name):
    with tf.variable_scope(name):
        # h, w, out, in
        w = tf.get_variable("w", [5, 5, 5, outputShape[-1], x.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [outputShape[-1]], initializer=tf.constant_initializer(0.0))
        convt = tf.nn.conv3d_transpose(x, w, output_shape=outputShape, strides=[1, 1, 2, 2, 1])
        return convt


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv


# leaky reLu unit
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


# fully-conected layer
def dense(x, inputFeatures, outputFeatures, scope=None, with_w=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [inputFeatures, outputFeatures], tf.float32,
                                 tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [outputFeatures], initializer=tf.constant_initializer(0.0))
        if with_w:
            return tf.matmul(x, matrix) + bias, matrix, bias
        else:
            return tf.matmul(x, matrix) + bias


def get_cos_distance(X1, X2, offset):
    # calculate cos distance between two sets
    # more similar more big
    # depth_coefficient = tf.reshape(tf.range(0.1, 3.3, 0.1, tf.float32), [1, 32])
    # depth_coefficient = tf.tile(depth_coefficient, [1, 72])
    # X1 = X1 * depth_coefficient
    # X2 = X2 * depth_coefficient
    X1 = X1 + offset
    X2 = X2 + offset
    X1_norm = tf.sqrt(tf.reduce_sum(tf.square(X1), axis=1))
    X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2), axis=1))
    X1_X2 = tf.reduce_sum(X1 * X2, axis=1)
    X1_X2_norm = X1_norm * X2_norm
    cos = X1_X2 / X1_X2_norm
    return cos

def get_pearson_cor(X1, X2):
    X1_mean = tf.reshape(tf.reduce_mean(X1, axis=1), [-1, 1])
    X2_mean = tf.reshape(tf.reduce_mean(X2, axis=1), [-1, 1])
    X1_new = X1 - X1_mean
    X2_new = X2 - X2_mean
    X1_norm = tf.sqrt(tf.reduce_sum(tf.square(X1_new), axis=1))
    X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2_new), axis=1))
    X1_X2 = tf.reduce_sum(X1_new * X2_new, axis=1)
    X1_X2_norm = X1_norm * X2_norm
    cor = X1_X2 / (X1_X2_norm + 0.0000001)
    return cor

# try slope
def conv_ontime_tensor(x, y):
    n, tl, m = x.shape
    _, _, tl = y.shape
#    x_out = tf.zeros([channels,tl,m],dtype=tf.float32)
    count = 0
    out_splittime = list()
    idxs = [8, 10, 12, 14, 16, 18, 20, 28, 36, 44, 52, 60]
    # idxs = [8, 10, 12, 14, 16, 18, 20, 22, 28, 34, 40, 46, 52]
    # idxs = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    # idxs = np.arange(10, 51, 4)
    for i in idxs:
        x_splittime = tf.split(x, axis=1, num_or_size_splits=[i+1, 76-i-1])
        # x_splittime = tf.gather(x, axis=1, indices=tf.constant(np.arange(0, i+1)))
        y_splittime = tf.split(y, axis=2, num_or_size_splits=[76-i-1, i+1])
        # y_splittime = tf.gather(y, axis=2, indices=tf.constant(np.arange(76-i-1, 76)))
        out_splittime.append(tf.reshape(tf.tensordot(y_splittime[1], x_splittime[0], axes=2), [-1, 1, 45]))
        # out_splittime.append(tf.reshape(tf.tensordot(y_splittime, x_splittime, axes=2), [-1, 1, 45]))
        # if count == 0:
        #     x_out = out_splittime[count]
        # else:
        #     x_out = tf.concat([x_out, out_splittime[count] - out_splittime[count - 1]], axis=1)
        if count == 1:
            x_out = out_splittime[count] - out_splittime[count-1]
        elif count > 1:
            x_out = tf.concat([x_out, out_splittime[count] - out_splittime[count-1]], axis=1)

        count = count + 1
    # x_out = tf.concat(out_splittime, axis=1)
    return x_out

def get_predicted_curve(x, weight, life):
    dt = 12.5 / 1024 * 2
    tlist = tf.reshape(tf.range(300, -1, -4, tf.float32), [1, 76])
    life_reshaped = tf.reshape(life, [-1, 5 * 32 * 72, 1])
    life_e = tf.exp(tf.tensordot(life_reshaped, -tlist*dt, axes=1))
    life_yield = life_e*tf.reshape(x * life, [-1, 5 * 32 * 72, 1])
    start = time.perf_counter()
    tpsfm = conv_ontime_tensor(weight, life_yield)
    end = time.perf_counter()
    print("Time elapsed:", end - start)
    tpsfm_reshaped = tf.reshape(tpsfm, [-1, 11 * 45])
    # loss = tf.reduce_mean(tf.reduce_sum(tf.square(tpsfm_reshaped - tpsf), axis=1), axis=0)
    return tpsfm_reshaped
