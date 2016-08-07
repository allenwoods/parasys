#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file       ac_model.py
@author     Allen Woods
@date       2016-08-07
@version    16-8-7 下午3:57 ???
Some other Description
"""
from keras.layers import Input, Dense,  GRU, Convolution1D, Reshape, Flatten, BatchNormalization, Dropout, GaussianDropout
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
import tensorflow as tf


def v_loss(R, v):
    return R - v


def a_loss(advance, a_t):
    # loss = advance + K.sum(a_t * K.log(1 / a_t))
    loss = tf.log(tf.matmul(a_t, advance, transpose_b=True))
    return loss


def build_graph(history_length, cross_num, cross_status, v_dim, a_dim):
    inputs = Input(shape=(cross_num, cross_status), name='inputs')

    v_hidden = Convolution1D(v_dim, cross_num, activation='relu')(inputs)
    v_values = GRU(v_dim, input_length=history_length, activation='linear',
                   W_regularizer=l2(0.01), name='v_values')(v_hidden)
    # print(v_values.get_shape())
    # a_hidden = Convolution1D(v_dim, cross_num, activation='tanh')(inputs)
    # a_probs = GRU(a_dim, input_length=15, activation='softmax', name='a_probs')(a_hidden)
    a_hidden = Flatten()(inputs)
    a_hidden = Dense(30, activation='relu', W_regularizer=l2(0.01))(a_hidden)
    a_hidden = Dropout(0.1)(a_hidden)
    a_hidden = Dense(30, activation='relu', W_regularizer=l2(0.01))(a_hidden)
    a_hidden = Dropout(0.1)(a_hidden)
    a_hidden = Dense(30, activation='relu', W_regularizer=l2(0.01))(a_hidden)
    # a_hidden = BatchNormalization()(a_hidden)
    a_probs = Dense(a_dim, activation='softmax', name='a_probs')(a_hidden)

    # print(a_probs.get_shape())

    ac_net = Model(input=inputs, output=[v_values, a_probs])
    return ac_net

