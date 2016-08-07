#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file       ac_model.py
@author     Allen Woods
@date       2016-08-07
@version    16-8-7 下午3:57 ???
Some other Description
"""
from keras.layers import Input, Dense, LSTM, Highway, GRU, Convolution1D, Convolution2D, TimeDistributed, Reshape, \
    Flatten, Embedding
from keras.models import Model
from keras import backend as K
import tensorflow as tf


def v_loss(R, v):
    return R - v


def a_loss(advance, a_probs):
    loss = K.log(K.max(a_probs)) * advance + K.sum(a_probs * K.log(1 / a_probs))
    print(loss)
    return loss


def build_graph(history_length, cross_num, cross_status, v_dim, a_dim):
    inputs = Input(shape=(history_length, cross_num, cross_status), name='inputs')

    hidden = TimeDistributed(Convolution1D(v_dim, cross_num, activation='relu'))(inputs)
    hidden = Reshape((history_length, v_dim))(hidden)

    v_values = GRU(v_dim, input_length=history_length, activation='linear', name='v_values')(hidden)
    # print(v_values.get_shape())
    a_probs = GRU(a_dim, input_length=15, activation='softmax', name='a_probs')(hidden)
    # print(a_probs.get_shape())

    ac_net = Model(input=inputs, output=[v_values, a_probs])
    return ac_net


class Main(object):
    def __init__(self):
        pass


if __name__ == '__main__':
    pass
