#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file       q_model.py
@author     Allen Woods
@date       2016-08-01
@version    16-8-1 下午9:59 ???
Some other Description
"""

import tensorflow as tf
from keras.layers import Convolution2D, Flatten, Dense, Input
from keras.models import Model


def build_network(num_actions, agent_history_length, resized_width, resized_height):
    state = tf.placeholder("float", [None, agent_history_length, resized_width, resized_height])
    inputs = Input(shape=(agent_history_length, resized_width, resized_height,))
    model = Convolution2D(nb_filter=16, nb_row=8, nb_col=8, subsample=(4, 4), activation='relu', border_mode='same')(
        inputs)
    model = Convolution2D(nb_filter=32, nb_row=4, nb_col=4, subsample=(2, 2), activation='relu', border_mode='same')(
        model)
    model = Flatten()(model)
    model = Dense(output_dim=256, activation='relu')(model)
    q_values = Dense(output_dim=num_actions, activation='linear')(model)
    m = Model(input=inputs, output=q_values)
    return state, m


if __name__ == '__main__':
    pass
