#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file       a3c_model.py
@author     Allen Woods
@date       2016-07-29
@version    16-7-29 下午3:29 ???
Some other Description
"""
import tensorflow as tf
from keras.layers import Convolution2D, LSTM, GRU, Flatten, Dense, Input, Reshape
from keras.models import Model


def build_policy_and_value_networks(num_actions, agent_history_length, cross_num, cross_status):
    # with tf.device("/cpu:0"):
    state = tf.placeholder("float", [None, agent_history_length, cross_num, cross_status])

    inputs = Input(name='Input', shape=(cross_num, cross_status,))

    action_hidden = LSTM(cross_num*cross_status, name='GRU', input_length=agent_history_length, activation='relu')(inputs)
    action_hidden = Reshape((1, cross_num, cross_status,), name='Reshape')(action_hidden)
    action_hidden = Convolution2D(16, cross_num, cross_status, name='Conv2D', activation='relu')(action_hidden)
    action_hidden = Flatten(name='Flatten')(action_hidden)
    action_probs = Dense(name="action_probs", output_dim=num_actions, activation='softmax')(action_hidden)

    state_hidden = LSTM(cross_num*cross_status, name='GRU', input_length=agent_history_length, activation='relu')(inputs)
    state_hidden = Reshape((1, cross_num, cross_status,), name='Reshape')(state_hidden)
    state_hidden = Convolution2D(16, cross_num, cross_status, name='Conv2D', activation='relu')(state_hidden)
    state_hidden = Flatten(name='Flatten')(state_hidden)
    state_value = Dense(name="state_value", output_dim=1, activation='linear')(state_hidden)

    policy_network = Model(input=inputs, output=action_probs)
    value_network = Model(input=inputs, output=state_value)

    p_params = policy_network.trainable_weights
    v_params = value_network.trainable_weights

    # print(type(state))
    p_out = policy_network(state)
    v_out = value_network(state)

    return state, p_out, v_out, p_params, v_params


if __name__ == '__main__':
    pass
