#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file       environ.py
@author     Allen Woods
@date       2016-07-29
@version    16-7-29 下午2:51 ???
SUMO simulation environment
"""
import numpy as np
from collections import deque
from debug.logger import timeit


class TrafficSim(object):
    """
    Small wrapper for gym atari environments.
    Responsible for preprocessing screens and holding on to a screen buffer
    of size agent_history_length from which environment state
    is constructed.
    """

    def __init__(self, sumo_env, cross_num=1, cross_status=4,
                 agent_history_length=15, thread_label=None):
        self.env = sumo_env
        self.cross_num = cross_num
        self.cross_status = cross_status
        self.agent_history_length = agent_history_length
        self.thread_label = str(thread_label)
        self.tls_actions = sumo_env.actions  # Action space is 2**tls_num

        # Screen buffer of size AGENT_HISTORY_LENGTH to be able
        # to build state arrays of size [1, AGENT_HISTORY_LENGTH, cross_num, cross_status]
        self.state_buffer = deque()
        self.sumo = None
        self.traci = None

    def reset_sumo(self):
        self.sumo =self.env.reset()

    def get_initial_state(self, history_length):
        """
        Resets SUMO, clears the state buffer
        """
        # Clear the state buffer
        self.state_buffer = deque()
        x_t, r_t, terminal, i = self.env.step(self.thread_label)
        self.cross_num = len(self.env.tls)
        self.cross_status = len(self.env.traci_env.directions)
        x_t = self.get_preprocessed_status(x_t)
        # print(x_t)
        s_t = np.stack(([x_t for i in range(history_length)]), axis=0)

        for i in range(self.agent_history_length - 1):
            self.state_buffer.append(x_t)
        return s_t

    def get_preprocessed_status(self, observation, index='halt'):
        """
        Get Status from Traci log
        :param index:
        :param observation: list, each one contain [step, tls_id, direction,
                            light_status, halting number, waiting time]
        :return: x_t, np.array, shape=(tls, directions, status);
                      For example, in a 3x3 net, shape=(9, 4, 2)

        """
        if index == 'halt':
            x_index = -2
        elif index == 'wait':
            x_index = -1
        else:
            raise ValueError('Index should be either \'halt\' or \'wait\'')
        x_t = np.array([x[x_index] for x in observation])
        shape = x_t.shape
        x_t = x_t.reshape((int(shape[0] / self.cross_status), self.cross_status))
        return x_t

    def step(self, action_index=None):
        """
        Excecutes an action in the sumo environment.
        Builds current state (concatenation of agent_history_length-1 previous frames and current one).
        Pops oldest frame, adds current frame to the state buffer.
        Returns current state.
        """
        if action_index == None:
            action = None
        else:
            action = self.tls_actions[action_index]
        x_t1, r_t, terminal, info = self.env.step(action)
        r_t = np.mean(r_t)  # !!!!!!Using mean reward of all tls reward!!!!!
        x_t1 = self.get_preprocessed_status(x_t1)
        # print(x_t1.shape)
        s_shape = tuple([self.agent_history_length]) + x_t1.shape
        # print(s_shape)
        previous_frames = np.array(self.state_buffer)
        s_t1 = np.empty(s_shape)
        s_t1[:self.agent_history_length - 1, ...] = previous_frames
        s_t1[self.agent_history_length - 1] = x_t1

        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)

        return s_t1, r_t, terminal, info


if __name__ == '__main__':
    pass
