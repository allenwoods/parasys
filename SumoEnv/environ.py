#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file       environ.py
@author     Allen Woods
@date       2016-07-29
@version    16-7-29 下午2:51 ???
SUMO simulation environment
"""
from .create_cfg import SumoCfg
from .simulation import TraciEnv
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
from collections import deque


class SumoEnv(object):
    """
    Small wrapper for gym atari environments.
    Responsible for preprocessing screens and holding on to a screen buffer
    of size agent_history_length from which environment state
    is constructed.
    """

    def __init__(self, sumo_env, cross_num, cross_status, agent_history_length):
        self.env = sumo_env
        self.cross_num = cross_num
        self.cross_status = cross_status
        self.agent_history_length = agent_history_length

        self.tls_actions = range(2 ** len(sumo_env.tls)) # Action space is 2**tls_num

        # Screen buffer of size AGENT_HISTORY_LENGTH to be able
        # to build state arrays of size [1, AGENT_HISTORY_LENGTH, cross_num, cross_status]
        self.state_buffer = deque()

    def get_initial_state(self):
        """
        Resets the atari game, clears the state buffer
        """
        # Clear the state buffer
        self.state_buffer = deque()

        x_t = self.env.reset()
        x_t = self.get_preprocessed_frame(x_t)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)

        for i in range(self.agent_history_length - 1):
            self.state_buffer.append(x_t)
        return s_t

    def get_preprocessed_frame(self, observation):
        """
        See Methods->Preprocessing in Mnih et al.
        1) Get image grayscale
        2) Rescale image
        """
        return resize(rgb2gray(observation), (self.cross_num, self.cross_status))

    def step(self, action_index):
        """
        Excecutes an action in the gym environment.
        Builds current state (concatenation of agent_history_length-1 previous frames and current one).
        Pops oldest frame, adds current frame to the state buffer.
        Returns current state.
        """

        x_t1, r_t, terminal, info = self.env.step(self.tls_actions[action_index])
        x_t1 = self.get_preprocessed_frame(x_t1)

        previous_frames = np.array(self.state_buffer)
        s_t1 = np.empty((self.agent_history_length, self.cross_status, self.cross_num))
        s_t1[:self.agent_history_length - 1, ...] = previous_frames
        s_t1[self.agent_history_length - 1] = x_t1

        # Pop the oldest frame, add the current frame to the queue
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)

        return s_t1, r_t, terminal, info


if __name__ == '__main__':
    pass
