#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file       environ.py
@author     Allen Woods
@date       2016-07-29
@version    16-7-29 下午2:51 ???
SUMO simulation environment
"""
import os
import sys
import time
import socket
from subprocess import Popen, PIPE
import pandas as pd
from time import strftime as current_time
from itertools import product
from .create_cfg import SumoCfg

sumo_root = os.environ.get('SUMO_HOME')

try:
    sumo_home = os.path.join(sumo_root, 'tools')
    sys.path.append(sumo_home)
    from sumolib import checkBinary
    import traci
    from traci import simulationStep
    from traci import simulation
    from traci import trafficlights
    from traci import lane
except ImportError:
    sys.exit(
        'Please declare environment variable \'SUMO_HOME\' as the root directory '
        'of your sumo installation (it should contain folders \'bin\', \'tools\' and \'docs\')')


def get_free_port():
    s = socket.socket()
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()
    return port


class SumoEnv:
    def __init__(self, data_dir, task_name, xnumber=1, ynumber=1,
                 xlength=1000, ylength=1000, net_type='grid', tls_type='static',
                 rouprob=10, epoch_steps=3600, gui=False, port=None):
        self.data_dir = data_dir
        self.task_name = task_name
        self.thread_label = task_name
        self.xnumber = xnumber
        self.ynumber = ynumber
        self.xlength = xlength
        self.ylength = ylength
        self.net_type = net_type
        self.tls_type = tls_type
        self.rouprob = rouprob
        self.epoch_steps = epoch_steps
        self.sumo_cfg = None
        self.traci_env = None
        if port is None:
            self.port = str(get_free_port())
        else:
            self.port = str(port)
        self.current_epoch = 1
        self.gui = gui
        self.tls = []
        self.actions, self.action_space_n = self.get_action_space()

    def get_action_space(self):
        direction = ['North', 'East']
        actions = direction * (self.xnumber * self.ynumber)
        return actions, len(actions)

    def reset(self):
        mission_start_time = current_time('%Y%m%d%H%M%S')
        if self.traci_env is not None:
            self.traci_env.close()
        self.sumo_cfg = SumoCfg(self.data_dir, self.task_name + str(self.current_epoch),
                                self.xnumber, self.ynumber,
                                self.xlength, self.ylength, self.net_type, self.tls_type,
                                self.rouprob, self.epoch_steps)
        print("Sumo_cfg created")
        self.sumo_cfg.make()
        sumoCMD, run_env = self.sumo_cfg.get_start_cmd(self.port, mission_start_time, gui=self.gui)
        print("Sumo Process Raised")
        # time.sleep(3)
        print("Try Raise Traci")
        self.traci_env = TraciEnv(self.port, label=self.thread_label)
        print("Get here")
        self.traci_env.start(sumoCMD, run_env)
        # traci.switch(self.thread_label)
        # traci.init(self.port)
        # time.sleep(3)
        self.tls = self.traci_env.tls
        self.current_epoch += 1
        return sumoCMD

    def step(self, action=None):
        traci.switch(self.thread_label)
        log, reward_list = self.traci_env.sim_step(action)
        terminate = self.traci_env.is_terminated()
        info = None
        return log, reward_list, terminate, info


class TraciEnv:
    def __init__(self, port, label='default', verbose=False):
        self.port = port
        # self.output_dir = output_dir
        self.label = label
        self.directions = ['North', 'East', 'South', 'West']
        self.verbose = verbose

    def start(self, cmd, envs):
        # traci.start(cmd, port=int(self.port), label=self.label)
        # time.sleep(5)
        # print(cmd + ['--remote-port', self.port])
        sumoProcess = Popen(cmd + ['--remote-port', self.port],
                            env=envs, stdout=PIPE, stderr=PIPE)
        time.sleep(3)
        print("SUMO PROCESS RAISED!")
        traci.init(port=int(self.port), label=self.label)
        # step = 1
        self.tls = [TrafficLight(t) for t in trafficlights.getIDList()]
        # log_list = []
        # while simulation.getMinExpectedNumber() > 0:
        #     log_list.append(self.sim_step(update_steps, strategy))
        #     print('Simulation Step: %d' % step)
        #     step += 1
        # log = [item for sublist in log_list for item in sublist]  # Flatten the log
        # log = self.log_to_pd(log)
        # return log

    @staticmethod
    def log_to_pd(log):
        return pd.DataFrame(log, columns=('step', 'traffic_light', 'direction',
                                          "traffic_light_status", 'halting_number',
                                          'waiting_time'))

    @staticmethod
    def close():
        traci.close()
        sys.stdout.flush()

    def sim_step(self, action):
        if action is not None:
            for i in range(len(self.tls)):
                self.tls[i].set_passway(action[i])
        traci.simulationStep()
        current_step = int(simulation.getCurrentTime() / 1000)
        print("Current Step: %d" % current_step)
        log_list = []
        reward_list = []
        for t in self.tls:
            t_log, t_reward = t.update(current_step)
            log_list.append(t_log)
            reward_list.append(t_reward)
            if self.verbose:
                t.show()
        log = [item for sublist in log_list for item in sublist]  # Flatten the log
        return log, reward_list

    @staticmethod
    def is_terminated():
        if simulation.getMinExpectedNumber() > 0:
            return False
        else:
            return True


class TrafficLight:
    def __init__(self, tls_id):
        self.tls_id = tls_id
        self.controlled_lanes = trafficlights.getControlledLanes(tls_id)
        self.directions = ['North', 'East', 'South', 'West']
        # Each edge(one direction) has 5 lanes,
        # two lanes as edge_0, another three as edge_1
        # edge_0_0(lane index:0) is trun-right lane
        # edge_1_1(lane index:3) is turn-left lane
        # edge_1_2(lane index:4) is turn-around lane
        # edge_0_1, edge_1_0 is straight lane
        self.N_edge = Edge(self.controlled_lanes[1], self.controlled_lanes[3])  # North edge_0, edge_1
        self.E_edge = Edge(self.controlled_lanes[6], self.controlled_lanes[8])
        self.S_edge = Edge(self.controlled_lanes[11], self.controlled_lanes[13])
        self.W_edge = Edge(self.controlled_lanes[16], self.controlled_lanes[18])
        self.edges = dict(zip(self.directions,
                              [self.N_edge, self.E_edge, self.S_edge, self.W_edge]))
        self.tls_status = dict()

    def set_passway(self, pass_way, green_0='gg', green_1='ggg', red_0='gr', red_1='rrr'):
        green_phase = green_0 + green_1
        red_phase = red_0 + red_1
        if pass_way == 'North' or pass_way == 'South':
            phase = green_phase + red_phase + green_phase + red_phase  # N:g E:r S:g W:r
            trafficlights.setRedYellowGreenState(self.tls_id, phase)
        elif pass_way == 'East' or pass_way == 'West':
            phase = red_phase + green_phase + red_phase + green_phase  # N:r E:g S:r W:g
            trafficlights.setRedYellowGreenState(self.tls_id, phase)
        else:
            raise ValueError("Pass way should be either North, East, South or West")

    def update(self, current_step):
        t_status = trafficlights.getRedYellowGreenState(self.tls_id)
        self.tls_status = dict(zip(self.directions,
                                   [(t_status[i:i + 2], t_status[i + 2:i + 5]) for i in range(0, len(t_status), 5)]))
        self.N_edge.update()
        self.E_edge.update()
        self.S_edge.update()
        self.W_edge.update()
        logger, reward = self.log(current_step)
        return logger, reward

    def show(self):
        print('Traffic Light: %s' % self.tls_id)
        for d in self.directions:
            print('%s\n Traffic Light:%s\n Edge Status: halt number:%s waiting: %f' %
                  (d, self.tls_status[d],
                   self.edges[d].edge_status['halt'],
                   self.edges[d].edge_status['wait']))

    def log(self, current_step):
        """
        Log 'step', 'traffic_light', 'direction', "traffic_light_status",
        'halting_number',"waiting_time'
        :param current_step:
        :return:
        """
        logger = []
        for d in self.directions:
            logger.append([current_step, self.tls_id, d,
                           self.tls_status[d],
                           self.edges[d].edge_status['halt'],
                           self.edges[d].edge_status['wait']])
        max_ns = max([self.edges['North'].edge_status['halt'], self.edges['South'].edge_status['halt']])
        max_ew = max([self.edges['East'].edge_status['halt'], self.edges['West'].edge_status['halt']])
        reward = abs(max_ns - max_ew)
        return logger, reward


class Edge:
    """
    Class Edge is an edge has one direction, two lanes
    """

    def __init__(self, lane_0_id, lane_1_id):
        self.lane_0_id = lane_0_id
        self.lane_1_id = lane_1_id
        self.update()

    def update(self):
        self.lane_0_status = self.get_lane_status(self.lane_0_id)
        self.lane_1_status = self.get_lane_status(self.lane_1_id)
        self.edge_status = {'halt': self.lane_0_status['halt'] + self.lane_1_status['halt'],
                            'wait': self.lane_0_status['wait'] + self.lane_1_status['wait']}

    @staticmethod
    def get_lane_status(laneid):
        halt_number = lane.getLastStepHaltingNumber(laneid)
        waiting_time = lane.getWaitingTime(laneid)
        return {'halt': halt_number, 'wait': waiting_time}
