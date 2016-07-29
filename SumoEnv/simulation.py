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
import socket
import pandas as pd
from SumoEnv.mission import lazy_police
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
                 rouprob=10, epoch_steps=3600, cocurrent=8):
        self.data_dir = data_dir
        self.task_name = task_name
        self.xnumber = xnumber
        self.ynumber = ynumber
        self.xlength = xlength
        self.ylength = ylength
        self.net_type = net_type
        self.tls_type = tls_type
        self.rouprob = rouprob
        self.epoch_steps = epoch_steps
        self.sumo_env = None
        self.traci_env = None
        self.ports = [get_free_port() for i in range(cocurrent)]
        self.current_step = 1

    def reset(self):
        self.sumo_env = SumoCfg(self.data_dir, self.task_name + str(self.current_step),
                                self.xnumber, self.ynumber,
                                self.xlength, self.ylength, self.net_type, self.tls_type,
                                self.rouprob, self.epoch_steps)
        self.current_step += 1


class TraciEnv:
    def __init__(self, port, host='localhost', label='default', verbose=False):
        self.port = port
        self.host = host
        # self.output_dir = output_dir
        self.label = label
        self.directions = ['North', 'East', 'South', 'West']
        self.verbose = verbose

    def run(self, update_steps=1, strategy='static'):
        traci.start(self.port, host=self.host)
        step = 1
        self.tls = [TrafficLight(t) for t in trafficlights.getIDList()]
        log_list = []
        while simulation.getMinExpectedNumber() > 0:
            log_list.append(self.sim_step(update_steps, strategy))
            print('Simulation Step: %d' % step)
            step += 1
        log = [item for sublist in log_list for item in sublist]  # Flatten the log
        log = self.log_to_pd(log)
        return log

    @staticmethod
    def log_to_pd(log):
        return pd.DataFrame(log, columns=('step', 'traffic_light', 'direction',
                                          "traffic_light_status", 'halting_number',
                                          'waiting_time'))

    @staticmethod
    def close():
        traci.close()
        sys.stdout.flush()

    def sim_step(self, update_steps, strategy):
        traci.simulationStep()
        current_step = int(simulation.getCurrentTime() / 1000)
        log_list = []
        for t in self.tls:
            log_list.append(t.update(current_step))
            if self.verbose:
                t.show()
            if strategy == 'lazy_police':
                lazy_police(update_steps, t, self.directions)
        log = [item for sublist in log_list for item in sublist]  # Flatten the log
        return log


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

    def set_phase(self, pass_way, green_0='gg', green_1='ggg', red_0='rr', red_1='rrr'):
        green_phase = green_0 + green_1
        red_phase = red_0 + red_1
        if pass_way == 'North' or pass_way == 'South':
            phase = green_phase + red_phase + green_phase + red_phase  # N:g E:r S:g W:r
            trafficlights.setRedYellowGreenState(self.tls_id, phase)
        else:
            phase = red_phase + green_phase + red_phase + green_phase  # N:r E:g S:r W:g
            trafficlights.setRedYellowGreenState(self.tls_id, phase)

    def update(self, current_step):
        t_status = trafficlights.getRedYellowGreenState(self.tls_id)
        self.tls_status = dict(zip(self.directions,
                                   [(t_status[i:i + 2], t_status[i + 2:i + 5]) for i in range(0, len(t_status), 5)]))
        self.N_edge.update()
        self.E_edge.update()
        self.S_edge.update()
        self.W_edge.update()
        logger = self.log(current_step)
        return logger

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
        'halting_number',"waiting_number'
        :param current_step:
        :return:
        """
        logger = []
        for d in self.directions:
            logger.append([current_step, self.tls_id, d,
                           self.tls_status[d],
                           self.edges[d].edge_status['halt'],
                           self.edges[d].edge_status['wait']])
        return logger


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
