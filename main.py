#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file       main.py
@author     Allen Woods
@date       2016-08-07
@version    16-8-7 下午4:38 ???
Some other Description
"""
import os

os.environ.setdefault('SUMO_HOME', '/usr/share/sumo')
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
import sys
from multiprocessing import cpu_count, Pool
from multiprocessing.pool import ThreadPool
import tensorflow as tf
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import socket
import numpy as np
import pandas as pd
from SumoEnv.create_cfg import SumoCfg
from SumoEnv.simulation import TraciEnv, SumoEnv
from SumoEnv.environ import TrafficSim
from time import strftime as current_time
from Model.ac_model import a_loss, v_loss, build_graph
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

DATA_DIR = os.path.join(os.getcwd(), 'data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
CFG_DIR = os.path.join(DATA_DIR, 'cfg')
NET_DIR = os.path.join(DATA_DIR, 'network')
SUMMARY_DIR = os.path.join(DATA_DIR, 'summary')
TMAX = 10000  # Total run of simulation
tmax = 15  # update period

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


def get_free_port(length=1):
    ss = [socket.socket() for i in range(length)]
    port = []
    for s in ss:
        s.bind(('', 0))
        port.append(s.getsockname()[1])
        s.close()
    if len(port) == 1:
        port = port[0]
    return port


def init_sumo(cfg_dir, net_name, init_time, xnumber, ynumber):
    try:
        traci.close()
    except KeyError:
        print("Traci is not Running!")
    sumo_cfg = SumoCfg(cfg_dir, net_name, xnumber, ynumber)
    sumo_cfg.make()
    return sumo_cfg.get_start_cmd(None, init_time)[0]
    # traci.start(sumo_cfg.get_start_cmd(None, init_time)[0])


def egreedy(a_space, t, current_ix):
    e = np.max([0.001, 1 - t / 1000])
    weights = [e, 1 - e]
    a = [np.random.choice(a_space), current_ix]
    ix = np.random.choice(2, p=weights)
    a_t = a[ix]
    return a_t


def state_reward_generator(s_t, env, net, T, index='halt'):
    # env = SumoEnv(get_free_port())
    gamma = 0.6
    r_t = 0
    S = []
    R = []
    A = []
    next_s0 = None
    step = 0
    terminate = False
    actions = env.actions
    while step < 15:
        if not terminate:
            S.append(s_t)
            a_probs = net.predict(s_t)[1]
            print('a_probs')
            print(a_probs)
            current_at = np.argmax(a_probs)
            a_t = egreedy(env.action_space_n, T + step, current_at)
            # a_p = a_probs[0, a_t]
            log, r_t1, terminate, info = env.step(actions[a_t])
            r_t1 = np.mean(r_t1)
            s_t = env.parse_log(log, index)
            v_t = net.predict(s_t)[0]
            print('v_t')
            print(v_t)
            r_t += gamma * r_t1
            print('r_t')
            print(r_t1)
            advance = np.zeros((1, env.action_space_n))
            advance[0, a_t] = np.array(np.subtract(r_t, v_t)).reshape(1, )
            # advance = np.array(np.max([np.log(a_p)*np.subtract(r_t, v_t), 0.01])).reshape(1,)
            print('advance')
            print(advance)
            print(np.subtract(r_t, v_t))
            R.append(r_t.reshape(1))
            A.append(advance)
            next_s0 = s_t
        else:
            S.append(np.zeros(env.xnumber * env.ynumber, 4))
            R.append(np.zeros(1))
            A.append(np.zeros(1))
            next_s0 = None
        step += 1
    if np.min(advance) < 0:
        terminate = True
    return S, A, R, next_s0, terminate


def parse_log(log, index='halt'):
    target = {'halt': 4, 'wait': 5}[index]
    cross_num = int(len(log) / 4)
    states = []
    for i in range(cross_num):
        cross_log = log[4 * i: 4 * (i + 1)]
        states.append([item[target] for item in cross_log])
    return np.array(states)


def train_ac_net(env, ac_net, tf_log_dir):
    terminate = False
    next_s0 = env.reset()
    while not terminate:
        sim_step = traci.simulation.getCurrentTime() / 1000
        S, A, R, next_s0, terminate = state_reward_generator(next_s0, env, ac_net, sim_step)
        for h in zip(S, A, R):
            ac_net.fit(h[0], {'a_probs': h[1], 'v_values': h[2]},
                       batch_size=1, nb_epoch=1, callbacks=[])


def evaluate_ac_net_sync(env, ac_net, log_dir):
    # env = SumoEnv()
    start_cmd = env.reset(phase='Test')
    traciPool = Pool()
    runPool = ThreadPool()
    tasks = ['static', 'ac_net']
    terminated = False
    for t in tasks:
        traci.start(start_cmd, label=t)
    step = 0
    while not terminated and step < 4500:
        result = runPool.map(get_traci_process, traciPool.map(traci_step, tasks))
        # print(result)
        halts = [r[0] for r in result]
        terminated = sum([r[1] for r in result]) == sum(range(len(tasks))) + 1  # if both process is terminated
        print(halts)
        step += 1


def evaluate_ac_net(env, net, log_file, current_epoch):
    actions = env.actions
    s = env.reset()
    edges = traci.edge.getIDList()
    static_halt = 0
    step = 0
    try:
        summary = pd.read_csv(log_file)
    except OSError as err:
        summary = pd.DataFrame()
        summary['epoch'] = 0
        summary['static'] = 0
        summary['net'] = 0
    while traci.simulation.getMinExpectedNumber() > 0 and step < env.epoch_steps:
        log, r_t1, terminate, info = env.step()
        static_halt += sum([traci.edge.getLastStepHaltingNumber(e) for e in edges])
        print('static_halt: %d at step %d' % (static_halt, step))
        step += 1
    s = env.reset(phase='Test')
    net_halt = 0
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0 and step < env.epoch_steps:
        a_probs = net.predict(s)[1]
        a_t = np.argmax(a_probs)
        log, r_t1, terminate, info = env.step(actions[a_t])
        s = env.parse_log(log, 'halt')
        net_halt += sum([traci.edge.getLastStepHaltingNumber(e) for e in edges])
        print('net_halt: %d at step %d' % (net_halt, step))
        step += 1
    summary.loc[len(summary)] = [current_epoch, static_halt, net_halt]
    summary.to_csv(log_file, index=False)


def traci_step(label):
    print('sumo label: %s' % label)
    traci.switch(label)
    traci.simulationStep()
    edges = traci.edge.getIDList()
    halts = [traci.edge.getLastStepHaltingNumber(e) for e in edges]
    terminated = traci.simulation.getMinExpectedNumber()
    return halts, terminated


def get_traci_process(process_return):
    halt = process_return[0]
    min_expected = process_return[1]
    if min_expected > 0:
        terminate = False
    else:
        terminate = True
    return halt, terminate


def train(cfg_dir, summary_dir, net_dir, xnumber, ynumber,
          history_length, cross_num, cross_status, v_dim, a_dim, gui=False):
    sim_env = SumoEnv(cfg_dir, current_time('%Y%m%d%H%M%S'), xnumber, ynumber, gui=gui)
    ac_net = build_graph(history_length, cross_num, cross_status, v_dim, a_dim)
    ac_net.compile(optimizer='rmsprop', loss={'v_values': v_loss, 'a_probs': a_loss})
    train_ac_net(sim_env, ac_net, summary_dir)
    ac_net.save_weights(os.path.join(net_dir, current_time('%Y%m%d%H%M%S') + '.h5'))
    sim_env.close()


def evaluate(cfg_dir, summary_dir, net_dir, xnumber, ynumber,
             history_length, cross_num, cross_status, v_dim, a_dim, gui=False):
    net_files = sorted(os.listdir(net_dir))
    with tf.device('/cpu:0'):
        ac_net = build_graph(history_length, cross_num, cross_status, v_dim, a_dim)
    log_file = os.path.join(summary_dir, 'evaluation.csv')
    epoch = 0
    for f in net_files:
        print(f)
        net_weights = os.path.join(net_dir, f)
        ac_net.load_weights(net_weights)
        sim_env = SumoEnv(cfg_dir, str(epoch), xnumber, ynumber, gui=gui)
        evaluate_ac_net(sim_env, ac_net, log_file, epoch)
        epoch += 1


if __name__ == '__main__':
    xnumber = 1
    ynumber = 1
    cross_num = xnumber * ynumber
    cross_status = 4
    history_length = 15
    v_dim = 1
    a_dim = 2 ** cross_num
    T = 0
    TMAX = 10000
    phase = 'TEST'
    task_cfg_dir = os.path.join(CFG_DIR, 'ac')
    task_net_dir = os.path.join(NET_DIR, 'ac')
    task_summary_dir = os.path.join(SUMMARY_DIR, 'ac')
    if not os.path.isdir(task_net_dir):
        os.makedirs(task_net_dir)
    if not os.path.isdir(task_summary_dir):
        os.makedirs(task_summary_dir)

    if phase is 'TRAIN':
        while T < TMAX:
            train(task_cfg_dir, task_summary_dir, task_net_dir,
                  xnumber, ynumber, history_length, cross_num, cross_status, v_dim, a_dim)
            T += 1
    else:
        evaluate(task_cfg_dir, task_summary_dir, task_net_dir,
                 xnumber, ynumber, history_length, cross_num, cross_status, v_dim, a_dim,
                 gui=False)
