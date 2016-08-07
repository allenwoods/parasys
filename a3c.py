#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file       a3c.py
@author     Allen Woods
@date       2016-07-29
@version    16-7-29 下午3:30 ???
Some other Description
"""
import os
import threading
from multiprocessing import cpu_count, Process
from multiprocessing.pool import ThreadPool as Pool
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
import numpy as np
import time

from SumoEnv.simulation import SumoEnv
from Model.a3c_model import build_policy_and_value_networks
from keras import backend as K
from keras.optimizers import RMSprop
from SumoEnv.environ import TrafficSim
from test_async import DATA_DIR, CFG_DIR, NET_DIR, SUMMARY_DIR


# Path params
EXPERIMENT_NAME = "sumo_a3c"
SUMMARY_SAVE_PATH = SUMMARY_DIR + EXPERIMENT_NAME
CHECKPOINT_SAVE_PATH = os.path.join(NET_DIR, EXPERIMENT_NAME + ".ckpt")
CHECKPOINT_NAME = os.path.join(NET_DIR, EXPERIMENT_NAME + ".ckpt")
CHECKPOINT_INTERVAL = 5000
SUMMARY_INTERVAL = 5
# TRAINING = False
TRAINING = True

SHOW_TRAINING = True
# SHOW_TRAINING = False

# Experiment params
NODES = "1x1"
ACTIONS = 2
NUM_CONCURRENT = 1
NUM_EPISODES = 20000

AGENT_HISTORY_LENGTH = 15
CROSS_NUM = 1
CROSS_STATUS = 4

# DQN Params
GAMMA = 0.99

# Optimization Params
LEARNING_RATE = 0.00001

# Shared global parameters
T = 0
TMAX = 80000000
t_max = 15


def sample_policy_action(num_actions, probs):
    """
    Sample an action from an action probability distribution output by
    the policy network.
    """
    action_index = np.random.choice(range(num_actions), p=probs)
    return action_index


def actor_learner_thread(num, env, session, graph_ops, summary_ops, saver):
    # We use global shared counter T, and TMAX constant
    global TMAX, T

    # Unpack graph ops
    s, a, R, minimize, p_network, v_network = graph_ops

    # Unpack tensorboard summary stuff
    r_summary_placeholder, update_ep_reward, val_summary_placeholder, update_ep_val, summary_op = summary_ops

    # Wrap env with AtariEnvironment helper class
    # task_cfg_dir, i, x, y, gui = env
    # sumo = SumoEnv(task_cfg_dir, 'concurrent_%d' % i,
    #                xnumber=x, ynumber=y, gui=gui)
    # env = TrafficSim(sumo_env=sumo, cross_num=CROSS_NUM, cross_status=CROSS_STATUS,
    #                  agent_history_length=AGENT_HISTORY_LENGTH)
    # print("TrafficSim Thread:%d has created" % num)

    # time.sleep(5 * num)

    # Set up per-episode counters
    ep_reward = 0
    ep_avg_v = 0
    v_steps = 0
    ep_t = 0

    probs_summary_t = 0
    print("Thread %d initing..." % num)
    s_t = env.get_initial_state(AGENT_HISTORY_LENGTH)
    print("Thread %d init completed" % num)
    terminal = False

    while T < TMAX:
        s_batch = []
        past_rewards = []
        a_batch = []

        t = 0
        t_start = t

        while not (terminal or ((t - t_start) == t_max)):
            # Perform action a_t according to policy pi(a_t | s_t)
            probs = session.run(p_network, feed_dict={s: [s_t]})[0]
            action_index = sample_policy_action(ACTIONS, probs)
            a_t = np.zeros([ACTIONS])
            a_t[action_index] = 1

            if probs_summary_t % 100 == 0:
                print("P, ", np.max(probs), "a, ", np.argmax(probs),
                      "V ", session.run(v_network, feed_dict={s: [s_t]})[0][0])

            s_batch.append(s_t)
            a_batch.append(a_t)

            s_t1, r_t, terminal, info = env.step(action_index)
            ep_reward += r_t

            r_t = np.clip(r_t, -1, 1)
            past_rewards.append(r_t)

            t += 1
            T += 1
            ep_t += 1
            probs_summary_t += 1

            s_t = s_t1


        if terminal:
            R_t = 0
        else:
            R_t = session.run(v_network, feed_dict={s: [s_t]})[0][0]  # Bootstrap from last state

        R_batch = np.zeros(t)
        for i in reversed(range(t_start, t)):
            R_t = past_rewards[i] + GAMMA * R_t
            R_batch[i] = R_t

        print("R_batch")
        print(R_batch)
        print("a_batch")
        print(a_batch)
        print("s_batch")
        print(s_batch)
        session.run(minimize, feed_dict={R: R_batch,
                                         a: a_batch,
                                         s: s_batch})

        # Save progress every 5000 iterations
        if T % CHECKPOINT_INTERVAL == 0:
            saver.save(session, CHECKPOINT_SAVE_PATH, global_step=T)

        if terminal:
            # Episode ended, collect stats and reset game
            session.run(update_ep_reward, feed_dict={r_summary_placeholder: ep_reward})
            print("THREAD:", num, "/ TIME", T, "/ REWARD", ep_reward)
            s_t = env.get_initial_state(AGENT_HISTORY_LENGTH)
            terminal = False
            # Reset per-episode counters
            ep_reward = 0
            ep_t = 0


def build_graph():
    # Create shared global policy and value networks
    s, p_network, v_network, p_params, v_params = build_policy_and_value_networks(num_actions=ACTIONS,
                                                                                  agent_history_length=AGENT_HISTORY_LENGTH,
                                                                                  cross_num=CROSS_NUM,
                                                                                  cross_status=CROSS_STATUS)

    # Shared global optimizer
    # optimizer = RMSprop(lr=LEARNING_RATE)
    optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE)

    # Op for applying remote gradients
    R_t = tf.placeholder("float", [None])
    # a_t = tf.placeholder("float", [None, ACTIONS])
    a_t = tf.placeholder("float", shape=p_network.get_shape())
    # log_prob = tf.log(tf.reduce_sum(tf.mul(p_network, a_t), reduction_indices=1))
    # p_loss = -log_prob * (R_t - v_network)
    p_loss = -0.1 * (R_t - v_network)
    v_loss = tf.reduce_mean(tf.square(R_t - v_network))

    total_loss = p_loss + (0.5 * v_loss)

    minimize = optimizer.minimize(total_loss)
    return s, a_t, R_t, minimize, p_network, v_network


# Set up some episode summary ops to visualize on tensorboard.
def setup_summaries():
    episode_reward = tf.Variable(0.)
    tf.scalar_summary("Episode Reward", episode_reward)
    r_summary_placeholder = tf.placeholder("float")
    update_ep_reward = episode_reward.assign(r_summary_placeholder)
    ep_avg_v = tf.Variable(0.)
    tf.scalar_summary("Episode Value", ep_avg_v)
    val_summary_placeholder = tf.placeholder("float")
    update_ep_val = ep_avg_v.assign(val_summary_placeholder)
    summary_op = tf.merge_all_summaries()
    return r_summary_placeholder, update_ep_reward, val_summary_placeholder, update_ep_val, summary_op


def train(session, graph_ops, saver):
    # Set up game environments (one per thread)

    task_cfg_dir = os.path.join(CFG_DIR, 'a3c_train')
    # envs = [SumoEnv(task_cfg_dir, 'concurrent_%d' % i,
    #                 xnumber=1, ynumber=1, gui=False) for i in range(NUM_CONCURRENT)]

    # envs = [(task_cfg_dir, i, 1, 1, False) for i in range(NUM_CONCURRENT)]
    sumos = [SumoEnv(task_cfg_dir, 'concurrent_%d' % i,
                     xnumber=1, ynumber=1, gui=True) for i in range(NUM_CONCURRENT)]
    envs = [TrafficSim(sumo_env=sumos[i], cross_num=CROSS_NUM, cross_status=CROSS_STATUS,
                       agent_history_length=AGENT_HISTORY_LENGTH) for i in range(NUM_CONCURRENT)]
    summary_ops = setup_summaries()
    summary_op = summary_ops[-1]

    # Initialize variables
    session.run(tf.initialize_all_variables())
    writer = tf.train.SummaryWriter(SUMMARY_SAVE_PATH, session.graph)

    # Start NUM_CONCURRENT training threads
    actor_learner_threads = [threading.Thread(target=actor_learner_thread,
                                              args=(thread_id, envs[thread_id], session, graph_ops, summary_ops, saver))
                             for thread_id in range(NUM_CONCURRENT)]
    # actor_learner_threads = [Process(target=actor_learner_thread,
    #                                  args=(thread_id, envs[thread_id], session, graph_ops, summary_ops, saver))
    #                          for thread_id in range(NUM_CONCURRENT)]
    # actor_learner_pool = Pool()
    # actor_learner_pool.starmap(actor_learner_thread, [(thread_id, envs[thread_id], session, graph_ops, summary_ops, saver)
    #                                               for thread_id in range(NUM_CONCURRENT)])
    for t in actor_learner_threads:
        t.start()
    # actor_learner_pool.close()
    # Show the agents training and write summary statistics
    last_summary_time = 0
    while True:
        # if SHOW_TRAINING:
        #     for env in envs:
        #         env.render()
        now = time.time()
        if now - last_summary_time > SUMMARY_INTERVAL:
            summary_str = session.run(summary_op)
            writer.add_summary(summary_str, float(T))
            last_summary_time = now
    for t in actor_learner_threads:
        t.join()
    # actor_learner_pool.join()


def evaluation(session, graph_ops, saver):
    pass
    # saver.restore(session, CHECKPOINT_NAME)
    # print("Restored model weights from ", CHECKPOINT_NAME)
    # task_cfg_dir = os.path.join(CFG_DIR, 'a3c_evaluate')
    # monitor_env = SumoEnv(task_cfg_dir, 'evaluate', xnumber=1, ynumber=1, gui=False)
    # # monitor_env = gym.make(GAME)
    # # monitor_env.monitor.start('/tmp/' + EXPERIMENT_NAME + "/eval")
    #
    # # Unpack graph ops
    # s, a_t, R_t, learning_rate, minimize, p_network, v_network = graph_ops
    #
    # # Wrap env with AtariEnvironment helper class
    # env = TrafficSim(sumo_env=monitor_env, cross_num=CROSS_NUM, cross_status=CROSS_STATUS,
    #                  agent_history_length=AGENT_HISTORY_LENGTH)
    #
    # for i_episode in range(100):
    #     s_t = env.get_initial_state(AGENT_HISTORY_LENGTH)
    #     ep_reward = 0
    #     terminal = False
    #     while not terminal:
    #         monitor_env.render()
    #         # Forward the deep q network, get Q(s,a) values
    #         probs = p_network.eval(session=session, feed_dict={s: [s_t]})[0]
    #         action_index = sample_policy_action(ACTIONS, probs)
    #         s_t1, r_t, terminal, info = env.step(action_index)
    #         s_t = s_t1
    #         ep_reward += r_t
    #     print(ep_reward)
    # monitor_env.monitor.close()


def main(_):
    g = tf.Graph()
    with g.as_default(), tf.Session() as session:
        K.set_session(session)
        graph_ops = build_graph()
        saver = tf.train.Saver()

        if TRAINING:
            train(session, graph_ops, saver)
        else:
            evaluation(session, graph_ops, saver)


if __name__ == "__main__":
    os.environ.setdefault('SUMO_HOME', '/usr/share/sumo')
    tf.app.run()
