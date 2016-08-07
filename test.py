#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file       test.py
@author     Allen Woods
@date       2016-08-07
@version    16-8-7 下午3:59 ???
Some other Description
"""
import numpy as np
from multiprocessing import Process, Pool, cpu_count
from multiprocessing.pool import ThreadPool
from functools import partial
from Model.ac_model import a_loss, v_loss, build_graph

def func():
    pass


class Main(object):
    def __init__(self):
        pass

def actor_learner_init(ac_net_thread):
    ac_net_thread.compile(optimizer='rmsprop',loss={'v_values': v_loss , 'a_probs': a_loss})

def actor_learner_thread(net_thread, thread_id, samples, rewards):
    # ac_net_thread.compile(optimizer='rmsprop',loss={'v_values': v_loss , 'a_probs': a_loss})
    print('Start Learner %d'%thread_id)
    s_t = samples[thread_id].reshape(1,15,9,4)
    r_t = rewards[thread_id]
    s_t1 = net_thread.predict(s_t)[0]
    advance = r_t - s_t1
    r_t = np.array([[r_t]])
    print('Learner %d Fitting...'%thread_id)
    net_thread.fit(s_t, {'a_probs':advance, 'v_values':r_t}, verbose=0)
    return net_thread.get_weights()

def update_main_weights(net_thread_weights, old_W, net_main):
    delta = np.subtract(net_thread_weights, old_W)
    new_W = np.add(net_main.get_weights(), delta)
    net_main.set_weights(new_W)
    print('Main net updated')
    print(net_main.get_weights())

if __name__ == '__main__':
    concurrent = cpu_count()
    samples = [np.random.random([15, 9, 4]) for i in range(concurrent)]  # Example State
    rewards = [np.random.random() for i in range(concurrent)]
    cross_num = 9
    cross_status = 4
    history_length = 15
    v_dim = 1
    a_dim = 2 ** cross_num

    ac_net_main = build_graph(history_length, cross_num, cross_status, v_dim, a_dim)
    ac_net_threads = [build_graph(history_length, cross_num, cross_status, v_dim, a_dim) for i in range(concurrent)]

    # actor_pool = ThreadPool()
    for i in range(concurrent):
        actor_learner_init(ac_net_threads[i])

    old_w = ac_net_main.get_weights()
    print(old_w)
    update_main = partial(update_main_weights, old_W=old_w, net_main=ac_net_main)

    for i in range(concurrent):
        # actor_pool.apply_async(actor_learner_thread, args=(ac_net_threads[i], i, samples, rewards),
                               # callback=update_main)
        update_main(actor_learner_thread(ac_net_threads[i], i, samples, rewards))
    # actor_pool.close()
    # actor_pool.join()