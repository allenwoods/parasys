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
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.callbacks import EarlyStopping

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


def lazy_police(update_steps, t, directions):
    if (simulation.getCurrentTime() % 100) % update_steps == 0:
        max_halt = max([t.edges[d].edge_status['halt'] for d in directions])
        passway = (direction for direction, edge in t.edges.items()
                   if edge.edge_status['halt'] == max_halt).__next__()  # Search keys according to value
        print('Haltest: %s' % passway)
        t.set_phase(passway)


class AutoEncoder:
    def __init__(self, log, target, dataset_shape, train_size=0):
        if type(log) == pd.DataFrame:
            self.log_pd = log
        elif type(log) == str and log.split('/')[-1].split('.')[-1] == 'csv':
            self.load_data(log)
        else:
            raise ValueError('Log Type incorrect')
        self.target_set = self.log_pd[target].as_matrix().reshape(dataset_shape)
        self.train_size = train_size
        if train_size == 0:
            self.train_set = self.target_set
        else:
            self.train_set = self.target_set[:train_size]
            self.test_set = self.target_set[train_size:]

    def load_data(self, log_csv):
        self.log_pd = pd.DataFrame.from_csv(log_csv, index_col=False)

    def create(self, encoding_dim,
               encode_activation='relu', w_regularizer=regularizers.l1(10e-5),
               optimizer='nadam', loss='mse', metrics=None):
        if metrics is None:
            metrics = ['accuracy']
        sample_shape = self.train_set[0].shape
        input_mat = Input(shape=sample_shape)
        encoded = Dense(encoding_dim, activation=encode_activation,
                        W_regularizer=w_regularizer)(input_mat)
        decoded = Dense(sample_shape[0])(encoded)
        self.autoencoder = Model(input=input_mat, output=decoded)
        self.autoencoder.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, epoch, minibatch_size, val_split=0.1,
            es_monitor='val_loss', es_patience=5, verbose=0):
        self.autoencoder.fit(self.train_set, self.train_set, nb_epoch=epoch,
                             batch_size=minibatch_size, validation_split=val_split,
                             callbacks=[EarlyStopping(monitor=es_monitor, patience=es_patience)],
                             verbose=verbose)

    def evaluate(self, test_set=None, batch_size=32):
        if self.train_size == 0 and test_set == None:
            raise ValueError('No test set!')
        elif test_set is not None:
            result = self.autoencoder.evaluate(test_set, test_set, batch_size=batch_size)
        else:
            result = self.autoencoder.evaluate(self.test_set, self.test_set)
        return result