#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:55:34 2019

@author: Amartya
"""

from __future__ import division

import numpy as np

from pycog import tasktools
Nin = 1
N = 100
Nout = 10

# E/I
ei, EXC, INH = tasktools.generate_ei(N)

def generate_trial(rng, dt, params):
    T = 1000

    signal_time = rng.uniform(100, T - 600)
    delay = 500
    width = 20
    magnitude = 4

    epochs = {}
    epochs['T'] = T
    t, e  = tasktools.get_epochs_idx(dt, epochs) # Time, task epochs in discrete time
    trial = {'t': t, 'epochs': epochs}           # Trial

    trial['info'] = {}

    signal_time /= dt
    delay /= dt
    width /= dt

    # Input matrix
    X = np.zeros((len(t), Nin))

    for tt in range(len(t)):
        if tt > signal_time:
            X[tt][0] = np.exp(-(tt - signal_time) / delay) * magnitude

    trial['inputs'] = X

    #---------------------------------------------------------------------------------
    # Target output
    #---------------------------------------------------------------------------------

    #if params.get('target_output', False):
    if True:
        Y = np.zeros((len(t)+1, Nout)) # Output matrix
        M = np.ones((len(t), Nout)) # Mask matrix

        for i in range(Nout):
            for tt in range(len(t)+1):
                Y[tt][i] = np.exp( -(tt - (signal_time + delay / Nout * (i + 1)))**2 / (2 * width**2)) * magnitude

        trial['outputs'] = Y

    return trial
trial_args = {}

from pycog import RNN
rnn = RNN('examples/work/data/delay_react/delay_react.pkl', {'dt': 10.0, 'var_rec': 0.01**2})
rnn.run(inputs=(generate_trial, trial_args), seed=200)
z = rnn.z
#true data
trial_args = {'target_output':True}
params = trial_args
seed=200
rng = np.random.RandomState(seed)
dt = 10.0
trial = generate_trial(rng, dt, params)
Y = trial['outputs'].T
print(Y.shape)
print(z.shape)
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
print('Square root of mean squared error:', np.sqrt(mse(Y, z)))
print('R_2 Score:', r2_score(Y, z))







