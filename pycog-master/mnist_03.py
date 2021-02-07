from __future__ import division

import numpy as np

from pycog import tasktools

import pandas as pd

r = pd.read_csv('mnist-in-csv/mnist_train.csv')

#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

Nin = 1
N = 128
Nout = 10
output_activation = 'softmax'
dt = 1
n_gradient = 16
learning_rate = 0.001


# E/I
ei, EXC, INH = tasktools.generate_ei(N)

# Time constant
tau = 50

#-----------------------------------------------------------------------------------------
# Noise
#-----------------------------------------------------------------------------------------

var_rec = 0.01**2

def generate_trial(rng, dt, params):
    
    T = 784

    epochs = {}
    epochs['T'] = T
    t, e  = tasktools.get_epochs_idx(dt, epochs) # Time, task epochs in discrete time
    trial = {'t': t, 'epochs': epochs}           # Trial

    trial['info'] = {}
    
    global r
    index = rng.randint(0,len(r))
    #Input matrix
    
    X = np.zeros((len(t), Nin))
    X[:,0] = np.array(r.iloc[index,1:])
    
    trial['inputs'] = X
    
    

    #---------------------------------------------------------------------------------
    # Target output
    #---------------------------------------------------------------------------------

    if params.get('target_output', False):
        Y = np.zeros(Nout) # Output matrix
        Y[r.iloc[index,0]]=1
        

        trial['outputs'] = Y

    return trial

min_error = 0.1

n_validation = 100


if __name__ == '__main__':
    from pycog          import RNN
    from pycog.figtools import Figure

    rnn  = RNN('examples/work/data/2nd_task/2nd_task.pkl', {'dt': 0.5, 'var_rec': 0.01**2})
    trial_args = {}
    info = rnn.run(inputs=(generate_trial, trial_args), seed=200)
    seed=200
    rng = np.random.RandomState(seed)
    dt = 0.5
    trial_args = {'target_output':True}
    params = trial_args
    trial = generate_trial(rng, dt, params)
    Y = trial['outputs'].T
    
    
    

    fig  = Figure()
    plot = fig.add()

    plot.plot(rnn.t/tau, rnn.u[0], color=Figure.colors('blue'))
    for i in range(Nout):
        plot.plot(rnn.t/tau, rnn.z[i], color=Figure.colors('red'))
        plot.plot(rnn.t[1:]/tau, Y[i], color =Figure.colors('blue'))
    plot.xlim(rnn.t[0]/tau, rnn.t[-1]/tau)
    plot.ylim(0, 15)

    plot.xlabel(r'$t/\tau$')
    plot.ylabel(r'$t/\tau$')
    
    

    fig.save(path='.', name='mnist_01')
