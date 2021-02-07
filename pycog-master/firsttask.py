from __future__ import division

import numpy as np

from pycog import tasktools


#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

Nin = 1
N = 100
Nout = 10
#mode = 'continuous'
n_gradient = 20

# E/I
ei, EXC, INH = tasktools.generate_ei(N)

# Time constant
tau = 50

#-----------------------------------------------------------------------------------------
# Noise
#-----------------------------------------------------------------------------------------

var_rec = 0.01**2

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
            X[tt][0] = np.exp(-(tt - signal_time) / delay) * magnitude * (np.sin(-0.015*tt)+4)

    trial['inputs'] = X 

    #---------------------------------------------------------------------------------
    # Target output
    #---------------------------------------------------------------------------------

    if params.get('target_output', False):
        Y = np.zeros((len(t), Nout)) # Output matrix
        M = np.ones((len(t), Nout)) # Mask matrix

        for i in range(Nout):
            for tt in range(len(t)):
                Y[tt][i] = np.exp( -(tt-1 - (signal_time + delay / Nout * (i + 1)))**2 / (2 * width**2)) * magnitude

        trial['outputs'] = Y

    return trial

min_error = 0.1

n_validation = 100


if __name__ == '__main__':
    from pycog          import RNN
    from pycog.figtools import Figure

    #rnn  = RNN('examples/work/data/2nd_task/2nd_task.pkl', {'dt': 0.5, 'var_rec': 0.01**2})
    #trial_args = {}
    #info = rnn.run(inputs=(generate_trial, trial_args), seed=200)
    seed=200
    rng = np.random.RandomState(seed)
    dt = 0.5
    trial_args = {'target_output':True}
    params = trial_args
    trial = generate_trial(rng, dt, params)
    Y = trial['outputs'].T
    u = trial['inputs'][:,0]
    t = np.concatenate(([0], trial['t']))
   
    
    
    
    

    fig  = Figure()
    plot = fig.add()

    plot.plot(t[1:]/tau, u, color=Figure.colors('green'))
    for i in range(Nout):
        #plot.plot(rnn.t/tau, rnn.z[i], color=Figure.colors('red'))
        plot.plot(t[1:]/tau, Y[i], color =Figure.colors('blue'))
    plot.xlim(t[0]/tau, t[-1]/tau)
    plot.ylim(0, 15)

    plot.xlabel(r'$t/\tau$')
    plot.ylabel(r'Neural Activations')
    
    plot.legend(['Input', 'Target Output'])

    fig.save(path='.', name='testing')
