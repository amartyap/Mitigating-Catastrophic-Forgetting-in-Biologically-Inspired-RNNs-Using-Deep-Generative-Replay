from __future__ import division

import numpy as np

from pycog import tasktools

import pandas as pd

r = pd.read_csv('mnist-in-csv/mnist_test.csv')

#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

Nin = 1
N = 250
Nout = 10
output_activation = 'softmax'
dt = 1
mode='continuous'

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
    #index = rng.randint(0,len(r))
    #Input matrix
    
    X = np.zeros((len(t), Nin))
    X[:,0] = np.array(r.iloc[rng,1:])
    
    trial['inputs'] = X
    
    

    #---------------------------------------------------------------------------------
    # Target output
    #---------------------------------------------------------------------------------

    if params.get('target_output', False):
        Y = np.zeros(Nout) # Output matrix
        Y[r.iloc[rng,0]]=1
        

        trial['outputs'] = Y

    return trial

min_error = 0.1

n_validation = 100

##CHECKING MNIST PERFORMANCE
if __name__ == '__main__':
    from pycog          import RNN
    from pycog.figtools import Figure
 
    rnn  = RNN('examples/work/data/mnist_03/mnist_03.pkl', {'dt': 1, 'var_rec': 0.01**2})
    trial_args = {'target_output':True}
    sum1=0
    for i in range(len(r)):
        rnn.run(inputs=(generate_trial, trial_args), rng=i, seed=200)
        sum1 += rnn.acc
    
    print 'no of accurate', sum1
    print('The accuracy rate is  ', (sum1/10000)*100, '%')
    
        
        
    
    
    
    
    
''' fig  = Figure()
    plot = fig.add()

    plot.plot(rnn.t/tau, rnn.u[0], color=Figure.colors('blue'))
    for i in range(Nout):
        plot.plot(rnn.t/tau, rnn.z[i], color=Figure.colors('red'))
        plot.plot(rnn.t[1:]/tau, Y[i], color =Figure.colors('blue'))
    plot.xlim(rnn.t[0]/tau, rnn.t[-1]/tau)
    plot.ylim(0, 15)

    plot.xlabel(r'$t/\tau$')
    plot.ylabel(r'$t/\tau$')
    
    

    fig.save(path='.', name='mnist_01')'''
    
    
    
    
