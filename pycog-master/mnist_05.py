from __future__ import division

import numpy as np

from pycog import tasktools


from torchvision import datasets, transforms

import torch

trainset_01 = datasets.MNIST('/Users/Amartya/data', train=True, download=True,
                   transform = np.asarray)
idx = trainset_01.targets <= 1
trainset_01.targets = trainset_01.targets[idx]
trainset_01.data = trainset_01.data[idx]
train_loader01 = torch.utils.data.DataLoader(trainset_01,
    batch_size=1, shuffle=True)
data = iter(train_loader01)
#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------

Nin = 28
N = 64
Nout = 10
output_activation = 'softmax'
dt = 1
learning_rate = 0.001
n_gradient = 64


# E/I
ei, EXC, INH = tasktools.generate_ei(N)

# Time constant
tau = 50

#-----------------------------------------------------------------------------------------
# Noise
#-----------------------------------------------------------------------------------------

var_rec = 0.01**2

def generate_trial(rng, dt, params):
    
    T = 28

    epochs = {}
    epochs['T'] = T
    t, e  = tasktools.get_epochs_idx(dt, epochs) # Time, task epochs in discrete time
    trial = {'t': t, 'epochs': epochs}           # Trial

    trial['info'] = {}
    
    global data
    try:
        X, Y = data.next()
        X = X.numpy().reshape(28,28)
    except StopIteration:
        data = iter(train_loader01)
        X, Y = data.next()
        X = X.numpy().reshape(28,28)
        
    
    
    trial['inputs'] = X
    
    

    #---------------------------------------------------------------------------------
    # Target output
    #---------------------------------------------------------------------------------

    if params.get('target_output', False):
        output = np.zeros(Nout) # Output matrix
        output[Y.item()] = 1
        
        trial['outputs'] = output

    return trial

min_error = 0.01

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
