from __future__ import division

import numpy as np

from pycog import tasktools


from torchvision import datasets, transforms

import torch


testset_01 = datasets.MNIST('/Users/Amartya/data', train=False, download=True,
                   transform = np.asarray)
idx = testset_01.targets <= 5
#idx = torch.as_tensor(np.logical_or(testset_01.targets == 4, testset_01.targets == 5), dtype = torch.bool)
testset_01.targets = testset_01.targets[idx]
testset_01.data = testset_01.data[idx]
test_loader01 = torch.utils.data.DataLoader(testset_01,
    batch_size=1, shuffle=True)
data = iter(test_loader01)

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
        data = iter(test_loader01)
        X, Y = data.next()
        X = X.numpy().reshape(28,28)
        
    trial['inputs'] = X
    
    
    
    

    #---------------------------------------------------------------------------------
    # Target output
    #---------------------------------------------------------------------------------

    if params.get('target_output', False):
        #Y = np.zeros(Nout) # Output matrix
        output = np.zeros(Nout) # Output matrix
        output[Y.item()] = 1
        
        trial['outputs'] = output
        


    return trial

min_error = 0.01

n_validation = 100

##CHECKING MNIST PERFORMANCE
if __name__ == '__main__':
    from pycog          import RNN
    from pycog.figtools import Figure
 
    rnn  = RNN('examples/work/data/mnist_07/mnist_07.pkl', {'dt': 1, 'var_rec': 0.01**2})
    trial_args = {'target_output':True}
    sum1=0
    for i in range(len(testset_01)):
        rnn.run(inputs=(generate_trial, trial_args), rng=i, seed=200)
        sum1 += rnn.acc
    
    print('no of accurate', sum1)
    print('The accuracy rate is  ', (sum1/len(testset_01))*100, '%')
    print(len(testset_01))


    
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
    
    
    