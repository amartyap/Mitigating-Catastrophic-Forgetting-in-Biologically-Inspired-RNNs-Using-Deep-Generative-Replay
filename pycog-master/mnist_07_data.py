from __future__ import division

import numpy as np

from pycog import tasktools

import torch

from torch.utils.data import Dataset

from torchvision import datasets

from pycog import RNN



Nin = 28
N = 64
Nout = 10
output_activation = 'softmax'
dt = 1
learning_rate = 0.001
n_gradient = 64



class GeneratedDataset1(Dataset):

    def __init__(self, data_file, transform=None):
        
        self.datafile = data_file
        self.data = torch.load(self.datafile)
        self.data = self.data.view(-1,1,28,28)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        img = self.data[index]

        if self.transform:
            img = self.transform(img)
        target = None

        return img
    

gendataset0123 = GeneratedDataset1('/Users/Amartya/new0123.pt')
gendataset0123.data = gendataset0123.data*255
gendataset0123.data =gendataset0123.data.view(-1,28,28)
gendataloader0123 = torch.utils.data.DataLoader(gendataset0123,
    batch_size=1, shuffle=False)

data = iter(gendataloader0123)


#-----------------------------------------------------------------------------------------
# Network structure
#-----------------------------------------------------------------------------------------



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
        X  = data.next()
        X = X.detach().numpy().reshape(28,28)
    except StopIteration:
        print('omg omg')
        data = iter(gendataloader0123)
        X  = data.next()
        X = X.detach().numpy().reshape(28,28)

        
    trial['inputs'] = X
    
    #---------------------------------------------------------------------------------
    # Target output
    #---------------------------------------------------------------------------------

    if params.get('target_output', False):
        output = np.zeros(Nout) # Output matrix
        
        trial['outputs'] = output

    return trial

min_error = 0.01

n_validation = 100

images0123 = np.zeros((len(gendataset0123), 28, 28))
targets0123 = np.zeros(len(gendataset0123))
rnn  = RNN('examples/work/data/mnist_06/mnist_06.pkl', {'dt': 1, 'var_rec': 0.01**2})
trial_args = {'target_output':False}
for i in range(len(gendataset0123)):
    rnn.run(inputs=(generate_trial, trial_args), rng=i, seed=200)
    images0123[i,:,:] = rnn.input
    targets0123[i] = rnn.digit

torch.save((torch.as_tensor(images0123), torch.as_tensor(targets0123)), '/Users/Amartya/file0123.pt')
    





