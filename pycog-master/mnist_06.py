from __future__ import division

import numpy as np

from pycog import tasktools

import torch

from torch.utils.data import Dataset

from torchvision import datasets



Nin = 28
N = 64
Nout = 10
output_activation = 'softmax'
dt = 1
learning_rate = 0.001
n_gradient = 64


class GeneratedDataset(Dataset):

    def __init__(self, data_file, transform=None):
        
        self.datafile = data_file
        self.data, self.targets = torch.load(self.datafile)
        self.data = self.data.view(-1,1,28,28)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        img, target = self.data[index], int(self.targets[index])

        if self.transform:
            img = self.transform(img)

        return img, target


generated01_trainset = GeneratedDataset('/Users/Amartya/file1.pt')

generated01_trainset.data *= 255

generated01_trainset.data = generated01_trainset.data.view(-1,28,28)


trainset_23 = datasets.MNIST('/Users/Amartya/data', train=True, download=True,
                   transform = np.asarray)
idx = torch.as_tensor(np.logical_or(trainset_23.targets == 2, trainset_23.targets == 3), dtype = torch.bool)

trainset_23.targets = trainset_23.targets[idx]
trainset_23.data = trainset_23.data[idx]
trainset_23.data = trainset_23.data.float()
trainset_23.data = trainset_23.data[:1408]

'''old_trainloader = torch.utils.data.DataLoader(generated01_trainset,
    batch_size=1, shuffle=True)
new_trainloader = torch.utils.data.DataLoader(trainset_23,
    batch_size=1, shuffle=True)'''

total_data = torch.cat((generated01_trainset.data.view(-1,28,28), trainset_23.data), 0)
total_targets = torch.cat((generated01_trainset.targets, trainset_23.targets), 0)
torch.save((total_data, total_targets), '/Users/Amartya/file0123.pt')



final_trainset = GeneratedDataset('/Users/Amartya/file0123.pt')
final_trainset.data = final_trainset.data.view(-1,28,28)
final_trainloader = torch.utils.data.DataLoader(final_trainset,
    batch_size=1, shuffle=True)


data = iter(final_trainloader) 
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
        X, Y = data.next()
        X = X.detach().numpy().reshape(28,28)
    except StopIteration:
        data = iter(final_trainloader)
        X, Y = data.next()
        X = X.detach().numpy().reshape(28,28)

        
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

