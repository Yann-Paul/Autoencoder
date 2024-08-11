import torch
import torch.tensor as tensor
import numpy as np
import scipy.io
import h5py

mode = torch.zeros(8)
index = [1,2,3,4,5,6,8,10]
for i in range(8):
    mode[i] = torch.load("loss_mode_"+str(index[i]))


Data = torch.zeros([80,20,3,256,256],dtype=torch.float)

for i in range(80):
    for j in range(20):
        j_first = int((j+1)/10)
        j_second = (j+1) % 10
        data_load = h5py.File('threeVortices_DATA/data_out._a_'+str(j_first)+'.'+str(j_second)+'_'+str((i+1)*25).zfill(10)+'.mat')
        #data_load = tensor(scipy.io.loadmat('threeVortices_DATA/data_out._a_'+str(j_first)+'.'+str(j_second)+'_'+str((i+1)*25).zfill(10)+'.mat')
        for k, v in data_load.items():
            Data[i,j,:,:,:] = tensor(v)

for i in range(3):
    Data[:,:,0,:,:]/(Data[:,:,0,:,:].max()-Data[:,:,0,:,:].min())

