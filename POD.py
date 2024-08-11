import torch
import torch.nn as nn
import torch.tensor as tensor
import matplotlib.pyplot as plt
import numpy as np
from tqdm  import tqdm
import h5py
import scipy.io
import matplotlib
from sklearn.decomposition import PCA

data_types = 3

train_batch_size = 1476
test_batch_size = 164

code_dimension = 6

read_data = 1   # read_data = 1: Data is loaded with matlab file. 0: Data is loaded with TORCH file
calc_min_max = 0 #  Caution takes about 40 minutes to calculate. boolean for calulating the min max MSE and the min avg MSE of the Datavectors.
plot_eigenfaces = 1 # plots at most 9 eigenfaces/eigenvectors

plot_reconstruction = 1 # plot the reconstructed image and the original data for the referred alpha and t
recons_alpha = 12
recons_t = 43

loss_crit = nn.MSELoss(reduction = 'mean')
np.random.seed(42)

# for loading the matlab file
def load_data():
    Data = torch.zeros([82,20,data_types,256,256],dtype=torch.float)
    D_X_Data = torch.zeros([82,20,data_types,256,256],dtype=torch.float)
    D_Y_Data = torch.zeros([82,20,data_types,256,256],dtype=torch.float)

    for i in range(82):
        for j in range(20):
            j_first = int((j+1)/10)
            j_second = (j+1) % 10
            if i==0:
                i_first = 1
            else:
                i_first = (i)*25
                if i==81:
                    i_first = 2001
            data_load = h5py.File('threeVortices_DATA/data_out._a_'+str(j_first)+'.'+str(j_second)+'_'+str(i_first).zfill(10)+'.mat')
            #data_load = tensor(scipy.io.loadmat('threeVortices_DATA/data_out._a_'+str(j_first)+'.'+str(j_second)+'_'+str((i+1)*25).zfill(10)+'.mat')
            for k, v in data_load.items():
                for d in range(data_types):
                    Data[i,j,d,:,:] = tensor(v[d,:,:])

    for i in range(3):
        Data[:,:,i,:,:] = (Data[:,:,i,:,:]-Data[:,:,i,:,:].min())/(Data[:,:,i,:,:].max()-Data[:,:,i,:,:].min())
    return Data

if read_data:
    Data = load_data()
    torch.save(Data,"data.TORCH")
else:
    Data = torch.load("data.TORCH")

index_bool = np.zeros([82,20])
index_train = index_bool
for i in range(82):
    shuffle = np.arange(20)
    np.random.shuffle(shuffle)
    index_train[i,shuffle[range(18)]] = 1
index_train = (index_train == 1)
index_test = (index_train == 0)

pooling_layer = nn.AvgPool2d(2,2,0)

Data_pool = np.reshape(pooling_layer(torch.flatten(Data,start_dim = 0,end_dim=1)).data.numpy(),(82,20,3,128,128))

Data_train = Data[index_train,:,:,:]
Data_train_pool = pooling_layer(Data_train)

# Data has to be split into two tensors to fit the size in pca (training matrix only has 90% of Data)
Data_full_1 = Data[:,0:18,:,:,:]
Data_full_2 = Data[:,2:20,:,:,:]
Data_full_1 = torch.flatten(Data_full_1,start_dim = 0, end_dim = 1)
Data_full_2 = torch.flatten(Data_full_2,start_dim = 0, end_dim = 1)
Data_full_1 = pooling_layer(Data_full_1)
Data_full_2 = pooling_layer(Data_full_2)
Data_full_1 = torch.flatten(Data_full_1,start_dim = 1, end_dim = 3)
Data_full_2 = torch.flatten(Data_full_2,start_dim = 1, end_dim = 3)

W = Data_train_pool
W = torch.flatten(W,start_dim = 1,end_dim = 3)
W = W.T

Data_modes_out = torch.zeros((82,20,3,128,128))
Data_modes_test_out = torch.zeros([164,3,128,128])
Data_modes_test = torch.zeros([164,3,128,128])
Data_test = Data[index_test,:,:,:]
Data_test = pooling_layer(Data_test)


pca = PCA(n_components=code_dimension)
pca.fit(W.T)

# training loss
W_out = pca.inverse_transform(pca.transform(W.T)).T
loss = loss_crit(W,tensor(W_out))
print("mean squared error for training batch is: " + str(loss.data.numpy()))

Data_full_1_out = pca.inverse_transform(pca.transform(Data_full_1))
Data_full_2 = pca.inverse_transform(pca.transform(Data_full_2))

#Data_full_1 = np.reshape(Data_full_1,(82,18,3,128,128))
Data_full_1_out = np.reshape(Data_full_1_out,(82,18,3,128,128))
Data_full_2 = np.reshape(Data_full_2,(82,18,3,128,128))

Data_full_out = np.concatenate((Data_full_1_out,Data_full_2[:,16:18,:,:,:]),1)

# test loss
Data_test_out = Data_full_out[index_test,:,:,:]
loss = loss_crit(Data_test,tensor(Data_test_out))
print("mean squared error for test batch is: " + str(loss.data.numpy()))

# calculate the min max and min avg
if calc_min_max:
    max_loss = torch.zeros(82,20)
    avg_loss = torch.zeros(82,20)
    for i in tqdm(range(82)):
        for j in range(20):
            max_loss[i,j] = torch.max(torch.sum((Data[:,:,:,:,:] - Data[i,j,:,:,:])**2/(3*128*128),(2,3,4)))
            avg_loss[i,j] = torch.sum((Data[:,:,:,:,:] - Data[i,j,:,:,:])**2)/(3*128*128*82*20)
    min_max_loss = torch.min(max_loss)
    min_avg_loss = torch.min(avg_loss)

    print("the min_max_loss is: " + str(min_max_loss))
    print("the min_avg_loss is: " + str(min_avg_loss))


# define LaTeX Font Settings
TeX = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 20,
    "font.size": 20,
    "legend.fontsize": 8,
    "axes.titlesize": 14,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,

}
matplotlib.rcParams.update(TeX)

# plot reconstructed images
if plot_reconstruction:
    fig1, ax = plt.subplots(2,3, sharex = True, sharey = True, figsize=(4, 3))
    plt.setp(ax, xlim=(0, 128), ylim=(0, 128))
    fig1.suptitle('code dim: ' + str(code_dimension), fontsize=14)

    label = ["$u$","$v$","$p$"]
    for i in range(2):
        for j in range(3):
            if i == 0:
                ax[i,j].pcolormesh(Data_pool[recons_t,recons_alpha,j,:,:])
            else:
                ax[i,j].pcolormesh(Data_full_out[recons_t,recons_alpha,j,:,:])
            ax[i,j].set_aspect('equal', 'box')
            ax[i,j].axis('off')
            ax[i,j].set_title(label[j])
    plt.tight_layout()

    plt.show()

# plot the eigenfaces/eigenvectors. If input dimension is lower than 9 some subplots will be empty
if plot_eigenfaces:
    eigenfaces = np.reshape(pca.components_,(code_dimension,3,128,128))

    for types in range(3):
        fig2, ax = plt.subplots(3,3, sharex = True, sharey = True, figsize=(4, 3))
        plt.setp(ax, xlim=(0, 128), ylim=(0, 128))
        for i in range(min(code_dimension,9)):
            ax[int(i/3),np.mod(i,3)].pcolormesh(eigenfaces[i,types,:,:])
            ax[int(i/3),np.mod(i,3)].set_aspect('equal', 'box')
            ax[int(i/3),np.mod(i,3)].axis('off')
            ax[int(i/3),np.mod(i,3)].set_title(str(i+1))
        fig2.suptitle(label[types])
        plt.show()



