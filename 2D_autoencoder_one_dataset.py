import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim import SGD
import torch.tensor as tensor
import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm  import tqdm
import scipy.io
import matplotlib

train_loop = 1
train_cycle = 10
train_batch_size = 150
test_batch_size = train_batch_size   # has to be the same for dimension in decoder
learning_rate = 0.01

Data_u = torch.zeros([2000,1,256,256],dtype=torch.float)
Data_v = torch.zeros([2000,1,256,256],dtype=torch.float)
Data_p = torch.zeros([2000,1,256,256],dtype=torch.float)


for i in range(2000):
    Data_u[i,:,:] = tensor(scipy.io.loadmat('Data_2/u_'+str(i+1)+'.mat')["u"])
    Data_v[i,:,:] = tensor(scipy.io.loadmat('Data_2/v_'+str(i+1)+'.mat')["v"])
    Data_p[i,:,:] = tensor(scipy.io.loadmat('Data_2/p_'+str(i+1)+'.mat')["p"])


class Conv_Net(nn.Module):

    def __init__(self):
        super(Conv_Net, self).__init__()
        # convolutional layers for encoder
        self.enc_conv1 = nn.Conv2d(3, 6, 3,1,padding = 1)
        self.enc_conv1p = nn.MaxPool2d(3,2,padding = 1,return_indices=True)
        self.enc_conv2 = nn.Conv2d(6, 1, 3,1,padding = 1)
        self.enc_conv2p = nn.MaxPool2d(3,2,padding = 1,return_indices=True)
        #self.enc_conv3 = nn.Conv2d(3,3,3,2,padding = 1)
        #self.enc_conv4 = nn.Conv2d(3,1,3,2,padding = 1)
        # convolutional layers for decoder
        self.dec_conv2p = nn.MaxUnpool2d(3,2,padding = 1)
        self.dec_conv2 = nn.ConvTranspose2d(1, 6, 3,1,padding = 1)
        self.dec_conv1p = nn.MaxUnpool2d(3,2,padding = 1)
        self.dec_conv1 = nn.ConvTranspose2d(6, 3, 3,1,padding = 1)
        #self.dec_conv1 = nn.ConvTranspose2d(3, 3, 3,2,padding = 1, output_padding = 1)
        #self.dec_conv1 = nn.MaxUnpool2d(3,2,padding = 1)
        # linear layers for encoder
        self.enc_linear1 = nn.Linear(1*64*64, 256)
        self.enc_linear4 = nn.Linear(256, 8)
        # linear layers for decoder
        self.dec_linear4 = nn.Linear(8, 256) 
        self.dec_linear1 = nn.Linear(256, 1*64*64)       

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x,indices1 = self.enc_conv1p(x)
        x = F.relu(self.enc_conv2(x))
        x,indices2 = self.enc_conv2p(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.enc_linear1(x))
        x = self.enc_linear4(x)
        return x,indices1,indices2

    def decode(self,x,indices1,indices2):
        x = F.relu(self.dec_linear4(x))
        x = F.relu(self.dec_linear1(x))
        x = x.view(train_batch_size,1,64,64)
        x = self.dec_conv2p(x,indices2,output_size=torch.Size([128,128]))
        x = F.relu(self.dec_conv2(x))
        x = self.dec_conv1p(x,indices1,output_size=torch.Size([256,256]))
        x = self.dec_conv1(x)
        #x = self.dec_conv1(x)
        return x

    def forward(self,x):
        y,indices1,indices2 = self.encode(x)
        x = self.decode(y,indices1,indices2)
        return x,y

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


Net = Conv_Net()

# gradient descent with adaptive moment estimation (more or less the standart)
optimizer = SGD(params=Net.parameters(), lr=learning_rate)

# loss criterion: Mean Squared Error
loss_crit = nn.MSELoss(reduction = 'mean')

losses = []

# create Data
rand_index = np.random.randint(0,2000,(train_loop,int(train_batch_size/2)))
index_old = np.random.randint(0,2000,int(train_batch_size/2))
batch_old = torch.cat((Data_u[index_old,:,:,:], Data_v[index_old,:,:,:],Data_p[index_old,:,:,:]),1)

# train_loop
for step in tqdm(range(train_loop)):
    index_new = rand_index[step,:]
    batch_new = torch.cat((Data_u[index_new,:,:,:], Data_v[index_new,:,:,:],Data_p[index_new,:,:,:]),1)
    # creates a batch of size 400
    batch_train = torch.cat((batch_old,batch_new),0)

    # train_cycle
    for cstep in tqdm(range(train_cycle)):
        full,reduced = Net(batch_train)

        # compute the loss (the "quality" of the output of the neural net) where the output data is noramlized by the amplitude of each signal
        loss = loss_crit(full,batch_train)
        losses.append(loss.item())  
        optimizer.zero_grad()

        # compute the gradients of the net-parameters with regards to the loss
        loss.backward()
        optimizer.step()

    batch_old = batch_new

# plot the convergence of the training
plt.figure()
plt.semilogy(np.arange(train_loop*train_cycle), losses)
plt.xlabel('trainstep')
plt.ylabel('loss')
plt.show()

# test the network
index_test = np.random.randint(0,2000,test_batch_size)
batch_test = torch.cat((Data_u[index_test,:,:,:], Data_v[index_test,:,:,:],Data_p[index_test,:,:,:]),1)
full,reduced = Net(batch_test)
loss_test = loss_crit(full,batch_test)

# Output for the Batch
#print("minimal resolution per period " + str(minimal_resolution))
print("size of training batch " + str(train_batch_size))
print("number of test signals " + str(test_batch_size))
print("Number of training cycles " + str(train_loop))
print("Number of training steps per cycle " + str(train_cycle))
print("loss after training " + str(loss.item()))
print("loss after testing " + str(loss_test.item()))


# define LaTeX Font Settings
TeX = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,

}

matplotlib.rcParams.update(TeX)

W = full.data.numpy()

fig2, ax = plt.subplots(2,3, sharex = True, sharey = True, figsize=(4, 3))
plt.setp(ax, xlim=(0, 128), ylim=(0, 128))

label = ["$u$","$v$","$p$"]
for i in range(2):
    for j in range(3):
        ax[i,j].pcolormesh(W[i,j,:,:])
        ax[i,j].set_aspect('equal', 'box')
        ax[i,j].axis('off')
        ax[i,j].set_title(label[j])
plt.tight_layout()
plt.show()

""" #calculate accuracy
# compute and print accuracy of trained network
Test_Data_full_array = Test_Data_full.data.numpy()
Test_Data_reduced_array = Test_Data_reduced.data.numpy()
Test_Data_reduced_vector = Test_Data_reduced_array[:,0]
accuracy = sum((Test_Data_autoencoded.data.numpy()-Test_Data_full.data.numpy())/Test_Data_reduced_vector[:,None]<rtol_acc)/test_signals
print(f'Average Accuracy on test data points: {accuracy}')
print('\n')
average_acc = sum(accuracy)/(input_dim)
print("average accuracy:" + str(average_acc))


# find the best and worst signal (by square error) for plotting
error_array = (Test_Data_autoencoded.data.numpy()-Test_Data_full.data.numpy())**2/Test_Data_reduced_vector[:,None]
error_functions = np.sum(error_array,axis=1)/input_dim
error_functions_sort = np.sort(error_functions)
emax = error_functions_sort[test_signals-1]
emiddle = error_functions_sort[int(test_signals/2)]
emin = error_functions_sort[0]

indexmax = np.ndarray.tolist(error_functions).index(emax)
indexmiddle = np.ndarray.tolist(error_functions).index(emiddle)
indexmin = np.ndarray.tolist(error_functions).index(emin)

# plot the best signal
x = np.linspace(0, 2*np.pi, input_dim)
y = Test_Data_full.data.numpy()
plt.plot(x,y[indexmin])
y = Test_Data_autoencoded.data.numpy()
plt.plot(x,y[indexmin])
plt.title('squared error:' + str(emin))
plt.show()

# plot the medium signal
y = Test_Data_full.data.numpy()
plt.plot(x,y[indexmiddle])
y = Test_Data_autoencoded.data.numpy()
plt.plot(x,y[indexmiddle])
plt.title('squared error:' + str(emiddle))
plt.show()

# plot the worst signal
y = Test_Data_full.data.numpy()
plt.plot(x,y[indexmax])
y = Test_Data_autoencoded.data.numpy()
plt.plot(x,y[indexmax])
plt.title('squared error:' + str(emax))
plt.show()
    
#name of the network
#name = "wave_autoencoder_" + "_accuracy_" + str(average_acc) + "_indim_" + str(input_dim) + "_trainsignals_" + str(train_signals) + "_testsignals_" + str(test_signals) + "_epochs_" + str(N_TRAIN_STEPS) + "_tolerance_" + str(rtol_acc) + "_layers_" + str(layers) + "_learningrate_" + str(learning_rate) + "_minimal_sigma_" + str(minimal_sigma)
#torch.save(Wave,name)
#print("object saved in working directory") """
