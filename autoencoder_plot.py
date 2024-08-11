import torch
import torch.nn as nn
from torch.optim import Adam
import torch.tensor as tensor
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import scipy.io

# parameters for the network
input_dim = 128 # has to be 128
layers = 2 # for now only 2 and 5 layers can be used

Data_X = torch.zeros([256,256,80],dtype=torch.float)
Data_Y = torch.zeros([256,256,80],dtype=torch.float)
Data_Rot = torch.zeros([256,256,80],dtype=torch.float)

Data_X[:,:,0] = tensor(scipy.io.loadmat('Data/X_1.mat')["X"])
Data_Y[:,:,0] = tensor(scipy.io.loadmat('Data/Y_1.mat')["Y"])
Data_Rot[:,:,0] = tensor(scipy.io.loadmat('Data/Rot_1.mat')["Rot"])

for i in range(79):
    j = (i+1)*25-1
    Data_X[:,:,i+1] = tensor(scipy.io.loadmat('Data/X_'+str(j)+'.mat')["X"])
    Data_Y[:,:,i+1] = tensor(scipy.io.loadmat('Data/Y_'+str(j)+'.mat')["Y"])
    Data_Rot[:,:,i+1] = tensor(scipy.io.loadmat('Data/Rot_'+str(j)+'.mat')["Rot"])

# further parameters for the network and network class
if layers == 2:
    average_acc = 0.7919375000000001
    train_signals = 5000    # number of datasets for training
    test_signals = 500    # number of datasets for testing
    N_TRAIN_STEPS = 10000
    rtol_acc = 0.001       # tolerance for the accuracy
    learning_rate = 0.005
    minimal_sigma = 0.1

    class Wave_Gauss_autoencoder(nn.Module):  # inherit from default pytoch neural net module
        def __init__(self):
            # initialize as nn.Module
            super().__init__()
            # layers
            self.layer_in_1 = nn.Linear(in_features=input_dim, out_features=int(input_dim/2))
            self.layer_in_2 = nn.Linear(in_features=int(input_dim/2), out_features=3)
            self.layer_out_6 = nn.Linear(in_features=3, out_features=int(input_dim/2))
            self.layer_out_7 = nn.Linear(in_features=int(input_dim/2), out_features=input_dim)
            # activation functions
            self.activation1 = nn.LeakyReLU(0.1) 
            self.activation3 = nn.LeakyReLU(0.20) 
        def forward(self, w):  # overwright forward method from nn.Module. This needs to be done for any neural net in pytorch!
            x = self.activation1(self.layer_in_1(w))
            y = self.activation1(self.layer_in_2(x))
            x = self.activation1(self.layer_out_6(y))
            w = self.activation1(self.layer_out_7(x))
            return y.squeeze(),w.squeeze()  # squeeze to remove empty dimension"""
    #Wave_Gauss_autoencoder = class_Wave_Gauss_auto.Wave_Gauss_autoencoder_2(input_dim)


# parameters for plotting
discretization_points = 100 # number of time steps
plot_duration = 0.01   # seconds for one plot series

# parameters for dataset
min_amp = 0.5
max_amp = 5
min_sigma = minimal_sigma
max_sigma = 3
min_phase = 0
max_phase = 10
# values for normalization
max = tensor([max_amp,max_sigma,max_phase],dtype=torch.float)
min = tensor([min_amp,min_sigma,min_phase],dtype=torch.float)

# function for the dataset
def create_data(discretization_points, discretization_type):
    x = np.linspace(0, 10, input_dim)
    amp = np.random.uniform(min_amp,max_amp)*np.ones(discretization_points)
    sigma = np.random.uniform(min_sigma,max_sigma)*np.ones(discretization_points)
    phase = np.random.uniform(min_phase,max_phase)*np.ones(discretization_points)

    if discretization_type == 0:
        amp = np.linspace(min_amp,max_amp,discretization_points)
        x1 = amp
        A = np.outer(amp,np.exp(-(x-phase[0])**2/sigma[0]**2))
        A = A.T
    if discretization_type == 1:
        sigma = np.linspace(min_sigma,max_sigma,discretization_points)
        x1 = sigma
        A = amp*np.exp(-np.divide.outer((x-phase[0])**2,sigma**2))
    if discretization_type == 2:
        phase = np.linspace(min_phase,max_phase,discretization_points)
        x1 = phase
        A = amp*np.exp(-np.subtract.outer(x,phase)**2/sigma**2)  
  
    Data_full = tensor(A.T,dtype=torch.float)
    B = np.array([amp.flatten(),sigma.flatten(),phase.flatten()])
    Data_reduced = tensor(B.T,dtype=torch.float)
    return(x1,Data_full,Data_reduced)


# loading the network
autoencoder = "wave_autoencoder_" + "_accuracy_" + str(average_acc) + "_indim_" + str(input_dim) + "_trainsignals_" + str(train_signals) + "_testsignals_" + str(test_signals) + "_epochs_" + str(N_TRAIN_STEPS) + "_tolerance_" + str(rtol_acc) + "_layers_" + str(layers) + "_learningrate_" + str(learning_rate) + "_minimal_sigma_" + str(minimal_sigma)
Wave = torch.load(autoencoder)
Wave.eval()

# latexplot
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

# go through each changeable wave parameter and plot the wave and code parameters
for discretization_type in range(3):
    # get the data
    x2,Data_full,Data_reduced = create_data(discretization_points,discretization_type)
    Wave_reduced,Wave_full = Wave(Data_full)

    # for plotting
    x = np.linspace(0, 10, input_dim)
    y = Data_full.data.numpy()
    z = Wave_full.data.numpy()

    # plot the best signal
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(discretization_points):
        # plot the best signal
        ax1 = fig.add_subplot(2,1,1)
        if discretization_type == 0:
            ax1.set(ylim=(-max_amp*0.2,max_amp*1.2), xlim=(0, 2*np.pi))
            ax1.title.set_text('amplitude change')
        elif discretization_type == 1:
            ax1.title.set_text('sigma change')
        elif discretization_type == 2:
            ax1.title.set_text('position change')
        ax1.plot(x,y[i,:])
        ax1.plot(x,z[i,:])
        ax2 = fig.add_subplot(2,1,2)
        ax2.set(xlim=(x2[0],x2[discretization_points-1]),ylim = (float(torch.min(Wave_reduced)),float(torch.max(Wave_reduced))))
        ax2.plot(x2[range(i)],Wave_reduced[range(i),0].data.numpy(),'r')
        ax2.plot(x2[range(i)],Wave_reduced[range(i),1].data.numpy(),'b')
        ax2.plot(x2[range(i)],Wave_reduced[range(i),2].data.numpy(),'g')
        if discretization_type == 0:
            ax2.set_xlabel("amplitude")
        elif discretization_type == 1:
            ax2.set_xlabel("variance")
        elif discretization_type == 2:
            ax2.set_xlabel("position")
        plt.show(block=False)
        plt.pause(plot_duration/discretization_points)
        if i == discretization_points-1:
            plt.pause(1)
        plt.clf()    
    plt.close()



