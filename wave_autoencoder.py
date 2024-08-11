import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD
import torch.tensor as tensor
import matplotlib.pyplot as plt
import numpy as np
import sys
from tqdm  import tqdm
import class_Data_Gauss

input_dim = 128 
train_signals = 5000 # number of datasets for training
test_signals = 500    # number of datasets for testing
N_TRAIN_STEPS = 10000
rtol_acc = 0.001        # tolerance for accuracy calculation
layers = 2              # only for saving the module
learning_rate = 0.005
minimal_sigma = 0.1

Data = class_Data_Gauss.Data_Gauss(input_dim,minimal_sigma)

class Wave_Gauss_autoencoder(nn.Module):  # inherit from default pytoch neural net module
    def __init__(self):
        # initialize as nn.Module
        super().__init__()

        # layers
        self.layer_in_1 = nn.Linear(in_features=input_dim, out_features=int(input_dim/2))
        self.layer_in_2 = nn.Linear(in_features=int(input_dim/2), out_features=3)
        self.layer_out_6 = nn.Linear(in_features=3, out_features=int(input_dim/2))
        self.layer_out_7 = nn.Linear(in_features=int(input_dim/2), out_features=int(input_dim))

        self.activation1 = nn.LeakyReLU(0.1)
        #self.activation2 = nn.ReLU()
        #self.activation3 = nn.LeakyReLU(10)
        #self.activation = nn.Tanh()


    def forward(self, s):  # overwright forward method from nn.Module. This needs to be done for any neural net in pytorch!
        t = self.activation1(self.layer_in_1(s))
        u = self.activation1(self.layer_in_2(t))
        t = self.activation1(self.layer_out_6(u))
        s = self.activation1(self.layer_out_7(t))
        return u.squeeze(),s.squeeze()  # squeeze to remove empty dimension

# build an instance of our or-gate-neural net. All weights and biases are initializted with random data here:
Wave = Wave_Gauss_autoencoder()

# create Data
Train_Data_full, Train_Data_reduced = Data.create_data(train_signals)
Test_Data_full, Test_Data_reduced = Data.create_data(test_signals)

# gradient descent with adaptive moment estimation (more or less the standart)
optimizer = Adam(params=Wave.parameters(), lr=learning_rate)

# loss criterion: Mean Squared Error
loss_crit = nn.MSELoss()

losses = []

    # train_loop
for step in tqdm(range(N_TRAIN_STEPS)):
    reduced,Wave_out = Wave(Train_Data_full)

    # compute the loss (the "quality" of the output of the neural net) where the output data is noramlized by the amplitude of each signal
    loss = loss_crit(Wave_out/Train_Data_reduced[:,0].view(train_signals,1),Train_Data_full/Train_Data_reduced[:,0].view(train_signals,1))
    losses.append(loss.item())  
    optimizer.zero_grad()

    # compute the gradients of the net-parameters with regards to the loss
    loss.backward()
    optimizer.step()

# plot the convergence of the training
plt.figure()
plt.semilogy(np.arange(N_TRAIN_STEPS), losses)
plt.xlabel('trainstep')
plt.ylabel('loss')
plt.show()

# test the network
reduced,Test_Data_autoencoded = Wave(Test_Data_full)
loss = loss_crit(Test_Data_autoencoded, Test_Data_full)

# Output for the Batch
#print("minimal resolution per period " + str(minimal_resolution))
print("input_dimension " + str(input_dim))
print("number of training signals " + str(train_signals))
print("number of test signals " + str(test_signals))
print("Number of train epochs " + str(N_TRAIN_STEPS))
print("Loss after training: " + str(loss.item()))

#calculate accuracy
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
name = "wave_autoencoder_" + "_accuracy_" + str(average_acc) + "_indim_" + str(input_dim) + "_trainsignals_" + str(train_signals) + "_testsignals_" + str(test_signals) + "_epochs_" + str(N_TRAIN_STEPS) + "_tolerance_" + str(rtol_acc) + "_layers_" + str(layers) + "_learningrate_" + str(learning_rate) + "_minimal_sigma_" + str(minimal_sigma)
torch.save(Wave,name)
print("object saved in working directory")

