#!/usr/bin/env python
# coding: utf-8



# Date: 2018.07.22
# Author: Runyu Zhang



'''
This demo demonstrate how to train a one layer, unsupervised NMF and one layer, supervised NMF, and how to analyze and visualize the 
result.

In real implementation,  it might be helpful and of more flexibility to write your own training process 
instead of using the training functions in 'train.Ipynb'. This should not be too hard - you can simply
make a few lines change in the functions in 'train.Ipynb' to create your own training function.
'''


# loading packages and functions
import torch
import numpy as np
from matplotlib import pyplot as plt
#import Ipynb_importer
from neural_nmf import Neural_NMF, Energy_Loss_Func, L21_Norm, Recon_Loss_Func
from lsqnonneg_module import LsqNonneg
from train import train_unsupervised, train_supervised
#
import torch.nn as nn
from writer import Writer

# data loading session
from data_loading import X_BOW as X
from data_loading import Label_BOW as label
from data_loading import Y_BOW as Y
from data_loading import L_BOW as L

from time import time

# set the network parameters
X = X.T
m = X.shape[0]
k1 = 20
k2 = 15
k3 = 10
c = 9
print(X.shape)

# unsupervised case,one layer
net = Neural_NMF([m, 9])
#loss_func = Energy_Loss_Func()
loss_func = Recon_Loss_Func()
X_input = X*1000

start = time()
history_unsupervised = train_unsupervised(net, X_input, loss_func, epoch = 200, lr = 1000, full_history=True, verbose=True)
end = time()

print("Training time: {}".format(end-start))
# by calling history_unsupervised.get('variable_name'), you can get the variables that you recorded in the writer
# getting these results might be helpful for debugging and choosing hyperparameters
A1_lst = history_unsupervised.get('A1')
S1_lst = history_unsupervised.get('S1')
grad_A1_lst = history_unsupervised.get('grad_A1')
print(S1_lst[0].shape)
print(A1_lst[0].shape)


# plot the loss curve
history_unsupervised.plot_scalar('loss')
# plot the heatmap for S1
history_unsupervised.plot_tensor('S1', [-1])

'''
The following code demonstrate the process of training a supervised one-layer NMF.

The training for supervised NMF is a little bit tricky here, I am not sure if this is the best way to do this. Here I am using
different learning rate for the NMF layer(lsqnonneg layer) and the classification layer. What's more, in the same epoch, the
NMF layer and the classification layer are trained separately, the NMF will do one gradient update in one epoch, and the
classification layer will do thirty. See more detail in the function 'train_supervised' in 'train.Ipynb'
'''

# supervised case
net = Neural_NMF([m, 9], 9)
net.linear.weight.data = 1e-3*torch.randn(9,9,dtype = torch.double)
loss_func = Energy_Loss_Func(lambd = 100000,classification_type = 'L2')
X_input = X*1000
history_supervised = train_supervised(net, X_input, loss_func, Y, epoch = 30, lr_nmf = 5000, lr_classification = 0.01, weight_decay = 1)

# plotting the loss curve
history_supervised.plot_scalar('loss')
# plotting the heatmap of S1
history_supervised.plot_tensor('S1', [-1])
# getting the history for different varialbes
A1_lst = history_supervised.get('A1')
S1_lst = history_supervised.get('S1')
grad_A1_lst = history_supervised.get('grad_A1')
B_lst = history_supervised.get('weight')
grad_B_lst = history_supervised.get('grad_weight')

S_lst, pred = net(X)
torch.argmax(pred, dim = 1) != label

plt.imshow(net.linear.weight.data)
plt.show()








m = X.shape[1]
n = X.shape[0]
k = 9
net = Neural_NMF([m,9])
loss_func = Energy_Loss_Func()
X_input = 1000*X




net = Neural_NMF([m, 12])
loss_func = Energy_Loss_Func()
X_input = X*1000
epoch = 400
lr = 1000
for i in range(epoch):
    net.zero_grad()
    S_lst = net(X)
    loss = loss_func(net, X_input, S_lst)
    loss.backward()
    history.add_scalar('loss', loss.data)
    for l in range(net.depth - 1):
        A = net.lsqnonneglst[l].A
        # record history
        history.add_tensor('A'+str(l+1), A.data)
        history.add_tensor('grad_A'+str(l+1), A.grad.data)
        history.add_tensor('S' + str(l+1), S_lst[l].data)
        # projection gradient descent
        A.data = A.data.sub_(lr*A.grad.data)
        A.data = A.data.clamp(min = 0)
    if (i+1)%10 == 0:
        print('epoch = ', i+1, '\n', loss.data)




fig = plt.figure(figsize = (15,105))
plt.imshow(S_lst[0].data.t())
plt.show()






