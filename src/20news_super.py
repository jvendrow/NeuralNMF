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
from news_group_loading import get_data

from time import time

# set the network parameters
X, Y_sub, Y_super, vocab = get_data()

m = X.shape[0]
k1 = 10
k2 = 6

#supervised case,one layer
net = Neural_NMF([m, k1, k2], 6)
loss_func = Energy_Loss_Func(lambd=1)

start = time()
history_supervised = train_supervised(net, X, Y_super, loss_func=loss_func, epoch = 150, lr_nmf = 1e13, lr_classification= 3e14, weight_decay=0.99, decay_epoch=10, full_history=True, verbose=True, verbose_epoch=10)
end = time()

print("Training time: {}".format(end-start))

# by calling history_unsupervised.get('variable_name'), you can get the variables that you recorded in the writer
# getting these results might be helpful for debugging and choosing hyperparameters
A1_lst = history_supervised.get('A1')
S1_lst = history_supervised.get('S2')
weight_lst = history_supervised.get('weight')

A1 = A1_lst[-1]
S1 = S1_lst[-1]
B = weight_lst[-1]

"""
for i in range(A1.shape[1]):
    col = A1[:,i].numpy()
    top = col.argsort()
    print(top.shape)
    top = top[-10:][::-1]

    print("Row {} rows".format(i))
    for j in top:
        print(vocab[j])
"""
approx = torch.mm(B, S1).numpy()
Y_pred = np.argmax(approx, axis=0)
Y = Y_super.numpy()
print(Y)
print(Y_pred)
print("Accuracy: {}/{}".format(Y[Y_pred==Y].shape[0], Y.shape[0]))



#A2_lst = history_supervised.get('A2')
#S2_lst = history_supervised.get('S2')
grad_A1_lst = history_supervised.get('grad_A1')
print(S1_lst[0].shape)
print(A1_lst[0].shape)


# plot the loss curve
history_supervised.plot_scalar('loss_nmf')
history_supervised.plot_scalar('loss_classification')
# plot the heatmap for S1
history_supervised.plot_tensor('S1', [-1])
history_supervised.plot_tensor('A1', [-1])
#history_supervised.plot_tensor('S2', [-1])
#history_supervised.plot_tensor('A2', [-1])
