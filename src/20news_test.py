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
from neural_nmf_sup import Neural_NMF, Energy_Loss_Func, L21_Norm, Recon_Loss_Func
from lsqnonneg_module import LsqNonneg
from train import train_unsupervised, train_supervised
#
import torch.nn as nn
from writer import Writer

# data loading session
from news_group_loading import get_data

from time import time

# set the network parameters
X, Y_sub, Y_super = get_data()
#X = X.T
m = X.shape[0]
k1 = 10
k2 = 6
#c = 9
print(X.shape)

# unsupervised case,one layer
net = Neural_NMF([m, k1])
loss_func = Energy_Loss_Func()

start = time()
history_unsupervised = train_unsupervised(net, X, loss_func, epoch = 20, lr = 2e13, weight_decay=0.999, full_history=True, verbose=True)
end = time()

print("Training time: {}".format(end-start))
exit(1)
# by calling history_unsupervised.get('variable_name'), you can get the variables that you recorded in the writer
# getting these results might be helpful for debugging and choosing hyperparameters
A1_lst = history_unsupervised.get('A1')
S1_lst = history_unsupervised.get('S1')
A2_lst = history_unsupervised.get('A2')
S2_lst = history_unsupervised.get('S2')
grad_A1_lst = history_unsupervised.get('grad_A1')
print(S1_lst[0].shape)
print(A1_lst[0].shape)


# plot the loss curve
history_unsupervised.plot_scalar('loss')
# plot the heatmap for S1
history_unsupervised.plot_tensor('S1', [-1])
history_unsupervised.plot_tensor('A1', [-1])
history_unsupervised.plot_tensor('S2', [-1])
history_unsupervised.plot_tensor('A2', [-1])
