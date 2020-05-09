#!/usr/bin/env python
# coding: utf-8



import scipy.io as sio
import torch
from torch.autograd import Variable
import numpy as np




X_BOW = sio.loadmat('../data/synthetic_bag_of_words.mat')
X_BOW = X_BOW.get('bag_of_words')
X_BOW = Variable(torch.from_numpy(X_BOW).double())
n = 98

Label_BOW = np.zeros(n)
Label_BOW[10:25] = 1
Label_BOW[25:35] = 2
Label_BOW[35:49] = 3
Label_BOW[49:59] = 4
Label_BOW[59:68] = 5
Label_BOW[68:78] = 6
Label_BOW[78:88] = 7
Label_BOW[88:99] = 8
Label_BOW = torch.Tensor(Label_BOW).long()
Label_BOW = Variable(Label_BOW)

L_BOW = sio.loadmat('../data/L_mat_bow_data.mat')
L_BOW = L_BOW.get('L')
L_BOW = L_BOW.T
L_BOW = torch.from_numpy(L_BOW).double()
L_BOW = Variable(L_BOW)

Y_BOW = sio.loadmat('../data/Y_mat_bow_data.mat')
Y_BOW = Y_BOW.get('Y')
Y_BOW = Y_BOW.T
Y_BOW = torch.from_numpy(Y_BOW).double()
Y_BOW = Variable(Y_BOW)
