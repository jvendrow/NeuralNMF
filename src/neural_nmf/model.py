#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Class and functions for initializing (semi-supervised) hierarchical NMF model and forward-propagation
    of Neural NMF.
    The class consists of a sequence of factor matrices with dimensions defined by the input depth info 
    and an optional classification layer defined by the optional input number of classes.
    Examples
    --------
    >>> net = Neural_NMF([m, 9], none)
    >>> train(net, X, epoch=200, lr=1000)                            #uses function from train class
    epoch =  10 
     tensor(228.1642, dtype=torch.float64)
    epoch =  20 
     tensor(228.0765, dtype=torch.float64)
    epoch =  30 
    ...
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
from matplotlib import pyplot as plt
import scipy.io as sio
from neural_nmf import LsqNonneg
import numpy as np
import torch.nn.functional as F



class Neural_NMF(nn.Module):
    """
    Class for Neural NMF network structure.
    
    The Neural_NMF object contains several NMF layers(contained in self.lsqnonneglst, each element in 
    the list self.lsqnonneglst is a Lsqnonneg object) and a linear layer for classification(self.linear).
    Given X, the input, is mxn, this will initialize factor matrices in hierarchical NMF
        X = A_0 * A_1 * .. A_L * S_L, where:
            A_i is of size depth_info[i] x depth_info[i+1] and 
            S_L is of size depth_info[L] x n.
        If c is not None, it also initializes a classification layer defined by B*S_L where:
            B is of size c x depth_info[L].
    ...
    Parameters
    ----------
    depth_info: list
        The list [m, k1, k2,...k_L] contains the dimension information for all factor matrices.
    c: int_, optional
        Number of classes (default is None).
    Methods
    ----------
    forward(X)
        Forward propagate the Neural NMF network.
    """
    def __init__(self, depth_info, c = None):
        super(Neural_NMF, self).__init__()
        self.depth_info = depth_info
        self.depth = len(depth_info)
        self.c = c
        self.lsqnonneglst = nn.ModuleList([LsqNonneg(depth_info[i], depth_info[i+1]) 
                                           for i in range(self.depth-1)]) 
                                           #ititalized a list of Nonnegative least squared objects
        """
        if c is not None:
            self.linear = nn.Linear(depth_info[-1],c, bias = False).double() 
                                        #initialize classification layer (with last factor matrix)
            with torch.no_grad():
                self.linear.weight.copy_(torch.abs(torch.rand(depth_info[-1],c)))
        """

    def forward(self, X, Y=None, L=None, last_S_lst=None):
        """
        Runs the forward pass of the Neural NMF network.
        Parameters
        ----------
        X: PyTorch tensor
            The m x n input to the Neural NMF network. The first dimension, m, should match the first entry 
            of depth_info.
        Returns
        -------
        S_lst: list
            All S matrices ([S_0, S_1, ..., S_L]) calculated by the forward pass.
        pred: PyTorch tensor, optional
            The c x n output of the linear classification layer.
        """

        if len(X.shape) != 2:
            raise ValueError("Expected a two-dimensional Tensor, but X is of shape {}".format(X.shape))

        if X.shape[0] != self.depth_info[0]:
            raise ValueError("Dimension 0 of X should match entry 0 of depth_info, but values were {} and {}".format(X.shape[0], self.depth_info[0]))

        S_lst = []
        for i in range(self.depth-1):
            if last_S_lst == None:
                X = self.lsqnonneglst[i](X)  #Calculates the least squares objective S = min S>=0 ||X - AS||
            else:
                X = self.lsqnonneglst[i](X, last_S_lst[i]) #Calculates the least squares objective S = min S>=0 ||X - AS||
            S_lst.append(X)
        #once at end of network (X = S^L)
        if self.c is None:
            return S_lst
        else: #if in supervised case, multiply by B to produce pred = B*S^L

            #B = torch.mm(Y[:,L[0]==1], torch.pinverse(X[:,L[0]==1]))
            
            B = torch.mm(Y, torch.pinverse(X))
            #pred = torch.transpose(self.linear(torch.transpose(X,0,1)),0,1)
            pred = torch.mm(B,X)
            return S_lst, pred
        


