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
from lsqnonneg_module import LsqNonneg
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
        self.c= c
        self.lsqnonneglst = nn.ModuleList([LsqNonneg(depth_info[i], depth_info[i+1]) 
                                           for i in range(self.depth-1)]) 
                                           #ititalized a list of Nonnegative least squared objects
        if c is not None:
            self.linear = nn.Linear(depth_info[-1],c, bias = False).double() 
                                        #initialize classification layer (with last factor matrix)

    def forward(self, X):
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
            X = self.lsqnonneglst[i](X) #Calculates the least squares objective S = min S>=0 ||X - AS||
            S_lst.append(X)
        #once at end of network (X = S^L)
        if self.c is None:
            return S_lst
        else: #if in supervised case, multiply by B to produce pred = B*S^L
            pred = torch.transpose(self.linear(torch.transpose(X,0,1)),0,1)
            return S_lst, pred
        




class Energy_Loss_Func(nn.Module):
    """
    Defining the energy loss function as in the Neural NMF Paper. #Jamie: can we add math description?
    ...
    Parameters
    ----------
    lambd: float, optional
        The regularization parameter, defining weight of classification error in loss function. 
    classification_type: string, optional
        Classification loss indicator 'L2' or 'CrossEntropy' (default 'CrossEntropy').
    
    Methods
    ----------
    forward(net,X,S_lst)
        Forward propagates and computes energy loss value.
    """
    def __init__(self,lambd = 0, classification_type = 'CrossEntropy'):
        super(Energy_Loss_Func, self).__init__()
        self.lambd = lambd
        self.classification_type = classification_type
        self.criterion1 = ReconstructionLoss()
        if classification_type == 'L2':
            self.criterion2 = ClassificationLossL2()
        else:
            self.criterion2 = ClassificationLossCrossEntropy()
            
    def forward(self, net, X, S_lst, pred = None, label = None, L = None):
        """
        Runs the forward pass of the energy loss function.
        Parameters
        ----------
        net: Pytorch module Neural NMF object
            The Neural NMF object for which the loss is calculated.
        X: Pytorch tensor
            The input to the Neural NMF network (matrix to be factorized).
        S_lst: list
            All S matrices ([S_0, S_1, ..., S_L]) that were returned by the forward pass of the Neural 
            NMF object.
        pred: Pytorch tensor, optional
            The approximation to the classification one-hot indicator matrix of size c x n produced
            by forward pass (B*S_L) (default is None).
        label: Pytorch tensor, optional
            The classification (label) matrix for supervised model.  If the classification_type is 'L2',
            this matrix is a one-hot encoding matrix of size c x n.  If the classification_type is
            'CrossEntropy', this matrix is of size n with elements in [0,c-1] (default is None).
        L: Pytorch tensor, optional
            The label indicator matrix for semi-supervised model that indicates if labels are known 
            for n data points, of size c x n with columns of all ones or all zeros to indicate if label
            for that data point is known (default is None).
        Returns
        -------
        reconstructionloss: Pytorch tensor
            The total energy loss from X, the S matrices, and the A matrices, stored in a 1x1 Pytorch 
            tensor to preserve information for backpropagation.
        """
        total_reconstructionloss = self.criterion1(X, S_lst[0], net.lsqnonneglst[0].A)
        depth = net.depth
        for i in range(1,depth-1):
            total_reconstructionloss += self.criterion1(S_lst[i-1], S_lst[i], net.lsqnonneglst[i].A)
        if pred is None:
            # unsupervised case
            assert(label is None and L is None)
            return total_reconstructionloss
        else:
            # fully supervised case and semisupervised case
            classificationloss = self.criterion2(pred, label, L)
            return total_reconstructionloss + self.lambd*classificationloss




class Recon_Loss_Func(nn.Module):
    """
    Defining the reconstruction loss function as in the paper Deep NMF. #Jamie: can we add a math description?
    ...
    Parameters
    ----------
    lambd: float, optional
        The regularization parameter, defining weight of classification error in loss function. 
    classification_type: string, optional
        Classification loss indicator 'L2' or 'CrossEntropy' (default 'CrossEntropy').
    
    Methods
    ----------
    forward(net,X,S_lst)
        Forward propagates and computes energy loss value.
    """
    def __init__(self, lambd = 0, classification_type = 'CrossEntropy'):
        super(Recon_Loss_Func, self).__init__()
        self.lambd = lambd
        self.classification_type = classification_type
        self.criterion = Fro_Norm()
        if classification_type == 'L2':
            self.criterion2 = ClassificationLossL2()
        else:
            self.criterion2 = ClassificationLossCrossEntropy()
            
    def forward(self, net, X, S_lst, pred = None, label = None, L = None):
        """
        Runs the forward pass of the energy loss function.
        Parameters
        ----------
        net: Pytorch module Neural NMF object
            The Neural NMF object for which the loss is calculated.
        X: Pytorch tensor
            The input to the Neural NMF network (matrix to be factorized).
        S_lst: list
            All S matrices ([S_0, S_1, ..., S_L]) that were returned by the forward pass of the Neural 
            NMF object.
        pred: Pytorch tensor, optional
            The approximation to the classification one-hot indicator matrix of size c x n produced
            by forward pass (B*S_L) (default is None).
        label: Pytorch tensor, optional
            The classification (label) matrix for supervised model.  If the classification_type is 'L2',
            this matrix is a one-hot encoding matrix of size c x n.  If the classification_type is
            'CrossEntropy', this matrix is of size n with elements in [0,c-1] (default is None).
        L: Pytorch tensor, optional
            The label indicator matrix for semi-supervised model that indicates if labels are known 
            for n data points, of size c x n with columns of all ones or all zeros to indicate if label
            for that data point is known (default is None).
        Returns
        -------
        reconstructionloss: Pytorch tensor
            The total energy loss from X, the S matrices, and the A matrices, stored in a 1x1 Pytorch 
            tensor to preserve information for backpropagation.
        """

        depth = net.depth
        
        X_approx = S_lst[-1]
        for i in range(depth-2, -1, -1):
            X_approx = torch.mm(net.lsqnonneglst[i].A,X_approx)
        
        reconstructionloss = self.criterion(X_approx, X)

        if pred is None:
            # unsupervised case
            assert(label is None and L is None)
            return reconstructionloss
        else:
            # fully supervised case and semisupervised case
            classificationloss = self.criterion2(pred, label, L)
            return reconstructionloss + self.lambd*classificationloss




class Fro_Norm(nn.Module):
    """
    Calculate the Frobenius norm between two matrices of the same size. This function actually returns 
    the entrywise average of the square of the Frobenius norm. 
    Examples
    --------
    >>> criterion = Fro_Norm()
    >>> loss = criterion(X1,X2)
    """
    def __init__(self):
        super(Fro_Norm, self).__init__()
        self.criterion = nn.MSELoss()
    def forward(self,X1, X2):
        """
        Runs the forward pass of the Frobenius norm module
        Parameters
        ----------
        X1: Pytorch tensor
            The first input to the Frobenius norm loss function
        X2: Pytorch tensor
            The second input to the Frobenius norm loss function
        Returns
        -------
        loss: Pytorch tensor
            The Frobenius norm of X1 and X2, stored in a 1x1 Pytorch tensor to preserve information for 
            backpropagation.
        """
        len1 = torch.numel(X1.data)
        len2 = torch.numel(X2.data)
        assert(len1 == len2)
        X = X1 - X2
        #X.contiguous()
        loss =  self.criterion(X.reshape(len1), Variable(torch.zeros(len1).double()))
        return loss

class ReconstructionLoss(nn.Module):
    """
    Calculates the entrywise average of the square of Frobenius norm ||X - AS||_F^2.
    Examples
    --------
    >>> criterion = ReconstructionLoss()
    >>> loss = criterion(X, S, A)
    """
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.criterion = Fro_Norm()
    def forward(self, X, S, A):
        """
        Runs the forward pass of the ReconstructionLoss module
        Parameters
        ----------
        X: Pytorch tensor
            The first input to the loss function (m x n matrix).
        A: Pytorch tensor
            The first factor of the second input to the loss function (m x k matrix).
        S: Pytorch tensor
            The second factor of the second input to the loss function (k x n matrix).
        Returns
        -------
        reconstructionloss: Pytorch tensor
            The loss of X and A*S, stored in a 1x1 Pytorch tensor to preserve information for 
            backpropagation.
        """
        X_approx = torch.mm(A,S)
        reconstructionloss = self.criterion(X_approx, X)
        return reconstructionloss
    

class ClassificationLossL2(nn.Module):
    """
    Calculates the classification loss  ||L.*(Y - Y_pred)||_F^2.
    Examples
    ---------
    >>> criterion = ClassificationLossL2()
    >>> loss = criterion(Y, Y_pred)                                      #full supervision
    >>> loss = criterion(Y, Y_pred, L)                                   #semi-supervision
    """
    def __init__(self):
        super(ClassificationLossL2, self).__init__()
        self.criterion = Fro_Norm()
    def forward(self, Y, Y_pred, L = None):
        """
        Runs the forward pass of the ClassificationLossL2 module.
        Parameters
        ----------
        Y: Pytorch tensor
            The one-hot encoding matrix of the data (c x n).
        Y_pred: Pytorch tensor
            The approximation to Y (B*S_L) (c x n).
        L: Pytorch tensor
            The label indicator matrix (where all-zero columns indicate no label information and 
            all-one columns indicate label information for corresponding data points) (c x n).
        Returns
        -------
        classificationloss: Pytorch tensor
            The loss of Y and Y_pred, stored in a 1x1 Pytorch tensor to preserve information for 
            backpropagation.
        """
        if L is None:
            classificationloss = self.criterion(Y_pred, Y)
            return classificationloss
        else:
            classificationloss = self.criterion(L*Y_pred, L*Y)
            return classificationloss

class ClassificationLossCrossEntropy(nn.Module):
    """
    Calculates the classification cross-entropy.
    Examples
    ---------
    >>> criterion = ClassificationLossCrossEntropy()
    >>> loss = criterion(Y_pred, labels)                                      #full supervision
    >>> loss = criterion(Y_pred, labels, L)                                   #semi-supervision
    """
    def __init__(self):
        super(ClassificationLossCrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, Y_pred, label, L = None):
        """
        Runs the forward pass of the ClassificationLossCrossEntropy module.
        Parameters
        ----------
        Y_pred: Pytorch tensor
            The approximation to the one=hot encoding matrix (B*S_L) (c x n).
        label: Pytorch tensor
            The label vector (size n).
        L: Pytorch tensor
            The label indicator matrix (where all-zero columns indicate no label information and 
            all-one columns indicate label information for corresponding data points) (c x n).
        Returns
        -------
        classificationloss: Pytorch tensor
            The loss of Y_pred and label, stored in a 1x1 Pytorch tensor to preserve information for 
            backpropagation.
        """
        if L is None:
            classificationloss = self.criterion(torch.transpose(Y_pred,0,1), label)
            return classificationloss
        else:
            l = Variable(L[0,:].data.long())
            classificationloss = self.criterion(torch.transpose(L*Y_pred,0,1), l*label)
            return classificationloss




class L21_Norm(nn.Module):
    """
    Calculate the L21 norm of a matrix.
    Examples
    --------
    >>> criterion = L21_Norm()
    >>> loss = criterion(X)
    """
    def __init__(self):
        super(L21_Norm, self).__init__()
        self.criterion = nn.MSELoss()
    def forward(self, S):
        """
        Runs the forward pass of the L21 norm module.
        Parameters
        ----------
        S: Pytorch tensor
            The input to the L21_Norm module.
        Returns
        -------
        total: Pytorch tensor
            The L21 norm of S, stored in a 1x1 Pytorch tensor to preserve information for 
            backpropagation.
        """
        total = 0
        n = S.shape[1]
        for i in range(n):
            total += torch.norm(S[:,i])
        return total
