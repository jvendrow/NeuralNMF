#!/usr/bin/env python
# coding: utf-8



# Date:2018.07.21
# Author: Runyu Zhang




import torch
import torch.nn as nn
from torch.autograd import Variable
from matplotlib import pyplot as plt
import scipy.io as sio
#import Ipynb_importer
from lsqnonneg_module import LsqNonneg
import numpy as np
import torch.nn.functional as F




class Deep_NMF(nn.Module):
    """
    Class for Neural NMF network structure.
    
    the Deep_NMF object contains several NMF layers(contained in self.lsqnonneglst, each element in the list self.lsqnonneglst is a Lsqnonneg object)
    and a linear layer for classification(self.linear).

    Examples
    --------
    >>> net = Deep_NMF([m, 9], none)
    >>> train(net, X, epoch=200, lr=1000)
    epoch =  10 
     tensor(228.1642, dtype=torch.float64)
    epoch =  20 
     tensor(228.0765, dtype=torch.float64)
    epoch =  30 
    ...

    """
    def __init__(self, depth_info, c = None):
        """
        Initialize Neural NMF network

        Parameters
        ----------
        depth_info: list, [m, k1, k2,...k_L]
            Given X, the input, is mxn, this will approximate 
            X = A_0 * A_1 * .. A_L * S_L, where:
            A_i is of size depth_info[i] x depth_info[i+1] and 
            S_L is of size depth_info[L] x n

        c: default None, otherwise scalar
            Indicates the number of classes

        """

        super(Deep_NMF, self).__init__()
        self.depth_info = depth_info
        self.depth = len(depth_info)
        self.c= c
        self.lsqnonneglst = nn.ModuleList([LsqNonneg(depth_info[i], depth_info[i+1]) 
                                           for i in range(self.depth-1)]) #ititalized a list of Nonnegative least squared objects
        if c is not None:
            self.linear = nn.Linear(c,depth_info[-1], bias = False).double()
    def forward(self, X):
        """
        Runs the forward pass of the Neural NMF network

        Parameters
        ----------
        X: PyTorch Tensor, size mxn
            The input to the Neural NMF network. The first dimension,
            m, should match the first entry of depth_info

        Returns
        -------
        S_lst: list, [S_0, S_1, ..., S_L]
            All S matrices calculated by the forward pass

        pred: (Not sure the dimensions)
            The outputs of a linaer layer following the Neural NMF layers


        """
        S_lst = []
        for i in range(self.depth-1):
            X = self.lsqnonneglst[i](X) #Calculates the least squares objective S = min S>=0 ||X - AS||
            S_lst.append(X)
        if self.c is None:
            return S_lst
        else:
            pred = self.linear(X)
            return S_lst, pred
        




class Energy_Loss_Func(nn.Module):
    """
    Defining the energy loss function as in the Neural NMF Paper
    

    """
    def __init__(self,lambd = 0, classification_type = 'CrossEntropy'):
        """
        Initializes the energy loss

        Parameters
        ----------
        lambd: float
            The regularization parameter, defining how important the 
            classification error is for supervised NMF. 

        classification_type: String
            'L2' or 'CrossEntropy'. Default 'CrossEntropy'

        """

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
        Runs the forward pass of the energy loss function

        Parameters
        ----------
        net: Pytorch Module
            The Neural NMF object for which the loss is calculated 

        X: Pytorch Tensor
            The input to the Neural NMF network

        S_lst: list, [S_0, S_1, ..., S_L]
            All S matrices that were returned by the forward pass 
            of the Neural NMF object

        pred: ?

        label: ?

        L: ?

        Returns
        -------
        reconstructionloss: float
            The total energy loss from X, the S matrices, and
            the A matrices
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
    Defining the reconstruction loss function as in the paper deep NMF

    """
    def __init__(self, lambd = 0, classification_type = 'CrossEntropy'):
        """
        Parameters
        ----------
        lambd: float
        The regularization parameter, defining how important the
        classification error is for supervised NMF. 

        classification_type: String
            'L2' or 'CrossEntropy'. Default 'CrossEntropy'
        """

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
        Runs the forward pass of the energy loss function

        Parameters
        ----------
        net: Pytorch Module
            The Neural NMF object for which the loss is calculated 

        X: Pytorch Tensor
            The input to the Neural NMF network

        S_lst: list, [S_0, S_1, ..., S_L]
            All S matrices that were returned by the forward pass 
            of the Neural NMF object

        pred: ?

        label: ?

        L: ?

        Returns
        -------
        reconstructionloss: float
            The total reconstruction loss from X, the S matrices,
            and the A martices.
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




## Defining all kinds of loss functions that is needed

class Fro_Norm(nn.Module):
    """
    calculate the Frobenius norm between two matrices of the same size.
    Do: criterion = Fro_Norm()
        loss = criterion(X1,X2) and the loss is the entrywise average of the square of Frobenius norm.
    """
    def __init__(self):
        super(Fro_Norm, self).__init__()
        self.criterion = nn.MSELoss()
    def forward(self,X1, X2):
        len1 = torch.numel(X1.data)
        len2 = torch.numel(X2.data)
        assert(len1 == len2)
        X = X1 - X2
        #X.contiguous()
        return self.criterion(X.view(len1), Variable(torch.zeros(len1).double()))

class ReconstructionLoss(nn.Module):
    """
    calculate the reconstruction error ||X - AS||_F^2.
    Do: criterion = ReconstructionLoss()
        loss = criterion(X, S, A) and the loss is the entrywise average of the square of Frobenius norm ||X - AS||_F^2.
    """
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.criterion = Fro_Norm()
    def forward(self, X, S, A):
        X_approx = torch.mm(A,S)
        reconstructionloss = self.criterion(X_approx, X)
        return reconstructionloss
    

class ClassificationLossL2(nn.Module):
    """
    calculate the classification loss, using the criterion ||L.*(Y - Y_pred)||_F^2.
    Do: criterion = ReconstructionLoss()
        loss = criterion(Y, Y_pred) and the loss is the entrywise average of the square of Frobenius norm ||Y - Y_pred||_F^2.
        loss = criterion(Y, Y_pred, L) and the loss is the entrywise average of the square of the Frobenius norm ||L.*(Y - Y_pred)||_F^2
    """
    def __init__(self):
        super(ClassificationLossL2, self).__init__()
        self.criterion = Fro_Norm()
    def forward(self, Y, Y_pred, L = None):
        if L is None:
            classificationloss = self.criterion(Y_pred, Y)
            return classificationloss
        else:
            classificationloss = self.criterion(L*Y_pred, L*Y)
            return classificationloss

class ClassificationLossCrossEntropy(nn.Module):
    """
    calculate the classification loss, using the criterion ||L.*(Y - Y_pred)||_F^2.
    Do: criterion = ReconstructionLoss()
        loss = criterion(Y, Y_pred) and the loss is the entrywise average of the square of Frobenius norm ||Y - Y_pred||_F^2.
        loss = criterion(Y, Y_pred, L) and the loss is the entrywise average of the square of the Frobenius norm ||L.*(Y - Y_pred)||_F^2
    """
    def __init__(self):
        super(ClassificationLossCrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, Y_pred, label, L = None):
        if L is None:
            classificationloss = self.criterion(Y_pred, label)
            return classificationloss
        else:
            l = Variable(L[:,0].data.long())
            classificationloss = self.criterion(L*Y_pred, l*label)
            return classificationloss




class L21_Norm(nn.Module):
    """
    Defining the L21 Norm: ||X||_{2,1} = \sum ||X_i||_2
    This norm is defined to encourage row sparsity
    """
    def __init__(self):
        super(L21_Norm, self).__init__()
        self.criterion = nn.MSELoss()
    def forward(self, S):
        total = 0
        n = S.shape[1]
        for i in range(n):
            total += torch.norm(S[:,i])
        return total

