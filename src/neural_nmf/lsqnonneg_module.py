#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Defines class and functions for Nonnegative Least-Squares layers in hierarchical NMF
    model.
'''

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.function import once_differentiable
from scipy.optimize import nnls

from fnnls import fnnls

class LsqNonnegF(torch.autograd.Function):
    """
    Define the forward and backward process for q(X,A) = argmin_{S >= 0} ||X - AS||_F^2.

    """
    @staticmethod
    def forward(ctx, input, A, last_S=None):
        """
        Runs the forward pass of the nonnegative least squares task q(X,A).

        This solves the problem:
        min_{S >= 0} ||X - A*S||_F^2.
        Output[:,i] = argmin_{s >= 0} ||X[:,i] - A*s||_F^2.

        Parameters
        ----------
        ctx: context object
            Stashes information for the backwards pass.
        input: PyTorch tensor
            The input to the Neural NMF network, X. Input should have size (m,n).
        A: PyTorch tensor
            The A matrix for the current layer of the Neural NMF network, stored in the LsqNonneg class.

        Returns
        -------
        output: PyTorch tensor
            The output of the nonnegative least squares task q(X,A).

        """

        [output, res] = lsqnonneg_tensor_version(A.data, input.data, last_S)
        ctx.save_for_backward(input, A, None)
        ctx.intermediate = output
        return output.t().t()
    
    @staticmethod
    @once_differentiable #without this line then in backprop all the operations should be differentiable
    def backward(ctx, grad_output):
        """
        Runs the backwards pass of the nonnegative least squares task q(X,A). Computes the gradients of 
        q(X,A) with respect to X and A

        Parameters
        ----------
        ctx: context object
            Contains information stashed by the forward pass.
        grad_output: PyTorch tensor
            The gradient of S = q(X,A) passed on from the last layer

        Returns
        -------
        grad_input: PyTorch tensor
            The gradient of the nonnegative least squares task q(X,A) with respect to X.
        grad_A: PyTorch tensor
            The gradient of the nonnegative least squares task q(X,A) with respect to A.

        """

        input, A, temp = ctx.saved_tensors
        grad_input = grad_A = None
        output = ctx.intermediate
        if ctx.needs_input_grad[0]:
            grad_input = calc_grad_X(grad_output, A.data, output)# calculate gradient with respect to X
        if ctx.needs_input_grad[1]:
            grad_A = calc_grad_A(grad_output, A.data, output, input.data) # calculate gradient with respect to A
        return grad_input, grad_A, None




def lsqnonneg_tensor_version(A, X, last_S=None):
    """
    Calculates the nonnegative least squares solution q(X,A).

    Computes the following for each column of the output:
    output[:,i] = argmin_{s >= 0} ||X[:,i] - A*s||_F^2. 

    Parameters
    ---------
    A: PyTorch tensor
        The A matrix used to compute q(X,A).
    X: PyTorch tensor
        The X matrix used to compute q(X,A).

    Returns
    -------
    S: PyTorch tensor
        The S matrix, S = q(X,A).

    """
    A = A.numpy() # Transforming to numpy array size(m,k)
    X = X.numpy() # size(m,n)
    m = X.shape[0]
    n = X.shape[1]
    k = A.shape[1]
    S = np.zeros([k,n])
    res_total = 0
    
    #print([np.max(A),np.max(X)])
    for i in range(n):
        x = X[:,i]

        if last_S != None:
            P_initial = np.asarray([j for j in range(k) if last_S[j,i] > 0], dtype=int)
        else:
            P_initial = np.zeros(0, dtype=int)

        [s, res] = nnls(A, x)
        #[s, res] = fnnls(A, x)
        #[s, res] = fnnls(A, x, P_initial=P_initial)

        res_total += res
        S[:,i] = s

    S = torch.from_numpy(S).double() # Transforming to torch Tensor
    return S, res_total

def calc_grad_X(grad_S, A, S):
    """
    Calculates the gradient of q(X,A) with respect to X.

    Parameters
    ---------
    grad_S: PyTorch tensor
        The gradient of S = q(X,A) passed on from the last layer.
    A: PyTorch tensor
        The A matrix used to compute q(X,A).
    S: PyTorch tensor
        The output S = q(X,A).

    Returns
    -------
    grad_X: PyTorch tensor
        The gradient of q(X,A) with respsect to X.

    """
    A_np = A.numpy()
    S_np = S.numpy()
    grad_S_np = grad_S.numpy()
    m = A.shape[0]
    k = A.shape[1]
    n = S.shape[1]
    grad_X = np.zeros([m,n])
    for i in range(n):
        s = S_np[:,i]
        supp = s!=0
        grad_s_supp = grad_S_np[supp,i]
        A_supp = A_np[:,supp]
        grad_X[:,i] = np.linalg.pinv(A_supp).T@grad_s_supp
    grad_X = torch.from_numpy(grad_X).double()
    return grad_X

def calc_grad_A(grad_S, A, S, X):
    """
    Calculates the gradient of q(X,A) with respect to A.

    Parameters
    ---------
    grad_S: PyTorch tensor
        The gradient of S = q(X,A) passed on from the last layer.
    A: PyTorch tensor
        The A matrix used to compute q(X,A).
    S: PyTorch tensor
        The output S = q(X,A).
    X: PyTorch tensor
        The X matrix used to compute q(X,A).

    Returns
    -------
    grad_A: PyTorch tensor
        The gradient of q(X,A) with respsect to A.

    """

    A_np = A.numpy()
    S_np = S.numpy()
    grad_S_np = grad_S.numpy()
    X_np = X.numpy()
    m = A.shape[0]
    k = A.shape[1]
    n = S.shape[1]
    grad_A = np.zeros([m,k])
    for l in range(n):
        s = S_np[:,l]
        supp = s!=0
        A_supp = A_np[:,supp]
        grad_s_supp = grad_S_np[supp,l:l+1]
        x = X_np[:,l:l+1]
        A_supp_inv = np.linalg.pinv(A_supp)
        part1 = -(A_supp_inv.T@grad_s_supp)@(x.T@A_supp_inv.T)
        part2 = (x - A_supp@(A_supp_inv@x))@((grad_s_supp.T@A_supp_inv)@A_supp_inv.T)
        grad_A[:,supp] += part1 + part2
    grad_A = torch.from_numpy(grad_A).double()
    return grad_A


class LsqNonneg(nn.Module):
    """
    Defining a submodule 'LsqNonneg' of the nn.Module with network parameter.

    Parameters
    ----------
    m: int_
        The first dimension of A, size (m,k).
    k: int_
        The second dimension of A, size (m,k).
    initial_A: PyTorch tensor, optional
        The initialization of the A matrix (default None). If None, then A is generated randomly.

    Methods
    ----------
    forward(input)
        Computes the forward propagation of the nonnegative least-squares layer.

    """
    def __init__(self, m, k, initial_A = None):
        super(LsqNonneg, self).__init__()
        self.m = m;
        self.k = k;
        self.A = nn.Parameter(torch.DoubleTensor(m,k)) #switched
        if initial_A is None:
            self.A.data = torch.abs(torch.rand(m,k,dtype = torch.double)) # initialize the network parameter #Switched
        else:
            self.A.data = initial_A
        
    def forward(self, input, last_S=None):
        """
        The forward pass of the LsqNonneg submodule.

        Parameters
        ----------
        input: PyTorch tensor
            The input the the NMF layer, X.

        Returns
        -------
        S: PyTorch tensor
            The output of the LsqNonnegF forward pass, S = q(X,A).
        """

        return LsqNonnegF.apply(input, self.A,last_S)

