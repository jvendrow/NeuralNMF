#!/usr/bin/env python
# coding: utf-8



# Date:2018.07.21
# Author: Runyu Zhang




import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd.function import once_differentiable
from scipy.optimize import nnls




class LsqNonnegF(torch.autograd.Function):
    """
    Define the forward and backward process for q(X,A) = argmin_{S >= 0} ||X - AS||_F^2

    """
    @staticmethod
    def forward(ctx, input, A):
        """

        Runs the forward pass of the nonnegative least squares task q(X,A).

        This solves the problem:
        min_{S >= 0} ||X - A*S||_F^2
        output[:,i] = argmin_{s >= 0} ||X[:,i] - A*s||_F^2 

        Parameters
        ----------
        ctx: context object
            Stashes information for the backwards pass

        input: Pytorch Tensor
            The input to the Neural NMF network, X. Input should
            have size (m,n).

        A: Pytorch Tensor
            The A matrix for the current layer of the Neural NMF
            network, stored in the LsqNonneg class.

        Returns
        -------
        output: Pytorch Tensor
            The output of the nonnegative least squares task q(X,A)


        """

        [output, res] = lsqnonneg_tensor_version(A.data, input.data)
        # normalize the output
        #output_sum = torch.sum(output, dim =1) + 1e-10
        #output = output.t()/output_sum
        #output = output.t()
        #A.data = A.data.t()*output_sum
        #A.data = A.data.t()
        ctx.save_for_backward(input, A)
        #output = output.t()
        ctx.intermediate = output
        return output.t().t()
    
    @staticmethod
    @once_differentiable # don't know if this is needed, it seems like if without this line then in backprop all the operations should be differentiable
    def backward(ctx, grad_output):
        """
        Runs the backwards pass of the nonnegative least squares task q(X,A).
        Computes the gradients of q(X,A) with respect to X and A

        Parameters
        ----------
        ctx: context object
            Contains information stashed by the forwards pass


        grad_output: Pytorch Tensor
            The gradient of S = q(X,A) passed on from the last layer

        Returns
        -------
        grad_input: Pytorch Tensor
            The gradient of the nonnegative least squares task q(X,A)
            with respect to X

        grad_A: Pytorch Tensor
            The gradient of the nonnegative least squares task q(X,A)
            with respect to A

        """

        input, A = ctx.saved_tensors
        grad_input = grad_A = None
        output = ctx.intermediate
        if ctx.needs_input_grad[0]:
            grad_input = calc_grad_X(grad_output, A.data, output)# calculate gradient with respect to X
            #grad_input = grad_input.t()
        if ctx.needs_input_grad[1]:
            grad_A = calc_grad_A(grad_output, A.data, output, input.data) # calculate gradient with respect to A
            #grad_A = grad_A.t()
        return grad_input, grad_A




def lsqnonneg_tensor_version(A, X):
    """
    Calculates the nonnegative least squares solution q(X,A)

    Computes the following for each column of the output:
    output[:,i] = argmin_{s >= 0} ||X[:,i] - A*s||_F^2 

    Parameters
    ---------
    A: Pytorch Tensor
        The A matrix used to compute q(X,A)

    X: Pytorch Tensor
        The X matrix used to compute q(X,A)

    Returns
    -------
    S: Pytorch Tensor
        The S matrix, S = q(X,A)

    """
    A = A.numpy() # Transforming to numpy array size(m,k)
    X = X.numpy() # size(m,n)
    m = X.shape[0]
    n = X.shape[1]
    k = A.shape[1]
    S = np.zeros([k,n])
    res_total = 0
    for i in range(n):
        x = X[:,i]
        [s, res] = nnls(A, x)
        res_total += res
    S[:,i] = s
    S = torch.from_numpy(S).double() # Transforming to torch Tensor
    return S, res_total



def calc_grad_X(grad_S, A, S):
    """
    Calculates the gradient of q(X,A) with respect to X

    Parameters
    ---------
    grad_S: Pytorch Tensor
        The gradient of S = q(X,A) passed on from the last layer

    A: Pytorch Tensor
        The A matrix used to compute q(X,A)

    S: Pytorch Tensor
        The output S = q(X,A)

    Returns
    -------
    grad_X: Pytorch Tensor
        The gradient of q(X,A) with respsect to X

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
    Calculates the gradient of q(X,A) with respect to X

    Parameters
    ---------
    grad_S: Pytorch Tensor
        The gradient of S = q(X,A) passed on from the last layer

    A: Pytorch Tensor
        The A matrix used to compute q(X,A)

    S: Pytorch Tensor
        The output S = q(X,A)

    X: Pytorch Tensor
        The X matrix used to compute q(X,A)

    Returns
    -------
    grad_A: Pytorch Tensor
        The gradient of q(X,A) with respsect to A

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
    Defining a submodule 'LsqNonneg' of the nn.Module
    with network parameter: self.A which correspond to the A matrix in the NMF decomposition
    """
    def __init__(self, m, k, initial_A = None):
        """
        Initializes the LsqNonneg submodule

        Parameters
        ----------
        m: Integer
            The first domension of A, size (m,k)

        k: Integer
            The second domension of A, size (m,k)

        initial_A: Pytorch Tensor
            The initialization of the A matrix. If None,
            then A is generated randomly

        """
        super(LsqNonneg, self).__init__()
        self.m = m;
        self.k = k;
        self.A = nn.Parameter(torch.DoubleTensor(m,k)) #switched
        if initial_A is None:
            self.A.data = torch.abs(torch.randn(m,k,dtype = torch.double)) # initialize the network parameter #Switched
        else:
            self.A.data = initial_A
        
    def forward(self, input):
        """
        The forward pass of the LsqNonneg submodule

        Parameters
        ----------
        input: Pytorch Tensor
            The input the the NMF layer, X

        Returns
        -------
        S: Pytorch Tensor
            The output of the forward pass of the LsqNonnegF forward pass,
            which calculaetes S = q(X,A)
        """

        return LsqNonnegF.apply(input, self.A)




# from torch.autograd import gradcheck




# n = 10
# m = 10
# k = 5
# X_tensor = torch.randn(n,m, dtype = torch.double)
# A_tensor = torch.randn(k,m, dtype = torch.double)




# X = Variable(X_tensor, requires_grad=True)
# A = Variable(A_tensor, requires_grad = True)
# X = torch.abs(X)
# A = torch.abs(A)
# input = (X, A)
# test = gradcheck(LsqNonnegF().apply, input, eps = 1e-6, atol = 0, rtol = 1e-9)
# print(test)

