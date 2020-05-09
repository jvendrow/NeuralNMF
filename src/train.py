#!/usr/bin/env python
# coding: utf-8



# Date: 2018.07.22
# Author: Runyu Zhang




#import Ipynb_importer
#from neural_nmf_sup import Neural_NMF, Recon_Loss_Func, Energy_Loss_Func, Fro_Norm
from neural_nmf_sup import Neural_NMF, Recon_Loss_Func, Energy_Loss_Func, Fro_Norm

from writer import Writer
import torch
import numpy as np
#from lsqnonneg_module_sup import LsqNonneg
from lsqnonneg_module_sup import LsqNonneg
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.autograd import Variable


def train(net, X, loss_func="Recon Loss", supervised=False, label=None, L=None, epoch=10, lr=1e-3, lr_classification=1e-3, weight_decay=1, verbose=True, full_history=False):

    """
    Training the unsupervised Neural_NMF with projection gradient descent.
    
    Parameters
    ----------
    net: Pytorch Module
        The Neural_NMF object to be trained. Note that it should 
        be the unsupervised version, so c = None for the Neural_NMF.

    X: PyTorch Tensor
        The data matrix to input into the Neural NMF network

    loss_func: string or Pytorch Module
        The loss function. If the input is a string, uses one
        of the default loss functions. If the input is a Pytorch
        Module, uses that module, allowing for custom loss functions.
        Default loss functions:
            'Recon Loss': Reconstruction Loss Function
            'Energy Loss': Energy Loss Functions

    epoch: integer
        How many time you want to feed in the data matrix to the network, default 10

    lr: float
        The learning rate, default 1e-3.

    lr_classification: float
        The learning rate for the classification layer, default 1e-3.

    weight_decay: float
        The weight decay parameter, doing lr = lr*weight_decay every epoch.

    Returns
    -------
    history: Writer object
        Stores the history of the loss, A and S matrices, and A gradients.
        If supervised training, also stores weights and gradients from
        the linear layer.
        If full_history=False, only stores the final A and S matrices and 
        A gradient from after last epoch.

    """
    if supervised == False:

        history = train_unsupervised(net, X, loss_func=loss_func, epoch = epoch, lr = lr, weight_decay = weight_decary, verbose=verbose, full_history = full_history)

    else:
        history = train_supervised(net, X, label=label, L=L, loss_func=loss_func, epoch = 10, lr_nmf = lr, lr_classification = lr_classification, weight_decay = weight_decay, verbose=verbose, full_history = full_history)

    return history



def train_unsupervised(net, X, loss_func="Recon Loss", epoch = 10, lr = 1e-3, weight_decay = 1, verbose=True, full_history=False):
    """
    Training the unsupervised Neural_NMF with projection gradient descent.
    
    Parameters
    ----------
    net: Pytorch Module
        The Neural_NMF object to be trained. Note that it should 
        be the unsupervised version, so c = None for the Neural_NMF.

    X: PyTorch Tensor
        The data matrix to input into the Neural NMF network.

        The loss function This loss function should have lambd = None or 0

    loss_func: string or Pytorch Module
        The loss function. If the input is a string, uses one
        of the default loss functions. If the input is a Pytorch
        Module, uses that module, allowing for custom loss functions.
        Default loss functions:
            'Recon Loss': Reconstruction Loss Function
            'Energy Loss': Energy Loss Functions
        For unsupervised training, any custom loss function
        should have lambd = None or 0.


    epoch: integer
        How many time you want to feed in the data matrix to the network, default 10.

    lr: float
        The learning rate, default 1e-3.

    weight_decay: float
        The weight decay parameter, doing lr = lr*weight_decay every epoch.

    verbose: bool
        If true, display the current loss at select epochs.

    Returns
    -------
    history: Writer object
        If full_history=True, stores the history of the loss, A and S matrices, 
        and A gradients at each epoch.

    A_lst: list
        If full_history=False, stores a list of the A matries,
        A_0, A_1, ... A_L.

    S_lst: list
        If full_history=False, stores a list of the S matries,
        S_0, S_1, ... S_L.

        
    """

    loss_functions = {"Recon Loss": Recon_Loss_Func(), "Energy Loss": Energy_Loss_Func()}

    A_lst = []
    history = Writer() # creating a Writer object to record the history for the training process

    configs = [{} for i in range(net.depth-1)]

    for config in configs:
        config['learning_rate'] = 10e4
        config['beta1'] = 0.9
        config['beta2'] = 0.99
        config['epsilon'] = 1e-8
        config['t'] = 0

    S_lst = None
    for i in range(epoch):
        net.zero_grad()
        if S_lst != None:
            S_lst = net(X, [s.detach() for s in S_lst])
        else:
            S_lst = net(X)
        loss = None
        if type(loss_func) == str:
            loss = loss_functions[loss_func](net, X, S_lst)
        else:
            loss = loss_func(net, X, S_lst)
        loss.backward()
        history.add_scalar('loss', loss.data)
        for l in range(net.depth - 1):

            if epoch == 0:
                for config in configs:
                    config['v'] = torch.zeros_like(As[j].data)
                    config['a'] = torch.zeros_like(As[j].data)

            A = net.lsqnonneglst[l].A
            # record history
            if full_history:
                history.add_tensor('A'+str(l+1), A.data)
                history.add_tensor('grad_A'+str(l+1), A.grad.data)
                history.add_tensor('S' + str(l+1), S_lst[l].data)

            if not full_history and i == epoch-1:
                A_lst.append(A)

            # projection gradient descent
            A.data = A.data.sub_(lr*A.grad.data)
            #A.data, configs[l] = adam(A.data, A.grad.data, configs[l])
            A.data = A.data.clamp(min = 0)
        lr = lr*weight_decay
        if verbose and (i+1)%1 == 0:
            print('epoch = ', i+1, '\n', loss.data)

    if full_history:        
        return history
    else:
        return A_lst, S_lst

def train_supervised(net, X, label, L = None, loss_func="Recon Loss", epoch = 10, lr_nmf = 1e-3, lr_classification = 1e-3, weight_decay = 1, verbose=True, full_history=False):

    """
    Training the supervised Neural_NMF with projection gradient descent.

    For each epoch we update the NMF layer and the classification layer separately. First update the NMF layer for once
    and then update the classification layer for thirty times. The learning rate is 

    
    Parameters
    ----------
    net: Pytorch Module
        The Neural_NMF object to be trained. Note that it should 
        be the unsupervised version, so c = None for the Neural_NMF.

    X: PyTorch Tensor
        The data matrix to input into the Neural NMF network

    loss_func: string or Pytorch Module
        The loss function. If the input is a string, uses one
        of the default loss functions. If the input is a Pytorch
        Module, uses that module, allowing for custom loss functions.
        Default loss functions:
            'Recon Loss': Reconstruction Loss Function
            'Energy Loss': Energy Loss Functions

    label: ?

    L: ?

    epoch: integer
        How many time you want to feed in the data matrix to the network, default 10

    lr: float
        The learning rate for the NMF layers, default 1e-3

    lr_classification: float
        The learning rate for the classification layer, default 1e-3

    weight_decay: float
        The weight decay parameter, doing lr = lr*weight_decay every epoch

    verbose: bool
        If true, display the current loss at every epoch

    Returns
    -------
     history: Writer object
        If full_history=True, stores the history of the loss, A and S matrices, 
        A gradients, and the weights and graduents of the linear layer used for
        classification at each epoch.

    A_lst: list
        If full_history=False, stores a list of the A matries,
        A_0, A_1, ... A_L.

    S_lst: list
        If full_history=False, stores a list of the S matries,
        S_0, S_1, ... S_L.

    """

    A_lst = []
    history = Writer() # creating a Writer object to record the history for the training process
    for i in range(epoch):
        
        # doing gradient update for NMF layer
        net.zero_grad()
        S_lst, pred = net(X)
        loss = loss_func(net, X, S_lst, pred, label, L)
        loss.backward()
        for l in range(net.depth - 1):
            history.add_scalar('loss',loss.data)
            A = net.lsqnonneglst[l].A
            # record history
            if full_history:
                history.add_tensor('A'+str(l+1), A.data)
                history.add_tensor('grad_A'+str(l+1), A.grad.data)
                history.add_tensor('S' + str(l+1), S_lst[l].data)

            if not full_history and i == epoch-1:
                A_lst.append(A)
            # projection gradient descent
            A.data = A.data.sub_(lr_nmf*A.grad.data)
            A.data = A.data.clamp(min = 0)
            
        # doing gradient update for classification layer
        for iter_classifier in range(30):
            net.zero_grad()
            S_lst, pred = net(X)
            loss = loss_func(net, X, S_lst, pred, label, L)
            loss.backward()
            S_lst[0].detach()
            history.add_scalar('loss',loss.data)
            weight = net.linear.weight
            weight.data = weight.data.sub_(lr_classification*weight.grad.data)
            if full_history:
                history.add_tensor('weight', weight.data.clone())
                history.add_tensor('grad_weight', weight.grad.data.clone())
        
        
        lr_nmf = lr_nmf*weight_decay
        lr_classification = lr_classification*weight_decay
        if(verbose):
            print('epoch = ', i+1, '\n', loss.data)

    if full_history:        
        return history
    else:
        return A_lst, S_lst



def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('v', torch.zeros_like(w))
    config.setdefault('a', torch.zeros_like(w))
    config.setdefault('t', 0)
    
    next_w = None

    v = config['v']
    beta1 = config['beta1']
    beta2 = config['beta2']
    rate = config['learning_rate']
    a = config['a']
    e = config['epsilon']
    t = config['t'] + 1

    nu = 1e-8
    v = beta1 * v + (1 - beta1) * dw

    a = beta2 * a + (1 - beta2) * dw * dw 

    v_c  = v * 1 / (1-beta1**t)
    a_c = a * 1 / (1-beta2**t)


    next_w = w - rate * v_c / (np.sqrt(a_c) + e)


    config['v'] = v
    config['a'] = a
    config['t']  = t
    
    return next_w, config
