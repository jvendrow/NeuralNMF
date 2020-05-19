#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Functions for training (semi-supervised) hierarchical NMF model via Neural NMF.

    Examples
    --------
    >>> net = Neural_NMF([m, 9], none)                            #declares object from Neural NMF module
    >>> train(net, X, epoch=200, lr=1000)                            
    epoch =  10 
     tensor(228.1642, dtype=torch.float64)
    epoch =  20 
     tensor(228.0765, dtype=torch.float64)
    epoch =  30 
    ...
'''

from neural_nmf import Neural_NMF, Recon_Loss_Func, Energy_Loss_Func, Fro_Norm
from writer import Writer
import torch
import numpy as np
from lsqnonneg_module import LsqNonneg
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.autograd import Variable


def train(net, X, loss_func="Recon Loss", supervised=False, label=None, L=None, epoch=10, lr=1e-3, lr_classification=1e-3, weight_decay=1, class_iters=1, decay_epoch=1, verbose=True,verbose_epoch=1, full_history=False):

    """
    Training the Neural_NMF with projected gradient descent.
    
    Parameters
    ----------
    net: Pytorch Neural NMF module object
        The Neural NMF object to be trained. 
    X: PyTorch Tensor
        The data matrix input into the Neural NMF network (m x n).
    loss_func: string or Pytorch Module, optional
        The loss function. If the input is a string, uses one of the default loss functions 'Recon Loss' 
        or 'Energy Loss' (default is 'Recon Loss'). If the input is a Pytorch Module, uses that module, 
        allowing for custom loss functions.
    supervised: bool, optional
        Indicator for supervision (default is False).
    label: PyTorch tensor, optional
        The classification (label) matrix for supervised model.  If the classification_type is 'L2',
        this matrix is a one-hot encoding matrix of size c x n.  If the classification_type is
        'CrossEntropy', this matrix is of size n with elements in [0,c-1] (default is None).
    L: PyTorch tensor, optional
        The label indicator matrix for semi-supervised model that indicates if labels are known 
        for n data points, of size c x n with columns of all ones or all zeros to indicate if label
        for that data point is known (default is None).
    epoch: int_, optional
        Number of epochs in training procedure (default 10).
    lr: float_, optional
        The learning rate for the NMF layers (default 1e-3).
    lr_classification: float_, optional
        The learning rate for the classification layer (default 1e-3).
    weight_decay: float, optional
        The weight decay parameter, doing lr = lr*weight_decay every decay_epoch epochs (default 1).
    class_iters: int_, optional
        Number of gradient steps to take on classification term each epoch (default 1).
    decay_epoch: int_, optional
        Number of epochs to take before decaying the learning rates (default 1).
    verbose: bool, optional
        Indicator for whether to print the loss every verbose_epoch epochs (default True).
    verbose_epoch: int_, optional
        Number of epochs to take before printing the loss (default 1).
    full_history: bool, optional
        Indicator for whether to save all information from every epoch (default False).

    Returns
    -------
    history: Writer object or pair of lists of PyTorch tensors
        If Writer object (full_history=True), the history of the loss, A and S matrices, and A gradients 
        (if supervised training, also stores weights and gradients from the linear layer).
        If pair of lists of PyTorch tensors (full_history=False), stores the final A and S matrices and 
        A gradients from after last epoch.

    """
    if len(X.shape) != 2:
            raise ValueError("Expected a two-dimensional Tensor, but X is of shape {}".format(X.shape))

    if supervised == False:

        history = train_unsupervised(net, X, loss_func=loss_func, epoch = epoch, lr = lr, weight_decay = weight_decay, decay_epoch=decay_epoch, verbose=verbose, verbose_epoch=verbose_epoch, full_history = full_history)

    else:
        history = train_supervised(net, X, label=label, L=L, loss_func=loss_func, epoch = 10, lr_nmf = lr, lr_classification = lr_classification, weight_decay = weight_decay, class_iters=class_iters, decay_epoch=decay_epoch, verbose=verbose, verbose_epoch=verbose_epoch, full_history = full_history)

    return history



def train_unsupervised(net, X, loss_func="Recon Loss", epoch = 10, lr = 1e-3, weight_decay = 1, decay_epoch=1, verbose=True, verbose_epoch=1, full_history=False):
    """
    Training the unsupervised Neural_NMF with projected gradient descent.
    
    Parameters
    ----------
    net: Pytorch Neural NMF module object
        The Neural NMF object to be trained (since unsupervised, net.c = None). 
    X: PyTorch Tensor
        The data matrix input into the Neural NMF network (m x n).
    loss_func: string or Pytorch Module, optional
        The loss function. If the input is a string, uses one of the default loss functions 'Recon Loss' 
        or 'Energy Loss' (default is 'Recon Loss'). If the input is a Pytorch Module, uses that module, 
        allowing for custom loss functions. For unsupervised training, any custom loss function should 
        have lambd = None or 0.
    epoch: int_, optional
        Number of epochs in training procedure (default 10).
    lr: float_, optional
        The learning rate for the NMF layers (default 1e-3).
    weight_decay: float, optional
        The weight decay parameter, doing lr = lr*weight_decay every decay_epoch epochs (default 1).
    decay_epoch: int_, optional
        Number of epochs to take before decaying the learning rates (default 1).
    verbose: bool, optional
        Indicator for whether to print the loss every verbose_epoch epochs (default True).
    verbose_epoch: int_, optional
        Number of epochs to take before printing the loss (default 1).
    full_history: bool, optional
        Indicator for whether to save all information from every epoch (default False).

    Returns
    -------
    history: Writer object, optional
        If full_history=True, stores the history of the loss, A and S matrices, 
        and A gradients at each epoch.
    A_lst: list, optional
        If full_history=False, stores a list of the A PyTorch tensors, A_0, A_1, ... A_L.
    S_lst: list, optional
        If full_history=False, stores a list of the S PyTorch tensors, S_0, S_1, ... S_L.
        
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
            #S_lst = net(X)
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
                    config['v'] = torch.zeros_like(As[l].data)
                    config['a'] = torch.zeros_like(As[l].data)

            A = net.lsqnonneglst[l].A
            # record history
            if full_history:
                history.add_tensor('A'+str(l+1), A.data)
                history.add_tensor('grad_A'+str(l+1), A.grad.data)
                history.add_tensor('S' + str(l+1), S_lst[l].data)

            if not full_history and i == epoch-1:
               A_lst.append(A)

            # projected gradient descent
            A.data = A.data.sub_(lr*A.grad.data)
            #A.data, configs[l] = adam(A.data, A.grad.data, configs[l])
            A.data = A.data.clamp(min = 0)
        
        if verbose and (i+1)%verbose_epoch == 0:
            print('epoch = ', i+1, '\n', loss.data)
        if (i+1)%decay_epoch == 0:
            lr = lr*weight_decay

    if full_history:        
        return history
    else:
        return A_lst, S_lst

def train_supervised(net, X, label, L = None, loss_func="Recon Loss", epoch = 10, lr_nmf = 1e-3, lr_classification = 1e-3, weight_decay = 1, class_iters=1, decay_epoch=1, verbose=True, verbose_epoch=1, full_history=False):

    """
    Training the supervised Neural_NMF with projected gradient descent (PGD). In each epoch, we update the NMF
    layers with one PGD step and the classification layer with one PGD step.

    Parameters
    ----------
    net: Pytorch Neural NMF module object
        The Neural NMF object to be trained. 
    X: PyTorch tensor
        The data matrix input into the Neural NMF network (m x n).
    label: PyTorch tensor
        The classification (label) matrix for supervised model.  If the classification_type is 'L2',
        this matrix is a one-hot encoding matrix of size c x n.  If the classification_type is
        'CrossEntropy', this matrix is of size n with elements in [0,c-1] (default is None).
    L: PyTorch tensor, optional
        The label indicator matrix for semi-supervised model that indicates if labels are known 
        for n data points, of size c x n with columns of all ones or all zeros to indicate if label
        for that data point is known (default is None).
    loss_func: string or Pytorch Module, optional
        The loss function. If the input is a string, uses one of the default loss functions 'Recon Loss' 
        or 'Energy Loss' (default is 'Recon Loss'). If the input is a Pytorch Module, uses that module, 
        allowing for custom loss functions. 
    epoch: int_, optional
        Number of epochs in training procedure (default 10).
    lr_nmf: float_, optional
        The learning rate for the NMF layers (default 1e-3).
    lr_classification: float_, optional
        The learning rate for the classification layer (default 1e-3).
    weight_decay: float, optional
        The weight decay parameter, doing lr = lr*weight_decay every decay_epoch epochs (default 1).
    class_iters: int_, optional
        Number of PGD updates to make to classification layer each epoch (default 1).
    decay_epoch: int_, optional
        Number of epochs to take before decaying the learning rates (default 1).
    verbose: bool, optional
        Indicator for whether to print the loss every verbose_epoch epochs (default True).
    verbose_epoch: int_, optional
        Number of epochs to take before printing the loss (default 1).
    full_history: bool, optional
        Indicator for whether to save all information from every epoch (default False).

    Returns
    -------
    history: Writer object, optional
        If full_history=True, stores the history of the loss, A and S matrices, 
        and A gradients at each epoch.
    A_lst: list, optional
        If full_history=False, stores a list of the A PyTorch tensors, A_0, A_1, ... A_L.
    S_lst: list, optional
        If full_history=False, stores a list of the S PyTorch tensors, S_0, S_1, ... S_L.

    """

    loss_functions = {"Recon Loss": Recon_Loss_Func(), "Energy Loss": Energy_Loss_Func()}

    A_lst = []
    history = Writer() # creating a Writer object to record the history for the training process

    configs = [{} for i in range(net.depth-1)]
    config_w = {}

    for config in configs:
        config['learning_rate'] = lr_nmf
        config['beta1'] = 0.9
        config['beta2'] = 0.99
        config['epsilon'] = 1e-8
        config['t'] = 0

    config_w['learning_rate'] = lr_classification
    config_w['beta1'] = 0.9
    config_w['beta2'] = 0.99
    config_w['epsilon'] = 1e-8
    config_w['t'] = 0



    for i in range(epoch):
        
        # doing gradient update for NMF layer
        net.zero_grad()
        S_lst, pred = net(X)
        loss = None
        if type(loss_func) == str:
            loss = loss_functions[loss_func](net, X, S_lst)
        else:
            loss = loss_func(net, X, S_lst)
        loss.backward()
        for l in range(net.depth - 1):

            history.add_scalar('loss_nmf',loss.data)
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
            #return history
            #A.data, configs[l] = adam(A.data, A.grad.data, configs[l])
            #print(A.grad.data)
            A.data = A.data.clamp(min = 0)
            
        # doing gradient update for classification layer
        for iter_classifier in range(class_iters):
            net.zero_grad()
            S_lst, pred = net(X)
            history.add_tensor('pred', pred)
            loss = None
            if type(loss_func) == str:
                loss = loss_functions[loss_func](net, X, S_lst,pred,label,L)
            else:
                loss = loss_func(net, X, S_lst,pred,label,L)
            loss.backward()
            S_lst[0].detach()
            history.add_scalar('loss_classification',loss.data)
            weight = net.linear.weight
            weight.data = weight.data.sub_(lr_classification*weight.grad.data)
            #weight.data, config_w = adam(weight.data, weight.grad.data, config_w)

            weight.data = weight.data.clamp(min = 0) #added by Jamie (shouldn't we make sure B >= 0?)
            if full_history:
                history.add_tensor('weight', weight.data.clone())
                history.add_tensor('grad_weight', weight.grad.data.clone())
        
        
        if(verbose) and (i+1)%verbose_epoch == 0:
            print('epoch = ', i+1, '\n', loss.data)
        if (i+1)%decay_epoch == 0:
            lr_nmf = lr_nmf*weight_decay
            lr_classification = lr_classification*weight_decay

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
