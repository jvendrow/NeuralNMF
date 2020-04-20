#!/usr/bin/env python
# coding: utf-8



# Date: 2018.07.22
# Author: Runyu Zhang




#import Ipynb_importer
from deep_nmf import Deep_NMF, Energy_Loss_Func, Fro_Norm
from writer import Writer
import torch
import numpy as np
from lsqnonneg_module import LsqNonneg
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.autograd import Variable




def train_unsupervised(net, X, loss_func, epoch = 10, lr = 1e-3, weight_decay = 1):
    '''
    ----- Discription
    Training the unsupervised Deep_NMF with projection gradient descent
    ----- Inputs:
    net: A Deep_NMF object, note that it should be the unsupervised version, so c = None for the Deep_NMF.
    X: The data matrix.
    loss_func: The loss function, should be a Energy_Loss_func object, with lambd = None or 0
    epoch: How many time you want to feed in the data matrix to the network, default 10
    lr: learning rate, default 1e-3
    weight_decay: the weight decay parameter, doing lr = lr*weight_decay every epoch
    '''
    history = Writer() # creating a Writer object to record the history for the training process
    for i in range(epoch):
        net.zero_grad()
        S_lst = net(X)
        loss = loss_func(net, X, S_lst)
        loss.backward()
        history.add_scalar('loss', loss.data)
        for l in range(net.depth - 1):
            A = net.lsqnonneglst[l].A
            # record history
            history.add_tensor('A'+str(l+1), A.data)
            history.add_tensor('grad_A'+str(l+1), A.grad.data)
            history.add_tensor('S' + str(l+1), S_lst[l].data)
            # projection gradient descent
            A.data = A.data.sub_(lr*A.grad.data)
            A.data = A.data.clamp(min = 0)
        lr = lr*weight_decay
        if (i+1)%10 == 0:
            print('epoch = ', i+1, '\n', loss.data)
    return history




def train_supervised(net, X, loss_func, label, L= None, epoch = 10, lr_nmf = 1e-3, lr_classification = 1e-3, weight_decay = 1):
    '''
    ---- Description
    Training the supervised Deep_NMF with projection gradient descent. Details for the training process:
        for each epoch we update the NMF layer and the classification layer separately. First update the NMF layer for once
        and then update the classification layer for thirty times. The learning rate is 
    ---- Inputs:
    net: A Deep_NMF object, note that it should be the unsupervised version, so c = None for the Deep_NMF.
    X: The data matrix.
    epoch: How many time you want to feed in the data matrix to the network, default 10
    loss_func: The loss function, should be a Energy_Loss_func object
    epoch: How many time you want to feed in the data matrix to the network, default 10
    lr_nmf: the learning rate for the NMF layer
    lr_classification: the learning rate for the classification layer
    weight_decay: the weight decay parameter, doing lr = lr*weight_decay every epoch
    '''
    
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
            history.add_tensor('A'+str(l+1), A.data)
            history.add_tensor('grad_A'+str(l+1), A.grad.data)
            history.add_tensor('S' + str(l+1), S_lst[l].data)
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
            history.add_tensor('weight', weight.data.clone())
            history.add_tensor('grad_weight', weight.grad.data.clone())
        
        
        lr_nmf = lr_nmf*weight_decay
        lr_classification = lr_classification*weight_decay
        
        print('epoch = ', i+1, '\n', loss.data)
    return history






