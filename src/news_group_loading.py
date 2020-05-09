#!/usr/bin/env python
# coding: utf-8



import scipy.io as sio
import torch
from torch.autograd import Variable
import numpy as np


def get_data():
    data = sio.loadmat('../data/20News_subsampled.mat')
    X = data.get("X_subsampled")
    Y_sub = data.get("Ysub")
    Y_super =data.get("Ysuper")

    X = Variable(torch.from_numpy(X.toarray()).double())

    return X, Y_sub, Y_super
