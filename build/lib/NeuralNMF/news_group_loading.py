#!/usr/bin/env python
# coding: utf-8



import scipy.io as sio
import torch
from torch.autograd import Variable
import numpy as np


def get_data():
    data = sio.loadmat('../../data/20News_subsampled.mat')
    X = data.get("X_subsampled")
    Y_sub = data.get("Ysub")
    Y_super = data.get("Ysuper")
    vocab = data.get("vocab")

    X = Variable(torch.from_numpy(X.toarray()))
    Y_sub = torch.from_numpy(np.argmax(Y_sub, axis=0))
    Y_super = torch.from_numpy(np.argmax(Y_super, axis=0))



    return X, Y_sub, Y_super, vocab
