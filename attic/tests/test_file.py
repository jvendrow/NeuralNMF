import pytest

import torch
from neural_nmf import Neural_NMF
from neural_nmf import train

def test():

    X = 10*torch.mm(torch.randn(100,5),torch.randn(5,20)) #produce random low rank data
    m, k1, k2, = X.shape[0], 10, 5
    net = Neural_NMF([m, k1, k2])

    history = train(net, X, epoch=6, lr=500, supervised=False)

test()
