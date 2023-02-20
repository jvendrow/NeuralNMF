
<p align="center">
<img width="600" src="https://github.com/jvendrow/NeuralNMF/blob/master/Neural%20NMF%20Logo.png?raw=true" alt="logo">
</p>

# Neural NMF

[![PyPI Version](https://img.shields.io/pypi/v/neuralnmf.svg)](https://pypi.org/project/neuralnmf/)

This package is an implementation of `Neural NMF`, a method for detecting latent hierarchical structure in data based on non-negative matrix factorization, as presented in the paper "Neural Nonnegative Matrix Factorization for Hierarchical Multilayer Topic Modeling" by T. Will, R. Zhang, E. Sadovnik, M. Gao, J. Vendrow, J. Haddock, D. Molitor, and D. Needell (2020).

Neural NMF solve a hierarchical nonnegative matrix factorization problem by representing the problem with a neural network architecture and applying backpropagation methods. In the unsupervised case, Neural NMF applies backprogation directly to the given loss function (usually either Energy Loss or Reconstruction Loss). In the supervised case, Neural NMF adds a linear layer to the last **S** matrix to estimate the given labels. 

---

## Installation

To install `Neural NMF`, run this command in your terminal:

```bash
$ pip install NeuralNMF
```

This is the preferred method to install `Neural NMF`, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, these [installation instructions](http://docs.python-guide.org/en/latest/starting/installation/) can guide
you through the process.

## Usage

**Quick Start**

To use `Neural NMF`, we first initialize our neural network with the layer sizes, and if applicable, the number of classes. We give the layer sizes as a list, where the first element is the 2nd dimension of the input matrix and each following dimensions is the rank of the approximation at the following layer. 
```python
>>> import torch
>>> from NeuralNMF import Neural_NMF
>>> X = 10*torch.mm(torch.randn(100,5),torch.randn(5,20)) #produce random low rank data
>>> m, k1, k2, = X.shape[0], 10, 5
>>> net = Neural_NMF([m, k1, k2])
```
One we have initialized our network, we train it using the *train* function (See the documentation in train.py for specific details of every optional parameter). 
```python
>>> from NeuralNMF import train
>>> history = train(net, X, epoch=6, lr=500, supervised=False)
epoch =  1 
 tensor(485.2435, dtype=torch.float64)
epoch =  2 
 tensor(475.1584, dtype=torch.float64)
epoch =  3 
 tensor(461.2400, dtype=torch.float64)
epoch =  4 
 tensor(444.1705, dtype=torch.float64)
epoch =  5 
 tensor(430.4947, dtype=torch.float64)
epoch =  6 
 tensor(422.7317, dtype=torch.float64)

```

## Citing
If you use our code in an academic setting, please consider citing our code by citing the following paper: 

Will, T., Zhang, R., Sadovnik, E., Gao, M., Vendrow, J., Haddock, J., Molitor, D., & Needell, D. (2020). Neural nonnegative matrix factorization for hierarchical multilayer topic modeling. 

## Authors
* Joshua Vendrow
* Jamie Haddock