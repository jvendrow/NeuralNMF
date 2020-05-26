
<p align="center">
<img width="600" src="https://github.com/jvendrow/NeuralNMF/blob/master/Neural%20NMF%20Logo.png?raw=true" alt="logo">
</p>

# Neural NMF

[![PyPI Version](https://img.shields.io/pypi/v/neuralnmf.svg)](https://pypi.org/project/neuralnmf/)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/neuralnmf.svg)](https://pypi.org/project/neuralnmf/)

This package is an implementation of `Neural NMF`, a method for detecting latent hierarchical structure in data based on non-negative matrix factorization, as presented in the paper "Neural Nonnegative Matrix Factorization for Hierarchical Multilayer Topic Modeling" by T. Will, R. Zhang, E. Sadovnik, M. Gao, J. Vendrow, J. Haddock, D. Molitor, and D. Needell (2020).

Neural NMF solve a hierarchical nonnegative matrix factorization problem by representing the problem with a neural network architecture and applying backpropagation methods. In the unsupervised case, Neural NMF applies backprogation directly to the given loss function (usually either Energy Loss or Reconstruction Loss). In the supervised case, Neural NMF adds a linear layer to the last **S** matrix to estimate the given labels. 

---

## Installation

To install `Neural NMF`, run this command in your terminal:

```bash
$ pip install neuralnmf
```

This is the preferred method to install `Neural NMF`, as it will always install the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, these [installation instructions](http://docs.python-guide.org/en/latest/starting/installation/) can guide
you through the process.

## Usage

**Quick Start**

To use `Neural NMF`, we first initialize our neural network with the layer sizes, and if applicable, the number of classes. We give the layer sizes as a list, where the first element is the 2nd dimension of the input matrix and each following dimensions is the rank of the approximation at the following layer. 
```python
>>> import torch
>>> from neural_nmf import Neural_NMF
>>> X = 10*torch.mm(torch.randn(100,5),torch.randn(5,20)) #produce random low rank data
>>> m, k1, k2, = X.shape[0], 10, 5
>>> net = Neural_NMF([m, k1, k2])
```
One we have initialized our network, we train it using the *train* function (See the documentation in train.py for specific details of every optional parameter). 
```python
>>> from train import train
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





-----------------------------

Old README:


Full Backward Propagation Package README:

INTRO:
    The Full Backward Propagation Package is an implementation for our group's own neural NMF backprop algorithm.
    
    The heart of the algorithm is finding a way to solve and differentiate the nonlinear function q which is defined by a convex optimization problem: q(X,A) = argmin_{S>=0} ||X - AS||_F^2. 
    In this pakage we turn the nonlinear function q into a subclass of nn.Module. The class name is 'Lsqnonneg'. This means that you can use Lsqnonneg as the same as calling nn.Linear or nn.Conv2d etc. Thus you can treat it as a module in the network and can combine it with all kinds of different structures in the nerual net. In general, there are all kinds of possibilities how this module can be embedded into a network, and you can play around with it to find out the numerical performance for this module. The code is in 'lsqnonneg_module.py'.(See more in Network Structure Section)
    
    'Training section' defines the training process for the network. Currently we are just using naive gradient descent to optimize the network. And we are still trying to exploit other first order optimization method. The 'Train.Ipynb' defines a standard training function for the network, but if you want more training flexibility, you can write your own training function, which should not be too hard.
    
    In 'History Recording Section' I write a 'Writer' class to record the history of the training process (e.g. gradient of A, value of A, total loss etc). This might make it easier for further analysis into the behavior of the algorithm. But if you want to use other ways to record history information or do not have the need of recording anything, you can skip this section.
    
    'Data Loading Section' is simply created for loading the training data and relevant information.
    
    Try running demo.py to see how these sections coordinates with each other! :)
 
 NOTE: all the variables in the network should be torch.DoubleTensor instead of torch.FloatTensor!
    
NETWORK STRUCTURE SECTION:
    deep_nmf.py: Currently this is the thing might be most frequently used. The basic structure is simply piling up the Lsqnonneg module (and add a classification layer at the top if it is a supervised or semisupervised task). Details are contained in the comments in this notebook.
    lsqnonneg_module.Ipynb: This is the basic module for the whole package. It defines the forward and backward process of the nonlinear function q. (Here the forward process means solving the optimization problem: min_{S>=0} ||X - AS||_F^2, and the backward process means find the derivative of q with respect to A and X)
    
TRAINING SECTION:
    train.py: Defines the training process of unsupervised and supervised(or semisupervised) learning. The optimization algorithm is simply projection gradient descent. The two functions are called 'train_unsupervised' and 'train_supervised'.

HISTORY RECORDING SECTION:
    writer.py: Creates a history recording class. This class is only designed for debugging and analyzing the result. If you don't need it or can find better ways to do this, then simply ignore this section.

DATA LOADING SECTION:
    data_loading.py:Loading the bag of words dataset.

DEMO:
    demo.py: A demo for training unsupervised, one-layer NMF and supervised, one_layer NMF.
