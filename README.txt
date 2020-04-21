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
