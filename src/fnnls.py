import numpy as np
import scipy
from scipy.linalg import lstsq
from time import time

def fnnls(Z, x, maxiter=10, P_init = set()):
    """
    Implementation of the Fast Non-megative Least Squares Algorithm described
    in the paper "A fast non-negativity-constrained least squares algorithm"
    by Rasmus Bro and Sijmen De Jong

    This algorithm seeks to find min_d ||x - Zd|| subject to d >= 0

    Parameters
    ----------
    Z: NumPy array
        Z is an m x n matrix

    x: Numpy array
        x is a m x 1 vector

    Returns
    -------
    d: Numpy array
        d is a nx1 vector
    """

    Z, x = map(np.asarray_chkfinite, (Z, x))

    """
    if type(Z) is not np.ndarray:
        raise TypeError("Expected a NumPy array, but Z is of type {}".format(type(Z)))

    if type(x) is not np.ndarray:
        raise TypeError("Expected a NumPy array, but x is of type {}".format(type(x)))

    """

    if len(Z.shape) != 2:
        raise ValueError("Expected a two-dimensional array, but Z is of shape {}".format(Z.shape))
    if len(x.shape) != 1:
        raise ValueError("Expected a one-dimensional array, but x is of shape {}".format(x.shape))

    m, n = Z.shape

    if x.shape[0] != m:
        raise ValueError("Incompatable dimensions. The first dimension of Z should match the length of x, but Z is of shape {} and x is of shape {}".format(Z.shape, x.shape))

    ZTZ = Z.T.dot(Z)
    ZTx = Z.T.dot(x)

    #A1
    P = P_init.copy()
    #A2
    R = {i for i in range(0,n) if i not in P_init}

    R_ind = list(R)
    #A3
    d = np.zeros(n)
    #A4
    w = ZTx - (ZTZ) @ d

    s = np.zeros(n)

    epsilon = 2.2204e-16

    tolerance = epsilon * np.linalg.norm(ZTZ, ord=1) * n

    max_iter_in = 100*n

    #B1
    no_update = 0

    times_out = list()
    times = list()
    i = 0

    while len(R) and np.max(w[R_ind]) > tolerance and i < max_iter_in:
        i += 1

        current_passive = P.copy() #make copy of passive set to check for change at end of loop

        #B2 
        ind = R_ind[np.argmax(w[R_ind])]

        #B3
        P.add(ind)
        R.remove(ind)

        P_ind = list(P)
        R_ind = list(R)

        #B4
        #s[P_ind] = np.linalg.lstsq((ZTZ)[P_ind][:,P_ind], (ZTx)[P_ind], rcond=None)[0]
        s[P_ind] = lstsq((ZTZ)[P_ind][:,P_ind], (ZTx)[P_ind], lapack_driver='gelsy')[0]
        #if len(P_ind) > 1:
        #    s[P_ind] = np.linalg.inv((ZTZ)[P_ind][:,P_ind]).dot((ZTx)[P_ind])
        #else:
        #    s[P_ind] = ((ZTx)[P_ind]) / (ZTZ)[P_ind][:,P_ind]

        #C1
        j = 0

        while len(P) and np.min(s[P_ind]) <= tolerance and j < max_iter_in:
            times.append(time())
            #C2
            q = [i for i in P_ind if s[i] <= tolerance]
            alpha = np.min(d[q] / (d[q] - s[q]))

            times.append(time())
            #C3
            d = d + alpha * (s-d) #set d as close to s as possible while maintaining non-negativity
            #C4
            passive = {p for p in P_ind if s[p] <= tolerance}


            P.difference_update(passive)
            R.update(passive)
            P_ind = list(P)
            R_ind = list(R)


            #C5
            s[P_ind] = lstsq((ZTZ)[P_ind][:,P_ind], (ZTx)[P_ind], lapack_driver='gelsy')[0]

            #C6
            s[R_ind] = np.zeros(len(R))

            j += 1

        #B5
        d = s.copy() 
        w = ZTx - (ZTZ) @ d

        if(current_passive == P): #check of there has been a check to the passive set
            no_update += 1
            break
        else:
            no_update = 0

        lst = [times_out[i+1] - times_out[i] for i in range(len(times_out)-1)]
        times_out = list()


    res = np.linalg.norm(x - Z@d) #Calculate residual loss ||x - Zd||

    
    return [d, res]
