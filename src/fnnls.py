import numpy as np

def fnnls(Z, x, maxiter=10):
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

    ZTZ = Z.T @ Z

    ZTx = Z.T @ x

    #A1
    P = set()
    #A2
    R = {i for i in range(0,n)}
    #A3
    d = np.zeros(n)
    #A4
    w = ZTx - (ZTZ) @ d

    s = np.zeros(n)

    epsilon = 2.2204e-16
    tolerance = 10 * epsilon * np.linalg.norm(ZTZ) * max(n,m)

    max_iter_in = 30*n

    #B1
    i = 0

    no_update = 0

    while len(R) and np.max(w[list(R)]) > tolerance:

        current_passive = P.copy() #make copy of passive set to check for change at end of loop

        #B2 
        ind = list(R)[np.argmax(w[list(R)])]
        #B3
        P.add(ind)
        R.remove(ind)

        P_ind = list(P)
        #B4
        s[P_ind] = np.linalg.lstsq((ZTZ)[P_ind][:,P_ind], (ZTx)[P_ind], rcond=None)[0]
        
        #C1
        j = 0
        while np.min(s[P_ind]) <= tolerance and j < max_iter_in:

            #C2
            alpha = np.min(d[P_ind] / d[P_ind] - s[P_ind])
            #C3
            #d = d + alpha * (s-d) #set d as close to s as possible while maintaining non-negativity
            #C4
            #passive = set(np.asarray(P_ind)[s[P_ind] <= tolerance])
            passive = {p for p in P_ind if s[p] <= tolerance}

            #P = passive
            #R = {s for s in range(0,n) if s not in passive}
            P.difference_update(passive)
            R.update(passive)
            P_ind = list(P)
            R_ind = list(R)
            #C5
            s[P_ind] = np.linalg.lstsq((ZTZ)[P_ind][:,P_ind], (ZTx)[P_ind], rcond=None)[0]
            #c6
            s[R_ind] = np.zeros(len(R))

            j += 1

        #B5
        d = s
        w = ZTx - (ZTZ) @ d

        if(current_passive == P): #check of there has been a check to the passive set
            no_update += 1
        else:
            no_update = 0

        if no_update > 5: #if the passive set hasn't change for multiple iterations, end
            break

    res = np.linalg.norm(x - Z@d) #Calculate residual loss ||x - Zd||

    return [d, res]
