import numpy as np

def fnnls(Z, x):
    """
    Implementation of the Fast Non-megative Least Squares Algorithm described
    in the paper "A fast non-negativity-constrained least squares algorithm"
    by Rasmus Bro and Sijmen De Jong

    This algorithm seeks to find min_d ||x - Zd||^2 subject to d >= 0

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

    m, n = Z.shape


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

    tolerance = 10 * 2.2204e-16 * np.linalg.norm(ZTZ) * max(n,m)

    max_iter_out = 0.1*n
    max_iter_in = n
    #B1
    i = 0
    while len(R) and np.max(w[list(R)]) > tolerance and i < max_iter_in:
        i += 2
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
            j += 1
            #C2
            alpha = np.min(d / (d-s + 1e-8))
            #C3
            d = d + alpha * (s-d)
            #C4
            #passive = set(np.asarray(P_ind)[s[P_ind] <= tolerance])
            passive = {p for p in P_ind if s[p] <= tolerance}

            P = passive
            R = {s for s in range(0,n) if s not in passive}
            P_ind = list(P)
            R_ind = list(R)
            #C5
            s[P_ind] = np.linalg.lstsq((ZTZ)[P_ind][:,P_ind], (ZTx)[P_ind], rcond=None)[0]
            #c6
            s[R_ind] = np.zeros(len(R))

        #B5
        d = s
        w = ZTx - (ZTZ) @ d

    res = np.linalg.norm(x - Z@d)
    return [d, res]
