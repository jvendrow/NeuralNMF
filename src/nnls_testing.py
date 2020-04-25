#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import nnls

from time import time


nnls_times = []
dimensions = np.arange(10, 400, 20)


repetitions = 150

for d in dimensions:
    print(d)

    #define matrix A and vector x
    #------------------------------
    A = np.abs(np.random.rand(d, d))

    x = np.abs(np.random.rand(d))


    #Test the speed of scipy.optimize.nnls
    #-----------------------------------
    start = time()

    for i in range(repetitions):

        [s, res] = nnls(A, x)

    end = time()

    #Print and store results
    #-----------------------
    print(end - start)
    nnls_times.append(end-start)


plt.plot(dimensions, nnls_times)
plt.show()
