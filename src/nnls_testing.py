#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import nnls

from time import time


class nnls_testing():

    def __init__(self):

        return

    def test(self, repetitions, dimensions, optimizers, names, verbose=True):
        """
        Measures the speed of each optimizer at each dimensions,
        averaged over repetitions

        Parameters
        ----------
        repetitions: int
            The number of times to run at optimizer at each dimensions
            for precision. 

        dimensions: numpy array
            A list of dimensions of matrix A and vector x to use 
            for testing.

        optimizers: list
            A list of nonnegative least squares functions that
            we will test the complexity of.

        names: list
            A list corresponding to the optimizers list of what
            to call the optimizers for plotting

        """

        nnls_times = []

        for d in dimensions:


            for index, optimizer in enumerate(optimizers):

                #A = np.abs(np.random.rand(d, d))

                #x = np.abs(np.random.rand(d))

                time_total = 0

                for i in range(repetitions):

                    #define matrix A and vector x
                    #------------------------------

                    A = np.abs(np.random.rand(d, d))

                    x = np.abs(np.random.rand(d))

                    #Measure the speed of running the optimizer
                    start = time()

                    [s, res] = optimizer(A, x)

                    end = time()
                
                    time_total += end - start

                #Print and store results
                #-----------------------
                if verbose:
                    print(d)
                    print(names[index] + ": " + str(time_total))

                if d == dimensions[0]:
                    nnls_times.append([time_total])

                else:
                    nnls_times[index].append(time_total)

            self.repetitions = repetitions
            self.dimensions = dimensions
            self.names = names
            self.nnls_times = nnls_times

    def plot(self):
        """
        Plots the time complexities of each optimizer

        """
        #Plot the times for nnls
        #-----------------------
        for nnls_time in self.nnls_times:
            plt.plot(self.dimensions, nnls_time)

        plt.xlabel("dimension")
        plt.ylabel("time (s) for " +  str(self.repetitions) + " runs")
        plt.legend(self.names)

        plt.show()


testing = nnls_testing()

repetitions = 25
dimensions = np.arange(20, 400, 20)
optimizers = [nnls, nnls]
names = ["scipy.optimize.nnls1", "scipy.optimize.nnls2"]

testing.test(repetitions, dimensions, optimizers, names, verbose=True)

testing.plot()
