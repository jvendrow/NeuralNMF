#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Class and functions for recording and visualizing Neural NMF results.
'''

from matplotlib import pyplot as plt


class Writer:

    """
    Records a history of the Neural NMF epochs in dictionaries and visualizes various information.

    Methods
    ----------
    add_scalar(name, scalar)
        Adds a value to the list of scalars stored under the proper key in the scalar dictionary.
    add_tensor(name, tensor)
        Adds a PyTorch tensor to the list of tensors stored under the proper key in the tensor dictionary.
    get(name)
        Fetches the list of scalars or tensors stored under name.
    plot_scalar(name)
        Plots the list of scalars stored in the scalar dictionary under the key name.
    plot_tensor(name)
        Plots the tensor stored in the idx_lst index of the list in the tensor dictionary corresponding
        to the key name.
    """

    def __init__(self):
        self.scalar_dict = {}
        self.tensor_dict = {}
        
    def add_scalar(self, name, scalar):
        """
        Adds a value to the list of scalars stored under the proper key in the scalar dictionary.

        Parameters
        ----------
        name: string
            The key value for accessing the scalar dictionary.
        scalar: float_
            The value to add to the list stored under scalar_dict[name].

        """
        if self.scalar_dict.get(name) is None:
            self.scalar_dict[name] = [scalar]
        else:
            self.scalar_dict[name].append(scalar)
            
    def add_tensor(self, name, tensor):
        """
        Adds a PyTorch tensor to the list of tensors stored under the proper key in the tensor dictionary.

        Parameters
        ----------
        name: string
            The key value for accessing the tensor dictionary.
        scalar: PyTorch tensor
            The tensor to add to the list stored under tensor_dict[name].

        """

        if self.tensor_dict.get(name) is None:
            self.tensor_dict[name] = [tensor]
        else:
            self.tensor_dict[name].append(tensor)
            
    def get(self, name):
        """
        Fetches the list of scalars or tensors stored under name. Searches the scalar list before the 
        tensor list.

        Parameters
        ----------
        name: string
            The key by which to search the scalar and tensor dictionaries. 

        Returns
        -------
        values: list, optional
            The list of scalars or tensors that correspond to name, the key value. If key value is not
            in either dictionary, returns None.

        """

        scalar =  self.scalar_dict.get(name)
        if scalar is not None:
            return scalar
        else:
            tensor = self.tensor_dict.get(name)
            if tensor is not None:
                return tensor
            else:
                print('No variable with name:', name, 'in the dictionary')
                
    def plot_scalar(self, name):
        """
        Plots the list of scalars stored in the scalar dictionary under the key name.

        Parameters
        ----------
        name: string
            The key by which to search the scalar dictionaries. 

        """

        lst = self.scalar_dict.get(name)
        plt.plot(lst)
        plt.show()
        
    def plot_tensor(self, name, idx_lst):
        """
        Plots the tensor stored in the idx_lst index of the list in the tensor dictionary corresponding
        to the key name.

        Parameters
        ---------
        name: string
            The key by which to search the tensor dictionaries.
        idx_list: list of int_
            The indices of the list of tensors to plot. 

        """
        tensor_lst = self.tensor_dict.get(name)
        for i in idx_lst:
            if i < len(tensor_lst):
                tensor = tensor_lst[i]
                #fig = plt.figure(figsize = (15,105))
                plt.imshow(tensor,aspect='auto',cmap='binary',interpolation='none')
                plt.show()
