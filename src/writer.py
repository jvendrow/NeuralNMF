#!/usr/bin/env python
# coding: utf-8


# Date:2018.07.21
# Author: Runyu Zhang


from matplotlib import pyplot as plt


class Writer:

    """
    Records a history of scalar and Tensor information
    and plots the Tensors.

    """

    def __init__(self):
        """
        Initialized the scalar and tensor dictionaries.

        """

        self.scalar_dict = {}
        self.tensor_dict = {}
        
    def add_scalar(self, name, scalar):
        """
        Adds a value to the list of scalars stored 
        under the proper key in the scalar dictionary.

        Parameters
        ----------
        name: String
            The key value for access the scalar dictionary

        scalar: float
            The value to add to the list stored under
            scalar_dict[name].

        """
        if self.scalar_dict.get(name) is None:
            self.scalar_dict[name] = [scalar]
        else:
            self.scalar_dict[name].append(scalar)
            
    def add_tensor(self, name, tensor):
        """
        Adds a Pytorch Tensor to the list of tensors stored 
        under the proper key in the Tensor dictionary.

        Parameters
        ----------
        name: String
            The key value for access the tensor dictionary

        scalar: Pytorch Tensor
            The Tensor to add to the list stored under
            scalar_dict[name].

        """

        if self.tensor_dict.get(name) is None:
            self.tensor_dict[name] = [tensor]
        else:
            self.tensor_dict[name].append(tensor)
            
    def get(self, name):
        """
        Fetches the list of scalars or tensors stores
        under name. Searches the scalar list before
        the tensor list.

        Parameters
        ----------
        name: String
            the key by which to search the scalar
            and tensor dictionaries. 

        Returns
        -------
        values: list 
            The list of scalars or Tensors that corresponds
            to name, the key value. If no the key value is not
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
        Plots the list of scalars stores in the scalar
        dictionary under the key name.

        Parameters
        ----------
        name: String
            The key by which to search the scalar
            dictionaries. 

        """

        lst = self.scalar_dict.get(name)
        plt.plot(lst)
        plt.show()
        
    def plot_tensor(self, name, idx_lst):
        """
        Plots the tensor stored in the idx_lst index
        of the list in the tensor dictionary corresponding
        to the key name.

        Parameters
        ---------
        name: String
            The key by which to search the tensor 
            dictionaries.

        idx_list: integer
            The index of the list of Tensors for
            the Tensor to plot. 

        """
        tensor_lst = self.tensor_dict.get(name)
        for i in idx_lst:
            if i < len(tensor_lst):
                tensor = tensor_lst[i]
                fig = plt.figure(figsize = (15,105))
                plt.imshow(tensor)
                plt.show()
