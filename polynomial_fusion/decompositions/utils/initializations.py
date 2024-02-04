import numpy as np
import torch
from typing import List, Optional, Union


def initialize_tensor(dimensions: List[int], init: str = "xavier") -> torch.Tensor:
    """Initialize tensor given an initialization method
    
    Parameters
    ----------
    dimensions: List[int]
        shape of the tensor
    init: str
        initialization method

    Returns
    ----------
    torch.Tensor
        initialized tensor
    """
    
    if init == "xavier":        
        return xavier_initialization(dimensions)
    
    if init == "zero":        
        return zero_initialization(dimensions)
    
    if init == "random_tensor_from_unit_rows":        
        return random_tensor_from_unit_rows(dimensions)    
    
    if init == "random_tensor_from_unit_columns":        
        return random_tensor_from_unit_columns(dimensions)
    
    if init == "random_tensor_from_unit_norm":
        return random_tensor_from_unit_norm(dimensions)    
    
def xavier_initialization(dimensions: List[int]) -> torch.Tensor:
    """Initialize tensor with Xavier normal distribution
    
    Parameters
    ----------
    dimensions: List[int]
        shape of the tensor

    Returns
    ----------
    torch.Tensor
        initialized tensor
    """
    
    if len(dimensions) == 1:
        X = torch.empty([dimensions[0], 1])
        torch.nn.init.xavier_normal_(X)
        
        return X[:, 0]

    else:
        X = torch.empty(dimensions)
        torch.nn.init.xavier_normal_(X)
        
        return X

def zero_initialization(dimensions: List[int]) -> torch.Tensor:
    """Initialize tensor with zeros
    
    Parameters
    ----------
    dimensions: List[int]
        shape of the tensor

    Returns
    ----------
    torch.Tensor
        initialized tensor
    """
    
    X = torch.zeros(dimensions)
    
    return X

def random_tensor_from_unit_rows(dimensions: List[int]) -> torch.Tensor:
    """Initialize tensor such that the mode-1 unfolding matrix
    has rows of expected squared l2 norm of 1
    
    Parameters
    ----------
    dimensions: List[int]
        shape of the tensor

    Returns
    ----------
    torch.Tensor
        initialized tensor
    """

    inverse_variance = np.prod(dimensions)/dimensions[0]
    stdev = 1/(inverse_variance**0.5)

    X = torch.empty(dimensions)
    X.normal_(mean = 0, std = stdev)

    return X

def random_tensor_from_unit_columns(dimensions: List[int]) -> torch.Tensor:
    """Initialize tensor such that the mode-1 unfolding matrix
    has columns of expected squared l2 norm of 1
    
    Parameters
    ----------
    dimensions: List[int]
        shape of the tensor

    Returns
    ----------
    torch.Tensor
        initialized tensor
    """

    inverse_variance = dimensions[0]
    stdev = 1/(inverse_variance**0.5)

    X = torch.empty(dimensions)
    X.normal_(mean = 0, std = stdev)

    return X

def random_tensor_from_unit_norm(dimensions: List[int]) -> torch.Tensor:
    """Initialize tensor such that it has expected squared l2 norm of 1
    
    Parameters
    ----------
    dimensions: List[int]
        shape of the tensor

    Returns
    ----------
    torch.Tensor
        initialized tensor
    """

    inverse_variance = np.prod(dimensions)
    stdev = 1/(inverse_variance**0.5)

    X = torch.empty(dimensions)
    X.normal_(mean = 0, std = stdev)

    return X