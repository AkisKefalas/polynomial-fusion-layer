import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Union

from .utils import helper
from .decompositions.full_tucker import FullTucker
from .decompositions.partial_tucker import PartialTucker
from .decompositions.cp import CP
from .decompositions.cmf import CMF
from .decompositions.tensor_train import TensorTrain


class PolynomialFusion(nn.Module):
    """Polynomial fusion layer with a matrix/tensor decomposition on the parameters (weights)

    Parameters
    ----------
    model_type : str
        Specify type of matrix/tensor decomposition of the parameters
    concat_last_mode : bool
        If true, the input corresponding to the last mode is concatenated to the final embedding
    input_dims : list
        Dimensionalities of each input feature space
    output_dim : int
        If None, the output_dim is set to the sum of the dimensionalities of the inputs
    rank : list or int
        Rank of matrix/tensor decomposition
    multiview: bool
        If true, models interactions of all-orders
    init : str
        Initialization of parameters

    Returns
    -------
    Z: torch.Tensor
        Joint embedding

    References
    ----------
    .. [1] T. Kefalas, K. Vougioukas, Y. Panagakis, S. Petridis, J. Kossaifi and M. Pantic,
           "Speech-driven facial animation using polynomial fusion of features",
           ICASSP 2020, pp. 3487-3491
    """

    def __init__(self, model_type: str, input_dims: List[int], output_dim: Optional[int],
                 rank: Union[List[int], int], concat_last_mode: bool = False,
                 multiview: bool = False, init: str = "xavier") -> None:
        super(PolynomialFusion, self).__init__()        

        self.model_type = model_type
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.rank = rank
        self.concat_last_mode = concat_last_mode
        self.multiview = multiview
        self.init = init        
        
        if self.concat_last_mode:
            err_msg = "Dimensionality of joint embedding must be greater than the "
            err_msg += "dimensionality of the concatenated embedding"
            assert self.output_dim > self.input_dims[-1], err_msg
            if isinstance(self.rank, list):
                err_msg = "Need to specify one rank per view (except the last "
                err_msg += "one which is concatenated)"
                assert len(self.rank) == len(self.input_dims), err_msg
        else:
            if isinstance(self.rank, list):
                assert len(self.rank) == len(self.input_dims)+1                              
        
        # Get model
        self.model = self.get_model()
    
    def get_model(self) -> Optional[nn.Module]:
        
        if self.model_type == "concat":
            return None        
        
        # Dimensionality of joint embedding
        if not self.output_dim:
            self.output_dim = np.sum(self.input_dims)

        # Construct tensor of dimensions
        self.full_tensor_dims = [self.output_dim] + list(self.input_dims)
        if self.concat_last_mode:
            first_dim = self.output_dim - self.input_dims[-1]
            self.full_tensor_dims = [first_dim] + list(self.input_dims)[:-1]
        
        n_modes = len(self.full_tensor_dims)
        if self.multiview:
            for i in range(1, n_modes):
                self.full_tensor_dims[i] += 1

        # Get model                 
        if self.model_type == "FullTucker":
            return FullTucker(self.full_tensor_dims, self.rank, self.init)

        elif self.model_type == "PartialTucker":
            return PartialTucker(self.full_tensor_dims, self.rank, self.init)            

        elif self.model_type == "CP":
            return CP(self.full_tensor_dims, self.rank, self.init)

        elif self.model_type == "TensorTrain":
            return TensorTrain(self.full_tensor_dims, self.rank, self.init)

        elif self.model_type == "CMF":
            return CMF(self.full_tensor_dims, self.rank, self.init)

        elif self.model_type == "CMF_with_shared_row_space":
            return CMF(self.full_tensor_dims, self.rank, self.init, share_row_space = True)

        elif self.model_type == "CMF_first_order":
            return CMF(self.full_tensor_dims, self.rank, self.init, first_order_interactions_only = True)
        
        else:
            raise ValueError
    
    def forward(self, X_list: List[torch.Tensor]) -> torch.Tensor:

        if self.model_type == "concat":
            return torch.cat(X_list, dim = 1)
        
        else:            
            # Prepare inputs for fusion
            num_views = len(X_list)
            Z = [None] * num_views
            if self.concat_last_mode:
                num_views = num_views - 1
                Z[-1] = helper.get_input_matrix_for_multiview_fusion(X_list[-1],
                                                                     add_multiview_constant = False)

            for i in range(num_views):
                Z[i] = helper.get_input_matrix_for_multiview_fusion(X_list[i],
                                                                    add_multiview_constant = self.multiview)                

            # Fusion of inputs
            if self.concat_last_mode:
                Z = torch.cat([self.model(Z[:-1]), Z[-1]])

            else:
                Z = self.model(Z)

            return Z.t()
