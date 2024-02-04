import torch
import torch.nn as nn
from typing import List, Union

from .utils.initializations import initialize_tensor


class CMF(nn.Module):
    """Coupled matrix-tensor factorizations.
    
    Models the first-order (i.e., additive) interactions of input features, and 
    optionally the highest-order interactions with a CP decomposition.
    For N sets of input features, the highest-order interactions are then of the Nth order.
    
    The column space is shared, the row space is factorized.
    
    Parameters
    ----------
    full_tensor_dims: list
        List of dimensionalities, with the joint embedding dimensionality first
        followed by the dimensionalities of the input features
        
    rank: int
        Rank of overall factorization
        Applies to column space, row space and CP decomposition of highest-order interactions
    
    first_order_interactions_only: bool
        If true, only the first-order interactions are modelled
        
    share_row_space: bool
        If true, the row-space matrices of the highest-order interactions are the same
        as the first-order row space matrices
        
    Returns
    -------
    Z: torch.Tensor
        joint embedding

    References
    ----------
    .. [1] T. Kefalas, K. Vougioukas, Y. Panagakis, S. Petridis,
           J. Kossaifi and M. Pantic,
           "Speech-driven facial animation using polynomial fusion of features",
           ICASSP 2020, pp. 3487-3491
           
    .. [2] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications", SIAM
           REVIEW, vol. 51, n. 3, pp. 455-500, 2009.           
       
    .. [3] F. Hitchcock, "The expression of a tensor or a polyadic, as a sum of products",
           Journal of Mathematics and Physics, vol. 6, no. 1-4, pp. 164–189, 1927.
           
    .. [4] G. Chrysos, S.Moschoglou, Y. Panagakis and S. Zafeiriou,
           "PolyGAN: high-order polynomial generators" CoRR, 2019
           available at: http://arxiv.org/abs/1908.06571           
    """

    def __init__(self, full_tensor_dims: List[int], rank: Union[List[int], int], init: str = "xavier",
                 batch_first: bool = False, first_order_interactions_only: bool = False,
                 share_row_space: bool = False) -> None:
        super(CMF, self).__init__()
        
        self.full_tensor_dims = full_tensor_dims
        self.rank = rank
        self.n_modes = len(full_tensor_dims)
        self.batch_first = batch_first
        self.share_row_space = share_row_space               
        self.first_order_interactions_only = first_order_interactions_only
        
        # Bias vector
        b_dim = [self.full_tensor_dims[0]]
        self.b = nn.Parameter(initialize_tensor(b_dim, init = init))
        
        # Column space matrix
        U_dims = [self.full_tensor_dims[0], rank]
        self.U = nn.Parameter(initialize_tensor(U_dims, init = init))
        
        # First-order row space matrices
        self.V_matrices = nn.ParameterList([None] * (self.n_modes-1))
        for i in range(self.n_modes-1):
            V_dims = [self.full_tensor_dims[i+1], rank]
            self.V_matrices[i] = nn.Parameter(initialize_tensor(V_dims, init = init))
         
        # Highest-order row space matrices
        if not self.first_order_interactions_only:
            if share_row_space:
                self.U_matrices = self.V_matrices
            else:            
                self.U_matrices = nn.ParameterList([None] * (self.n_modes-1))            
                for i in range(self.n_modes-1):
                    U_dims = [self.full_tensor_dims[i+1], rank]
                    self.U_matrices[i] = nn.Parameter(initialize_tensor(U_dims, init = init))

    def forward(self, X_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Note: when computing n-mode products (ref. [2]), involving separate Khatri-Rao products of
        factor matrices and input X_matrices, we use the equivalent Hadamard product
        formulation (Lemmas 1-2, ref. [4])
        """
        
        if self.batch_first:
            X_matrices = [X.t() for X in X_matrices]
        
        ## Bias term
        num_datapoints = X_matrices[0].shape[1]
        bias = torch.stack(num_datapoints * [self.b], dim = 1)        
        
        ## First-order interactions
        VX = 0
        for i in range(len(X_matrices)):
            VX += self.V_matrices[i].t() @ X_matrices[i]
            
        ## Highest-order interactions            
        if self.first_order_interactions_only:
            Z = bias + self.U @ VX

            return Z.t() if self.batch_first else Z
            
        else:
            UX = 1
            for i in range(len(X_matrices)):
                UX = UX * (self.U_matrices[i].t() @ X_matrices[i])

            Z = bias + self.U @ (VX + UX)            
        
            return Z.t() if self.batch_first else Z