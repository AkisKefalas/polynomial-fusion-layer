import torch
import torch.nn as nn
import tensorly as tl
from typing import List, Union

from .utils.initializations import initialize_tensor

tl.set_backend('pytorch')


class PartialTucker(nn.Module):
    """Tucker decomposition where the first factor matrix (U0) and the Tucker core G are merged into one tensor B.
    
    Parameters
    ----------
    full_tensor_dims: list
        List of dimensionalities, with the joint embedding dimensionality first
        followed by the dimensionalities of the input features
        
    rank: List[int] or int
        Rank of Tucker decomposition
        
    Returns
    -------
    Z: torch.Tensor
        Joint embedding
        
    
    References
    ----------
    .. [1] T. Kefalas, K. Vougioukas, Y. Panagakis, S. Petridis,
           J. Kossaifi and M. Pantic,
           "Speech-driven facial animation using polynomial fusion of features",
           ICASSP 2020, pp. 3487-3491
           
    .. [2] T.G. Kolda and B.W. Bader, "Tensor Decompositions and Applications", SIAM
           REVIEW, vol. 51, n. 3, pp. 455-500, 2009.           
       
    .. [3] L. Tucker, "Some mathematical notes on three-mode factor analysis", Psychometrika,
           vol. 31, no. 3, pp. 279–311, 1966.
    """    

    def __init__(self, full_tensor_dims: List[int], rank: Union[List[int], int],
                 init: str = "xavier", batch_first: bool = False) -> None:
        super(PartialTucker, self).__init__()

        self.full_tensor_dims = full_tensor_dims

        if isinstance(rank, int):
            self.rank = [rank] * len(full_tensor_dims) 
        else:
            self.rank = rank

        self.n_modes = len(full_tensor_dims)
        self.batch_first = batch_first
        
        # Core
        self.core_dims = [None] * self.n_modes
        self.core_dims[0] = self.full_tensor_dims[0]
        self.core_dims[1:self.n_modes] = self.rank[1:self.n_modes]

        self.B = nn.Parameter(initialize_tensor(self.core_dims, init = init))
        
        # Factor matrices
        self.U_matrices = nn.ParameterList([None] * (self.n_modes-1))
        for i in range(self.n_modes-1):
            U_dims = (self.full_tensor_dims[i+1], self.rank[i+1])
            self.U_matrices[i] = nn.Parameter(initialize_tensor(U_dims, init = init))            
        
    def forward(self, X_matrices: List[torch.Tensor]) -> torch.Tensor:
        
        if self.batch_first:
            X_matrices = [X.t() for X in X_matrices]

        UX = [None] * len(X_matrices)
        for i in range(len(X_matrices)):
            UX[i] = self.U_matrices[i].t() @ X_matrices[i]
   
        UX = tl.tenalg.khatri_rao(UX)
            
        B_1 = tl.unfold(self.B, mode = 0)        
        Z = B_1 @ UX

        return Z.t() if self.batch_first else Z