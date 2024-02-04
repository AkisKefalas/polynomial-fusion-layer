import torch
import torch.nn as nn
import tensorly as tl
from typing import List, Union

from .utils.initializations import initialize_tensor

tl.set_backend('pytorch')


class FullTucker(nn.Module):
    """Tucker decomposition.
    
    
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
        super(FullTucker, self).__init__()
        
        self.full_tensor_dims = full_tensor_dims
        if isinstance(rank, int):
            self.rank = [rank] * len(full_tensor_dims)
    
        else:
            self.rank = rank

        self.n_modes = len(full_tensor_dims)
        self.batch_first = batch_first

        # Core
        self.G = nn.Parameter(initialize_tensor(self.rank, init = init)) 
        
        # Factor matrices
        self.U_matrices = nn.ParameterList([None] * self.n_modes)
        for i in range(self.n_modes):

            U_dims = [self.full_tensor_dims[i], self.rank[i]]
            self.U_matrices[i] = nn.Parameter(initialize_tensor(U_dims, init = init))

    def forward(self, X_matrices: List[torch.Tensor]) -> torch.Tensor:
        
        if self.batch_first:
            X_matrices = [X.t() for X in X_matrices]

        UX = [None] * len(X_matrices)
        for i in range(len(X_matrices)):
            UX[i] = self.U_matrices[i+1].t() @ X_matrices[i]
          
        UX = tl.tenalg.khatri_rao(UX)
        
        G_1 = tl.unfold(self.G, mode = 0)        
        Z = self.U_matrices[0] @ (G_1 @ UX)

        return Z.t() if self.batch_first else Z