import torch
import torch.nn as nn
from typing import List, Union

from .utils.initializations import initialize_tensor


class CP(nn.Module):
    """Canonical polyadic decomposition.
    
    Parameters
    ----------
    full_tensor_dims: list
        List of dimensionalities, with the joint embedding dimensionality first
        followed by the dimensionalities of the input features
        
    rank: int
        Rank of CP decomposition
        
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
           
    .. [2] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications", SIAM
           REVIEW, vol. 51, n. 3, pp. 455-500, 2009.           
       
    .. [3] F. Hitchcock, "The expression of a tensor or a polyadic, as a sum of products",
           Journal of Mathematics and Physics, vol. 6, no. 1-4, pp. 164–189, 1927.

    .. [4] G. Chrysos, S.Moschoglou, Y. Panagakis and S. Zafeiriou,
           "PolyGAN: high-order polynomial generators" CoRR, 2019
           available at: http://arxiv.org/abs/1908.06571
    """    

    def __init__(self, full_tensor_dims: List[int], rank: Union[List[int], int],
                 init: str = "xavier", batch_first: bool = False) -> None:
        super(CP, self).__init__()

        self.full_tensor_dims = full_tensor_dims
        self.rank = rank
        self.n_modes = len(full_tensor_dims)
        self.batch_first = batch_first
        
        # CP factor matrices
        self.U_matrices = nn.ParameterList([None] * self.n_modes)
        for i in range(self.n_modes):

            U_dims = [self.full_tensor_dims[i], self.rank]
            self.U_matrices[i] = nn.Parameter(initialize_tensor(U_dims, init = init))
    
    def forward(self, X_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Note: when computing n-mode products (ref. [2]), involving separate Khatri-Rao products of
        factor matrices and input X_matrices, we use the equivalent Hadamard product
        formulation (Lemmas 1-2, ref. [4])
        """
        
        if self.batch_first:
            X_matrices = [X.t() for X in X_matrices]        
        
        UX = 1
        for i in range(len(X_matrices)):
            UX = UX * (self.U_matrices[i+1].t() @ X_matrices[i])
        
        Z = self.U_matrices[0] @ UX
        
        return Z.t() if self.batch_first else Z