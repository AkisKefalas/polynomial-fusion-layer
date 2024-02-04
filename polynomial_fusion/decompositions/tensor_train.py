import torch
import torch.nn as nn
import tensorly as tl
from typing import List, Union

from .utils.initializations import initialize_tensor

tl.set_backend('pytorch')


class TensorTrain(nn.Module):
    """Tensor-Train/MPS decomposition.
    
    Parameters
    ----------
    full_tensor_dims: list
        List of dimensionalities, with the joint embedding dimensionality first
        followed by the dimensionalities of the input features
        
    rank: List[int] or int
        Rank of MPS decomposition
        
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
           
    .. [2] Ivan V. Oseledets. "Tensor-train decomposition", SIAM J. Scientific Computing,
           33(5):2295–2317, 2011.
    """

    def __init__(self, full_tensor_dims: List[int], rank: Union[List[int], int],
                 init: str = "xavier", batch_first: bool = False) -> None:
        super(TensorTrain, self).__init__()

        self.full_tensor_dims = full_tensor_dims
        self.n_modes = len(full_tensor_dims)
        if isinstance(rank, int):
            self.rank = [None] * (self.n_modes+1)
            self.rank[0] = 1
            self.rank[self.n_modes] = 1
            for i in range(self.n_modes-1):
                self.rank[i+1] = rank
            
        else:
            self.rank = rank
        self.batch_first = batch_first

        # MPS cores
        self.cores = nn.ParameterList([None] * self.n_modes)
        for i in range(self.n_modes):
            core_dims = [self.rank[i], self.full_tensor_dims[i], self.rank[i+1]]
            self.cores[i] = nn.Parameter(initialize_tensor(core_dims, init = init))

    def forward(self, X_matrices: List[torch.Tensor]) -> torch.Tensor:
        
        if self.batch_first:
            X_matrices = [X.t() for X in X_matrices]

        batch_size = X_matrices[0].shape[1]
        num_views = len(X_matrices)
        Z = torch.zeros([self.full_tensor_dims[0], batch_size]).to(X_matrices[0].device)

        for i in range(batch_size):
            z_list = [None] * num_views
            for j in range(num_views):
                x = X_matrices[j][:, i]
                z_list[j] = tl.tenalg.mode_dot(tl.transpose(self.cores[j+1], [0, 2, 1]), x, mode = 2)

            z = z_list[0]
            for j in range(num_views-1):
                z = tl.dot(z, z_list[j+1])

            Z[:, i] = tl.tenalg.mode_dot(self.cores[0], z[:, 0], 2)[0, :]
        
        return Z.t() if self.batch_first else Z
