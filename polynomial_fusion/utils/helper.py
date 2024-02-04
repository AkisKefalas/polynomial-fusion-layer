import torch


def pad_multiview_constant(X: torch.Tensor, batch_first: bool = False) -> torch.Tensor:
    """Pads a constant 1 to the input features for multi-view processing

    Parameters
    ----------
    X : torch.Tensor
        input matrix
    batch first : bool
        If true, the first mode (i.e., row mode) of the matrix corresponds to sample indices

    Returns
    -------
    torch.Tensor
        input matrix with padded constant 1 

    References
    ----------
    .. [1] B. Cao, H. Zhou, G. Li, and P. Yu, “Multi-view machines,” in Proceedings of the Ninth
           ACM International Conference on Web Search and Data Mining, New York, NY, USA, 2016,
           WSDM ’16, pp. 427–436, ACM.
    """
    
    if batch_first:
        return torch.cat([X, torch.ones([X.shape[0], 1], device = X.device)], dim = 1)
    else:
        return torch.cat([X, torch.ones([1, X.shape[1]], device = X.device)])

def get_input_matrix_for_multiview_fusion(X: torch.Tensor, add_multiview_constant: str = False,
                                         ) -> torch.Tensor:
    """Reshapes input tensor X into a matrix for multi-view processing

    Parameters
    ----------
    X : torch.Tensor
        input of shape (batch_size, dimensionality) or (batch_size, num_timesteps, dimensionality)
    add_multiview_constant: bool
        If true, pads the dimensionality mode with constant 1      

    Returns
    -------
    torch.Tensor
        output of shape (dimensionality, batch_size) or (dimensionality, batch_size * num_timesteps)
    """
    
    assert len(X.shape) in [2, 3]
    
    if len(X.shape) == 2:
        batch_size, dimensionality = X.shape
        
    else:
        batch_size, num_timesteps, dimensionality = X.shape
        batch_size = batch_size * num_timesteps
    
    X = X.reshape([batch_size, dimensionality]).t()
    
    if add_multiview_constant:
        X = pad_multiview_constant(X)
        
    return X