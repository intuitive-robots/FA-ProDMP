import torch

def cholesky(matrix: torch.Tensor,
              init_jitter: float =  1e-12,
              max_iterations: int = 10
              )-> torch.Tensor:
    """calculates the Cholesky decomposition of a matrix.
    Positive-definiteness is enforced by adding jitter to the diagonal.

    Args:
        matrix (torch.Tensor): matrix to be decomposed
        init_jitter (float, optional): jitter to be added in the first iteration. Defaults to 1e-12.
        max_iterations (int, optional): max number of iterations trying to achieve positive-definiteness. Defaults to 10.

    Returns:
        torch.Tensor: the lower triangular Cholesky decomposition
    """
    jitter = init_jitter
    for _ in range(max_iterations):
        try:
            cholesky = torch.linalg.cholesky(matrix)
            return cholesky
        except:
            matrix += jitter * torch.eye(matrix.shape[0])
            jitter *= 10
    raise ValueError('cholesky could not be calculated')