import numpy as _numpy


def random_pd_matrix(dim: int):
    """Generate a random symmetric, positive-definite matrix.

    Parameters
    ----------
    dim : int
        The matrix dimension.

    Returns
    -------
    x : array of shape [n_dim, n_dim]
        The random symmetric, positive-definite matrix.

    """
    # Create random matrix
    a = _numpy.random.rand(dim, dim)
    # Create random PD matrix and extract correlation structure
    u, _, v = _numpy.linalg.svd(_numpy.dot(a.T, a))
    # Reconstruct a new matrix with random variances.
    return _numpy.dot(_numpy.dot(u, 1.0 + _numpy.diag(_numpy.random.rand(dim))), v)


def random_correlation_matrix(dim: int):
    """Generate a random symmetric, positive-definite matrix.

    Parameters
    ----------
    dim : int
        The matrix dimension.

    Returns
    -------
    x : array of shape [n_dim, n_dim]
        The random symmetric, positive-definite matrix.

    """
    cov = random_pd_matrix(dim)

    inv_sigma = _numpy.diag(1.0 / _numpy.diag(cov) ** 0.5)

    return inv_sigma @ cov @ inv_sigma
