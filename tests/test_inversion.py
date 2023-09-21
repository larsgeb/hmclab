from hmclab import Inversion
from hmclab.Distributions import MultivariateNormal

import numpy as np, dill, pytest


def test_creation():
    mean_vector = [0, 0]
    covariance_matrix = [[1, 0.5], [0.5, 2]]
    multivariate_norm = MultivariateNormal(mean_vector, covariance_matrix)

    # Create an inversion instance with the MultivariateNormal distribution
    inversion = Inversion(target_distribution=multivariate_norm)
