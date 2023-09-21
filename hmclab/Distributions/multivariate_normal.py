import numpy as np

from hmclab.base import _parse_vector_input

from .base import AbstractTargetDistribution


class MultivariateNormal(AbstractTargetDistribution):
    name: str = "Multivariate Normal Distribution"

    def __init__(self, mean, covariance_matrix):
        mean = _parse_vector_input(mean)
        self.dimensionality = mean.size
        self.mean = mean
        self.covariance_matrix = np.array(covariance_matrix)
        self.precision_matrix = np.linalg.inv(covariance_matrix)

    def log_prob(self, x):
        x = _parse_vector_input(x, size=self.dimensionality)
        diff = x - self.mean
        log_prob = (
            -0.5 * np.dot(np.dot(diff.T, self.precision_matrix), diff)
            - 0.5 * np.log(np.linalg.det(self.covariance_matrix))
            - 0.5 * len(x) * np.log(2 * np.pi)
        )
        return -log_prob

    def log_prob_grad(self, x):
        x = _parse_vector_input(x, size=self.dimensionality)
        diff = x - self.mean
        grad = -np.dot(self.precision_matrix, diff)
        return -grad

    @classmethod
    def create_default(cls, dimensionality):
        # Create a default instance with zero mean and identity covariance
        # matrix
        mean = np.zeros(dimensionality)
        covariance_matrix = np.identity(dimensionality)
        return cls(mean, covariance_matrix)
