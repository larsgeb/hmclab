"""
Prior distributions available to the HMC sampler.
"""
from abc import ABC, abstractmethod

import numpy
from termcolor import cprint


class Prior(ABC):

    name: str = ""
    dimensions: int = -1

    @abstractmethod
    def misfit(self, coordinates: numpy.ndarray) -> float:
        """

        Parameters
        ----------
        coordinates
        """
        pass

    @abstractmethod
    def gradient(self, coordinates: numpy.ndarray) -> numpy.ndarray:
        """

        Parameters
        ----------
        coordinates
        """
        pass

    @abstractmethod
    def generate(self) -> numpy.ndarray:
        """

        """
        pass


class Normal(Prior):
    """Normal distribution in model space.

    """

    def __init__(
        self,
        dimensions: int,
        means: numpy.ndarray = None,
        covariance: numpy.ndarray = None,
        inverse_covariance: numpy.ndarray = None,
    ):
        """

        Parameters
        ----------
        means
        covariance
        """
        self.name = "Gaussian prior"
        self.dimensions = dimensions
        self.diagonal: bool = False  # whether or not Gaussian is uncorrelated

        if means is None and covariance is None:
            # Neither means nor covariance is provided
            cprint(
                "Neither means or covariance matrix provided. Generating "
                "random means and variances.",
                "yellow",
            )
            self.means = numpy.random.rand(dimensions, 1)
            self.covariance = make_spd_matrix(self.dimensions)

        elif means is None or covariance is None:
            # Only one of means or covariance is provided
            raise ValueError(
                "No means or covariance matrix provided. Not sure what to do!"
            )
        else:
            # Both means and covariance are provided

            # Parse means
            if means.shape != (self.dimensions, 1):
                ValueError("Incorrect size of means vector.")
            self.means: numpy.ndarray = means

            # Parse covariance
            if covariance.shape == (means.size, means.size):
                self.diagonal = False
            elif covariance.shape == (means.size, 1):
                self.diagonal = True
                cprint(
                    "Seem that you only passed a vector as the covariance "
                    "matrix. It will be used as the covariance diagonal.",
                    "yellow",
                )
            else:
                ValueError("Incorrect size of covariance matrix.")
            self.covariance = covariance
        if self.diagonal:
            self.inverse_covariance = 1.0 / covariance
        else:
            self.inverse_covariance = numpy.linalg.inv(covariance)

    def misfit(self, coordinates: numpy.ndarray) -> float:
        """

        Parameters
        ----------
        coordinates

        Returns
        -------

        """
        if self.diagonal:
            return 0.5 * (
                (self.means - coordinates).T
                @ (self.inverse_covariance * (self.means - coordinates))
            ).item(0)
        else:
            return 0.5 * (
                (self.means - coordinates).T
                @ self.inverse_covariance
                @ (self.means - coordinates)
            ).item(0)

    def gradient(self, coordinates: numpy.ndarray) -> numpy.ndarray:
        """

        Parameters
        ----------
        coordinates

        Returns
        -------

        """
        if self.diagonal:
            return -self.inverse_covariance * (self.means - coordinates)
        else:
            return -self.inverse_covariance @ (self.means - coordinates)

    def generate(self) -> numpy.ndarray:
        """

        """
        raise NotImplementedError("This function is not finished yet")


class LogNormal(Prior):
    """Normal distribution in logarithmic model space.

    """

    def __init__(
        self,
        dimensions: int,
        means: numpy.ndarray = None,
        covariance: numpy.ndarray = None,
    ):
        """

        Parameters
        ----------
        means
        covariance
        """
        self.name = "log normal (logarithmic Gaussian) prior"
        self.dimensions = dimensions
        self.diagonal: bool = False  # whether or not Gaussian is uncorrelated

        if means is None and covariance is None:
            # Neither means nor covariance is provided
            cprint(
                "Neither means or covariance matrix provided. Generating "
                "random means and variances.",
                "yellow",
            )
            self.means = numpy.random.rand(dimensions, 1)
            self.covariance = make_spd_matrix(self.dimensions)

        elif means is None or covariance is None:
            # Only one of means or covariance is provided
            raise ValueError(
                "No means or covariance matrix provided. Not sure what to do!"
            )
        else:
            # Both means and covariance are provided

            # Parse means
            if means.shape != (self.dimensions, 1):
                ValueError("Incorrect size of means vector.")
            self.means: numpy.ndarray = means

            # Parse covariance
            if covariance.shape == (means.size, means.size):
                self.diagonal = False
            elif covariance.shape == (means.size, 1):
                self.diagonal = True
                cprint(
                    "Seem that you only passed a vector as the covariance "
                    "matrix. It will be used as the covariance diagonal.",
                    "yellow",
                )
            else:
                ValueError("Incorrect size of covariance matrix.")
            self.covariance: numpy.ndarray = covariance
        if self.diagonal:
            self.inverse_covariance = 1.0 / covariance
        else:
            self.inverse_covariance = numpy.linalg.inv(covariance)

    def misfit(self, coordinates: numpy.ndarray) -> float:
        logarithmic_coordinates = numpy.log(coordinates)
        if self.diagonal:
            return numpy.sum(logarithmic_coordinates).item(0) + 0.5 * (
                (self.means - logarithmic_coordinates).T
                @ (
                    self.inverse_covariance
                    * (self.means - logarithmic_coordinates)
                )
            ).item(0)
        else:
            return numpy.sum(logarithmic_coordinates).item(0) + 0.5 * (
                (self.means - logarithmic_coordinates).T
                @ self.inverse_covariance
                @ (self.means - logarithmic_coordinates)
            ).item(0)

    def gradient(self, coordinates: numpy.ndarray) -> numpy.ndarray:
        # ! Not sure about these formulas!
        logarithmic_coordinates = numpy.log(coordinates)
        if self.diagonal:
            return (
                -self.inverse_covariance
                * (self.means - logarithmic_coordinates)
                / coordinates
            ) + numpy.sum(1.0 / coordinates)
        else:
            return (
                -self.inverse_covariance
                @ (self.means - logarithmic_coordinates)
                / coordinates
            ) + numpy.sum(1.0 / coordinates)

    def generate(self) -> numpy.ndarray:
        raise NotImplementedError("This function is not finished yet")


class UnboundedUniform(Prior):
    def __init__(self, dimensions: int):
        """

        Parameters
        ----------
        dimensions
        """
        self.name = "unbounded uniform prior"
        self.dimensions = dimensions

    def misfit(self, coordinates: numpy.ndarray) -> float:
        """

        Parameters
        ----------
        coordinates

        Returns
        -------

        """
        return 0.0

    def gradient(self, coordinates: numpy.ndarray) -> numpy.ndarray:
        """

        Parameters
        ----------
        coordinates

        Returns
        -------

        """
        return numpy.zeros((self.dimensions, 1))

    # noinspection PyTypeChecker
    def generate(self) -> numpy.ndarray:  # One shouldn't be able to do this
        """

        """
        raise TypeError(
            "This prior is unbounded, so it is impossible to generate samples"
            "from it."
        )


def make_spd_matrix(dim):
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
    a = numpy.random.rand(dim, dim)
    # Create random PD matrix and extract correlation structure
    u, _, v = numpy.linalg.svd(numpy.dot(a.T, a))
    # Reconstruct a new matrix with random variances.
    return numpy.dot(numpy.dot(u, 1.0 + numpy.diag(numpy.random.rand(dim))), v)
