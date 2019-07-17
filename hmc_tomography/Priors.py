"""
Prior distributions available to the HMC sampler.
"""
from abc import ABC, abstractmethod

import numpy


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

    def __init__(self, means: numpy.ndarray, covariance: numpy.ndarray):
        """

        Parameters
        ----------
        means
        covariance
        """
        self.name = "Gaussian prior"
        self.means: numpy.ndarray = means
        self.dimensions = means.size
        self.covariance = covariance

    def misfit(self, coordinates: numpy.ndarray) -> float:
        """

        Parameters
        ----------
        coordinates

        Returns
        -------

        """
        return (
            0.5
            * (self.means - coordinates).T
            @ (self.covariance @ (self.means - coordinates)).item(0)
        )

    def gradient(self, coordinates: numpy.ndarray) -> numpy.ndarray:
        """

        Parameters
        ----------
        coordinates

        Returns
        -------

        """
        return self.covariance @ (self.means - coordinates)

    def generate(self) -> numpy.ndarray:
        """

        """
        raise NotImplementedError("This function is not finished yet")


class LogNormal(Prior):
    """Normal distribution in logarithmic model space.

    """

    def __init__(self, means: numpy.ndarray, covariance: numpy.ndarray):
        """

        Parameters
        ----------
        means
        covariance
        """
        self.name = "log normal (logarithmic Gaussian) prior"
        self.means: numpy.ndarray = means  # in log space
        self.dimensions = means.size
        self.covariance = covariance  # in log space

    def misfit(self, coordinates: numpy.ndarray) -> float:
        raise NotImplementedError("This function is not finished yet")

    def gradient(self, coordinates: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError("This function is not finished yet")

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
        return 0

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
        TypeError(
            "This prior is unbounded, so it is impossible to generate samples"
            "from it."
        )
