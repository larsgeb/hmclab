"""
Prior distributions available to the HMC sampler.
"""
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod

import numpy as _numpy
from termcolor import cprint as _cprint


class _AbstractPrior(_ABC):

    name: str = ""
    dimensions: int = -1
    bounded: bool = False

    @_abstractmethod
    def misfit(self, coordinates: _numpy.ndarray) -> float:

        pass

    @_abstractmethod
    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:

        pass

    @_abstractmethod
    def generate(self) -> _numpy.ndarray:
        pass

    @_abstractmethod
    def corrector(self, coordinates: _numpy.ndarray, momentum: _numpy.ndarray):

        pass

    def bounds_corrector(
        self, coordinates: _numpy.ndarray, momentum: _numpy.ndarray
    ):

        if self.lower_bounds is not None:
            # Lower bound correction ---------------------------------------
            too_low = coordinates < self.lower_bounds
            coordinates[too_low] += 2 * (
                self.lower_bounds[too_low] - coordinates[too_low]
            )
            momentum[too_low] *= -1.0
        if self.upper_bounds is not None:
            # Lower bound correction ---------------------------------------
            too_high = coordinates > self.upper_bounds
            coordinates[too_high] += 2 * (
                self.upper_bounds[too_high] - coordinates[too_high]
            )
            momentum[too_high] *= -1.0


class Normal(_AbstractPrior):
    """Normal distribution in model space."""

    lower_bounds: _numpy.ndarray = None
    upper_bounds: _numpy.ndarray = None

    def __init__(
        self,
        dimensions: int,
        means: _numpy.ndarray = None,
        covariance: _numpy.ndarray = None,
        lower_bounds: _numpy.ndarray = None,
        upper_bounds: _numpy.ndarray = None,
    ):

        self.name = "Gaussian prior"
        self.dimensions = dimensions
        self.diagonal: bool = False  # whether or not Gaussian is uncorrelated

        if means is None and covariance is None:
            # Neither means nor covariance is provided -------------------------
            _cprint(
                "Neither means or covariance matrix provided. Generating "
                "random means and variances.",
                "yellow",
            )
            self.means = _numpy.random.rand(dimensions, 1)
            self.covariance = _make_spd_matrix(self.dimensions)

        elif means is None or covariance is None:
            # Only one of means or covariance is provided ----------------------
            raise ValueError(
                "No means or covariance matrix provided. Not sure what to do!"
            )
        else:
            # Both means and covariance are provided ---------------------------

            # Parse means
            if means.shape != (self.dimensions, 1):
                raise ValueError("Incorrect size of means vector.")
            self.means: _numpy.ndarray = means

            # Parse covariance
            if covariance.shape == (means.size, means.size):
                self.diagonal = False
            elif covariance.shape == (means.size, 1):
                self.diagonal = True
                _cprint(
                    "Seem that you only passed a vector as the covariance "
                    "matrix. It will be used as the covariance diagonal.",
                    "yellow",
                )
            else:
                raise ValueError("Incorrect size of covariance matrix.")
            self.covariance = covariance
        if self.diagonal:
            self.inverse_covariance = 1.0 / self.covariance
        else:
            self.inverse_covariance = _numpy.linalg.inv(self.covariance)

        # Process optional bounds ----------------------------------------------
        if lower_bounds is not None and lower_bounds.shape == (
            self.dimensions,
            1,
        ):
            self.lower_bounds = lower_bounds
            self.bounded = True
        elif lower_bounds is not None:
            raise ValueError("Incorrect size of lower bounds vector.")

        if upper_bounds is not None and upper_bounds.shape == (
            self.dimensions,
            1,
        ):
            self.upper_bounds = upper_bounds
            self.bounded = True
        elif upper_bounds is not None:
            raise ValueError("Incorrect size of upper bounds vector.")

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        if self.bounded:
            if self.lower_bounds is not None and not _numpy.all(
                coordinates > self.lower_bounds
            ):
                return _numpy.inf
            if self.upper_bounds is not None and not _numpy.all(
                coordinates < self.upper_bounds
            ):
                return _numpy.inf

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

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:

        if self.diagonal:
            return -self.inverse_covariance * (self.means - coordinates)
        else:
            return -self.inverse_covariance @ (self.means - coordinates)

    def generate(self) -> _numpy.ndarray:
        """

        Returns
        -------

        """
        raise NotImplementedError("This function is not finished yet")

    def corrector(self, coordinates, momentum):
        """

        Parameters
        ----------
        coordinates
        momentum

        Returns
        -------

        """
        self.bounds_corrector(coordinates, momentum)


class LogNormal(_AbstractPrior):
    """Normal distribution in logarithmic model space."""

    def __init__(
        self,
        dimensions: int,
        means: _numpy.ndarray = None,
        covariance: _numpy.ndarray = None,
        lower_bounds: _numpy.ndarray = None,
        upper_bounds: _numpy.ndarray = None,
    ):
        """

        Parameters
        ----------
        dimensions
        means
        covariance
        lower_bounds
        upper_bounds
        """
        self.name = "log normal (logarithmic Gaussian) prior"
        self.dimensions = dimensions
        self.diagonal: bool = False  # whether or not Gaussian is uncorrelated

        if means is None and covariance is None:
            # Neither means nor covariance is provided
            _cprint(
                "Neither means or covariance matrix provided. Generating "
                "random means and variances.",
                "yellow",
            )
            self.means = _numpy.random.rand(dimensions, 1)
            self.covariance = _make_spd_matrix(self.dimensions)

        elif means is None or covariance is None:
            # Only one of means or covariance is provided
            raise ValueError(
                "No means or covariance matrix provided. Not sure what to do!"
            )
        else:
            # Both means and covariance are provided

            # Parse means
            if means.shape != (self.dimensions, 1):
                raise ValueError("Incorrect size of means vector.")
            self.means: _numpy.ndarray = means

            # Parse covariance
            if covariance.shape == (means.size, means.size):
                self.diagonal = False
            elif covariance.shape == (means.size, 1):
                self.diagonal = True
                _cprint(
                    "Seem that you only passed a vector as the covariance "
                    "matrix. It will be used as the covariance diagonal.",
                    "yellow",
                )
            else:
                raise ValueError("Incorrect size of covariance matrix.")
            self.covariance: _numpy.ndarray = covariance
        if self.diagonal:
            self.inverse_covariance = 1.0 / self.covariance
        else:
            self.inverse_covariance = _numpy.linalg.inv(self.covariance)

        # Process optional bounds ----------------------------------------------
        if lower_bounds is not None and lower_bounds.shape == (
            self.dimensions,
            1,
        ):
            self.lower_bounds = lower_bounds
            self.bounded = True
        elif lower_bounds is not None:
            raise ValueError("Incorrect size of lower bounds vector.")

        if upper_bounds is not None and upper_bounds.shape == (
            self.dimensions,
            1,
        ):
            self.upper_bounds = upper_bounds
            self.bounded = True
        elif upper_bounds is not None:
            raise ValueError("Incorrect size of upper bounds vector.")

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        logarithmic_coordinates = _numpy.log(coordinates)
        if self.diagonal:
            return _numpy.sum(logarithmic_coordinates).item(0) + 0.5 * (
                (self.means - logarithmic_coordinates).T
                @ (
                    self.inverse_covariance
                    * (self.means - logarithmic_coordinates)
                )
            ).item(0)
        else:
            return _numpy.sum(logarithmic_coordinates).item(0) + 0.5 * (
                (self.means - logarithmic_coordinates).T
                @ self.inverse_covariance
                @ (self.means - logarithmic_coordinates)
            ).item(0)

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        # ! Not sure about these formulas!
        logarithmic_coordinates = _numpy.log(coordinates)
        if self.diagonal:
            return (
                -self.inverse_covariance
                * (self.means - logarithmic_coordinates)
                / coordinates
            ) + _numpy.sum(1.0 / coordinates)
        else:
            return (
                -self.inverse_covariance
                @ (self.means - logarithmic_coordinates)
                / coordinates
            ) + _numpy.sum(1.0 / coordinates)

    def generate(self) -> _numpy.ndarray:
        raise NotImplementedError("This function is not finished yet")

    def corrector(self, coordinates, momentum):
        self.bounds_corrector(coordinates, momentum)


class UnboundedUniform(_AbstractPrior):
    def __init__(self, dimensions: int):
        """

        Parameters
        ----------
        dimensions
        """
        self.name = "unbounded uniform prior"
        self.dimensions = dimensions

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        """

        Parameters
        ----------
        coordinates

        Returns
        -------

        """
        return 0.0

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        """

        Parameters
        ----------
        coordinates

        Returns
        -------

        """
        return _numpy.zeros((self.dimensions, 1))

    def generate(self) -> _numpy.ndarray:  # One shouldn't be able to do this
        raise TypeError(
            "This prior is unbounded, so it is impossible to generate samples"
            "from it."
        )

    def corrector(self, coordinates: _numpy.ndarray, momentum: _numpy.ndarray):
        """
        Args:
            coordinates (_numpy.ndarray):
            momentum (_numpy.ndarray):
        """
        pass


class Uniform(_AbstractPrior):
    def __init__(
        self,
        dimensions: int,
        lower_bounds: _numpy.ndarray = None,
        upper_bounds: _numpy.ndarray = None,
    ):
        """
        Args:
            dimensions (int):
            lower_bounds (_numpy.ndarray):
            upper_bounds (_numpy.ndarray):
        """
        self.name = "uniform prior"
        self.dimensions = dimensions
        if upper_bounds is None:
            self.upper_bounds = _numpy.ones((dimensions, 1))
        else:
            self.upper_bounds = upper_bounds or _numpy.ones((dimensions, 1))
        if lower_bounds is None:
            self.lower_bounds = _numpy.zeros((dimensions, 1))
        else:
            self.lower_bounds = lower_bounds
        if (
            not self.lower_bounds.shape
            == self.upper_bounds.shape
            == (dimensions, 1)
        ):
            raise ValueError("Bounds vectors are of incorrect size.")
        self.widths = self.upper_bounds - self.lower_bounds
        if not _numpy.all(self.widths > 0.0):
            raise ValueError("Some upper bounds are below lower bounds.")
        self._misfit = -_numpy.sum(_numpy.log(self.widths)).item()

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        """

        Parameters
        ----------
        coordinates

        Returns
        -------

        """
        return self._misfit

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        """

        Parameters
        ----------
        coordinates

        Returns
        -------

        """
        return _numpy.zeros((self.dimensions, 1))

    def generate(self) -> _numpy.ndarray:  # One shouldn't be able to do this
        """

        """
        return self.lower_bounds + self.widths * _numpy.random.rand(
            self.dimensions, 1
        )

    def corrector(self, coordinates, momentum):
        self.bounds_corrector(coordinates, momentum)


def _make_spd_matrix(dim):
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
    return _numpy.dot(
        _numpy.dot(u, 1.0 + _numpy.diag(_numpy.random.rand(dim))), v
    )
