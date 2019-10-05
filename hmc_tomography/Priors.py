from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
from typing import List as _List

import numpy as _numpy
from termcolor import cprint as _cprint


class _AbstractPrior(_ABC):
    """Prior abstract base class

    """

    name: str = "abstract prior"
    """Name of the prior."""

    dimensions: int = -1
    """Model space dimension of the prior."""

    bounded: bool = False  # TODO phase out self.bounded
    """Boundedness of the prior. Allowed to be altered in the constructor."""

    lower_bounds: _numpy.ndarray = None
    """Lower bounds for every parameter. If initialized to None, no bounds are used."""

    upper_bounds: _numpy.ndarray = None
    """Upper bounds for every parameter. If initialized to None, no bounds are used."""

    @_abstractmethod
    def misfit(self, coordinates: _numpy.ndarray) -> float:
        """Computes the misfit of the prior at the given coordinates.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Numpy array shaped as (dimensions, 1) representing a column vector
            containing the coordinates :math:`\\mathbf{m}`.

        Returns
        -------
        misfit : float
            The prior misfit :math:`\\chi`.


        The prior misfit is related to the prior probability density as:

        .. math::

            \\chi_\\text{prior} (\\mathbf{m}) = -\\log p(\\mathbf{m}).
        """
        pass

    @_abstractmethod
    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        """Computes the misfit gradient of the prior at the given coordinates.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Numpy array shaped as (dimensions, 1) representing a column vector
            containing the coordinates :math:`\\mathbf{m}`.

        Returns
        -------
        gradient : numpy.ndarray
            The prior misfit gradient :math:`\\nabla_\\mathbf{m}\\chi`.


        The prior misfit gradient is related to the prior probability density
        as:

        .. math::

            \\nabla_\\mathbf{m} \\chi_\\text{prior} (\\mathbf{m}) = -
            \\nabla_\\mathbf{m} \\log p(\\mathbf{m}).
        """
        pass

    @_abstractmethod
    def generate(self) -> _numpy.ndarray:
        """A method to generate samples from the prior.
        
        Raises
        ------
        TypeError
            If the prior does not allow generation of samples.
        """
        pass

    @_abstractmethod
    def corrector(self, coordinates: _numpy.ndarray, momentum: _numpy.ndarray):
        """A method to correct an HMC particle, which is called after a single time
        integration step.


        Parameters
        ----------
        coordinates : numpy.ndarray
            Numpy array shaped as (dimensions, 1) representing a column vector
            containing the coordinates :math:`\\mathbf{m}` upon which to operate by
            reference.
        momentum : numpy.ndarray
            Numpy array shaped as (dimensions, 1) representing a column vector
            containing the momenta :math:`\\mathbf{p}` upon which to operate by
            reference.


        This method is allowed to include many things, but the most relevant
        is for bounded distributions. If no corrections are needed, a pass method
        should be implemented. One can call multiple correctors from this function.
        """
        pass

    def bounds_corrector(self, coordinates: _numpy.ndarray, momentum: _numpy.ndarray):
        """A general corrector for reflecting boundaries in model space. Requires
        lower_bounds and upper_bounds to be initialized to `numpy.ndarray`'s of correct
        size.


        Parameters
        ----------
        coordinates : numpy.ndarray
            Numpy array shaped as (dimensions, 1) representing a column vector
            containing the coordinates :math:`\\mathbf{m}` upon which to operate by
            reference.
        momentum : numpy.ndarray
            Numpy array shaped as (dimensions, 1) representing a column vector
            containing the momenta :math:`\\mathbf{p}` upon which to operate by
            reference.

        
        If a coordinate dimension exceeds its bounds, the coordinate is mirrored across
        the boundary and its momentum is negated.
        """
        if self.lower_bounds is not None:
            # Lower bound correction ---------------------------------------------------
            too_low = coordinates < self.lower_bounds
            coordinates[too_low] += 2 * (
                self.lower_bounds[too_low] - coordinates[too_low]
            )
            momentum[too_low] *= -1.0
        if self.upper_bounds is not None:
            # Upper bound correction ---------------------------------------------------
            too_high = coordinates > self.upper_bounds
            coordinates[too_high] += 2 * (
                self.upper_bounds[too_high] - coordinates[too_high]
            )
            momentum[too_high] *= -1.0


class Normal(_AbstractPrior):
    """Normal distribution in model space.

    Parameters
    ----------
    dimensions : int
        Dimension of the distribution.
    means : numpy.ndarray
        Numpy array shaped as (dimensions, 1) containing the means of the distribution.
    covariance : numpy.ndarray
        Numpy array shaped as either as (dimensions, dimensions) or (dimensions, 1).
        This array represents either the full covariance matrix for a multivariate
        Gaussian, or an column vector with variances for `dimensions` separate
        uncorrelated Gaussians.
    lower_bounds: numpy.ndarray
        Numpy array of shape (dimensions, 1) that contains the lower limits of each
        parameter.
    upper_bounds: numpy.ndarray
        Numpy array of shape (dimensions, 1) that contains the upper limits of each
        parameter.

    """

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

        self.diagonal: bool = False
        """Indicator whether or not the covariance matrix is diagonal, i.e. if the
        distribution is uncorrelated."""

        self.means: _numpy.ndarray = None
        """Means in model space."""

        self.covariance: _numpy.ndarray = None
        """Covariance matrix in model space."""

        if means is None and covariance is None:
            # Neither means nor covariance is provided ---------------------------------
            _cprint(
                "Neither means or covariance matrix provided. Generating random means"
                "and variances.",
                "yellow",
            )
            self.means = _numpy.random.rand(dimensions, 1)
            self.covariance = _make_spd_matrix(self.dimensions)

        elif means is None or covariance is None:
            # Only one of means or covariance is provided ------------------------------
            raise ValueError(
                "Only one of means or covariance matrix provided. Not sure what to do!"
            )
        else:
            # Both means and covariance are provided -----------------------------------

            # Parse means
            if means.shape != (self.dimensions, 1):
                raise ValueError("Incorrect size of means vector.")
            self.means: _numpy.ndarray = means

            # Parse covariance
            if covariance.shape == (means.size, means.size):
                # Supplied a full covariance matrix
                self.diagonal = False
            elif covariance.shape == (means.size, 1):
                # Supplied a diagonal of a covariance matrix
                self.diagonal = True
                _cprint(
                    "Seem that you only passed a vector as the covariance matrix. It"
                    "will be used as the covariance diagonal.",
                    "yellow",
                )
            else:
                raise ValueError("Incorrect size of covariance matrix.")
            self.covariance = covariance

        # Precomputing inverses to speed up misfit and gradient computation ------------
        if self.diagonal:
            self.inverse_covariance = 1.0 / self.covariance
        else:
            self.inverse_covariance = _numpy.linalg.inv(self.covariance)

        # Process optional bounds ------------------------------------------------------
        if lower_bounds is not None and lower_bounds.shape == (self.dimensions, 1):
            self.lower_bounds = lower_bounds
            self.bounded = True  # TODO phase out self.bounded
        elif lower_bounds is not None:
            raise ValueError("Incorrect size of lower bounds vector.")

        if upper_bounds is not None and upper_bounds.shape == (self.dimensions, 1):
            self.upper_bounds = upper_bounds
            self.bounded = True  # TODO phase out self.bounded
        elif upper_bounds is not None:
            raise ValueError("Incorrect size of upper bounds vector.")

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        if self.bounded:  # TODO phase out self.bounded
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
        raise NotImplementedError("This function is not finished yet")

    def corrector(self, coordinates, momentum):
        self.bounds_corrector(coordinates, momentum)


class LogNormal(_AbstractPrior):
    """Normal distribution in logarithmic model space.


    Parameters
    ----------
    dimensions : int
        Dimension of the distribution.
    means : numpy.ndarray
        Numpy array shaped as (dimensions, 1) containing the means of the distribution
        in logarithmic model space.
    covariance : numpy.ndarray
        Numpy array shaped as either as (dimensions, dimensions) or (dimensions, 1).
        This array represents either the full covariance matrix for a multivariate
        Gaussian, or an column vector with variances for `dimensions` separate
        uncorrelated Gaussians, all in logarithmic model space.
    lower_bounds: numpy.ndarray
        Numpy array of shape (dimensions, 1) that contains the lower limits of each
        parameter.
    upper_bounds: numpy.ndarray
        Numpy array of shape (dimensions, 1) that contains the upper limits of each
        parameter.


    TODO Validate this class' methods.
    """

    def __init__(
        self,
        dimensions: int,
        means: _numpy.ndarray = None,
        covariance: _numpy.ndarray = None,
        lower_bounds: _numpy.ndarray = None,
        upper_bounds: _numpy.ndarray = None,
    ):
        self.name = "log normal (logarithmic Gaussian) prior"
        self.dimensions = dimensions

        self.diagonal: bool = False
        """Indicator whether or not the covariance matrix is diagonal, i.e. if the
        distribution is uncorrelated."""

        self.means: _numpy.ndarray = None
        """Means in logarithmic model space."""

        self.covariance: _numpy.ndarray = None
        """Covariance matrix in logarithmic model space."""

        if means is None and covariance is None:
            # Neither means nor covariance is provided
            _cprint(
                "Neither means or covariance matrix provided. Generating random means"
                "and variances.",
                "yellow",
            )
            self.means = _numpy.random.rand(dimensions, 1)
            self.covariance = _make_spd_matrix(self.dimensions)

        elif means is None or covariance is None:
            # Only one of means or covariance is provided
            raise ValueError(
                "Only one of means or covariance matrix provided. Not sure what to do!"
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
                    "Seem that you only passed a vector as the covariance matrix. It"
                    "will be used as the covariance diagonal.",
                    "yellow",
                )
            else:
                raise ValueError("Incorrect size of covariance matrix.")
            self.covariance: _numpy.ndarray = covariance

        # Precomputing inverses to speed up misfit and gradient computation ------------
        if self.diagonal:
            self.inverse_covariance = 1.0 / self.covariance
        else:
            self.inverse_covariance = _numpy.linalg.inv(self.covariance)

        # Process optional bounds ------------------------------------------------------
        if lower_bounds is not None and lower_bounds.shape == (self.dimensions, 1):
            self.lower_bounds = lower_bounds
            self.bounded = True  # TODO phase out self.bounded
        elif lower_bounds is not None:
            raise ValueError("Incorrect size of lower bounds vector.")

        if upper_bounds is not None and upper_bounds.shape == (self.dimensions, 1):
            self.upper_bounds = upper_bounds
            self.bounded = True  # TODO phase out self.bounded
        elif upper_bounds is not None:
            raise ValueError("Incorrect size of upper bounds vector.")

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        logarithmic_coordinates = _numpy.log(coordinates)
        if self.diagonal:
            return _numpy.sum(logarithmic_coordinates).item(0) + 0.5 * (
                (self.means - logarithmic_coordinates).T
                @ (self.inverse_covariance * (self.means - logarithmic_coordinates))
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
    """Unbouded uniform distribution in model space.
    
    Parameters
    ----------
    dimensions : int
        Dimension of the distribution.
    
    """

    def __init__(self, dimensions: int):
        self.name = "unbounded uniform prior"
        self.dimensions = dimensions

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        return 0.0

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        return _numpy.zeros((self.dimensions, 1))

    def generate(self) -> _numpy.ndarray:  # One shouldn't be able to do this
        raise TypeError(
            "This prior is unbounded, so it is impossible to generate samples"
            "from it."
        )

    def corrector(self, coordinates: _numpy.ndarray, momentum: _numpy.ndarray):
        pass


class Uniform(_AbstractPrior):
    """Bounded uniform distribution in model space
    

    Parameters
    ----------
    dimensions : int
        Dimension of the distribution.
    lower_bounds: numpy.ndarray
        Numpy array of shape (dimensions, 1) that contains the lower limits of each
        parameter.
    upper_bounds: numpy.ndarray
        Numpy array of shape (dimensions, 1) that contains the upper limits of each
        parameter.


    Raises
    ------
    ValueError
        Raised if the size of the bounds do not correspond to the dimensions.
    ValueError
        Raised if a lower bound is a larger value than a higher bound.

    """

    def __init__(
        self,
        dimensions: int,
        lower_bounds: _numpy.ndarray = None,
        upper_bounds: _numpy.ndarray = None,
    ):
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
        if not self.lower_bounds.shape == self.upper_bounds.shape == (dimensions, 1):
            raise ValueError("Bounds vectors are of incorrect size.")
        self.widths = self.upper_bounds - self.lower_bounds
        if not _numpy.all(self.widths > 0.0):
            raise ValueError("Some upper bounds are below lower bounds.")
        self._misfit = -_numpy.sum(_numpy.log(self.widths)).item()

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        return self._misfit

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        return _numpy.zeros((self.dimensions, 1))

    def generate(self) -> _numpy.ndarray:  # One shouldn't be able to do this
        return self.lower_bounds + self.widths * _numpy.random.rand(self.dimensions, 1)

    def corrector(self, coordinates, momentum):
        self.bounds_corrector(coordinates, momentum)


class CompositePrior(_AbstractPrior):
    """Prior distribution combined from multiple unconditional distributions
    
    Parameters
    ==========
    dimensions : int
        Combined dimension of all the separate distributions
    list_of_priors : List[_AbstractPrior]
        List of all separate priors.


    Raises
    ======
    ValueError
        Raised if the passed dimensions do not correspond to the sum of the separate
        dimensions of each prior.
    """

    def __init__(self, dimensions: int, list_of_priors: _List[_AbstractPrior]):
        self.name = "composite prior"
        self.separate_priors: List[_AbstractPrior] = list_of_priors

        # Assert that the passed priors actually do represent the correct amount of
        # dimensions
        computed_dimensions: int = 0
        for prior in self.separate_priors:
            computed_dimensions += prior.dimensions
        assert computed_dimensions == dimensions

        self.dimensions = dimensions

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        raise NotImplementedError()

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        raise NotImplementedError()

    def generate(self) -> _numpy.ndarray:
        raise NotImplementedError()

    def corrector(self, coordinates: _numpy.ndarray, momentum: _numpy.ndarray):
        raise NotImplementedError()


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
    return _numpy.dot(_numpy.dot(u, 1.0 + _numpy.diag(_numpy.random.rand(dim))), v)

