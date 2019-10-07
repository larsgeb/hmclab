"""Prior classes and associated methods.
"""
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
from typing import List as _List
from typing import Union as _Union
import warnings as _warnings
import numpy as _numpy


class _AbstractPrior(_ABC):
    """Prior abstract base class

    """

    name: str = "abstract prior"
    """Name of the prior."""

    dimensions: int = -1
    """Model space dimension of the prior."""

    lower_bounds: _Union[_numpy.ndarray, None] = None
    """Lower bounds for every parameter. If initialized to None, no bounds are used."""

    upper_bounds: _Union[_numpy.ndarray, None] = None
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


        This method is called many times in an HMC appraisal. It is therefore
        beneficial to optimize the implementation.
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


        This method is called many times in an HMC appraisal. It is therefore
        beneficial to optimize the implementation.
        """
        pass

    @_abstractmethod
    def generate(self) -> _numpy.ndarray:
        """Method to draw samples from the prior.

        Returns
        -------
        sample : numpy.ndarray
            A numpy array shaped as (dimensions, 1) containing a sample of the prior.

        Raises
        ------
        TypeError
            If the prior does not allow generation of samples.


        This method is mostly a convenience class. The algorithm itself does not
        require the implementation. Therefore an implementation as such will suffice::

            def generate(self) -> _numpy.ndarray:
                raise NotImplementedError("This function is not implemented.")

        """
        pass

    def corrector(self, coordinates: _numpy.ndarray, momentum: _numpy.ndarray):
        """Method to correct an HMC particle, which is called after every time
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

        """
        if self.lower_bounds is not None:
            # Lower bound correction
            too_low = coordinates < self.lower_bounds
            coordinates[too_low] += 2 * (
                self.lower_bounds[too_low] - coordinates[too_low]
            )
            momentum[too_low] *= -1.0
        if self.upper_bounds is not None:
            # Upper bound correction
            too_high = coordinates > self.upper_bounds
            coordinates[too_high] += 2 * (
                self.upper_bounds[too_high] - coordinates[too_high]
            )
            momentum[too_high] *= -1.0

    def update_bounds(
        self,
        lower_bounds: _Union[_numpy.ndarray, None],
        upper_bounds: _Union[_numpy.ndarray, None],
    ):
        """Method to update bounds of a prior distribution.

        Parameters
        ==========
        lower_bounds : numpy.ndarray or `None`
            Either an array shaped as (dimensions, 1) with floats for the lower bounds,
            or `None` for no bounds. If some dimensions should be bounded, while others
            should not, use ``-numpy.inf`` within the vector as needed.
        upper_bounds : numpy.ndarray or `None`
            Either an array shaped as (dimensions, 1) with floats for the upper bounds,
            or `None` for no bounds. If some dimensions should be bounded, while others
            should not, use ``numpy.inf`` within the vector as needed.


        This method updates the bounds of a distribution. Note that invocating it,
        requires both bounds to be passed. If only one is to be updated, simply pass
        the current object of the other bound::

            prior.update_bounds(numpy.zeros((4, 1)), prior.upper_bounds)


        If both vectors are passed, ensure that all upper bounds are above the
        corresponding lower bounds.

        """

        # Check the types --------------------------------------------------------------
        if lower_bounds is not None and type(lower_bounds) is not _numpy.ndarray:
            raise ValueError("Lower bounds object not understood.")
        if upper_bounds is not None and type(upper_bounds) is not _numpy.ndarray:
            raise ValueError("Upper bounds object not understood.")

        # Set the bounds ---------------------------------------------------------------
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds

        # Check for both arrays, if they are not None, if the dimension is correct. ----
        if (
            self.lower_bounds is not None
            and self.lower_bounds.shape != (self.dimensions, 1)
        ) or (
            self.upper_bounds is not None
            and self.upper_bounds.shape != (self.dimensions, 1)
        ):
            raise ValueError(f"Bounds vectors are of incorrect size.")

        # Check that all upper bounds are (finitely) above lower bounds ----------------
        if (
            self.lower_bounds is not None
            and self.upper_bounds is not None
            and _numpy.any(self.upper_bounds <= self.lower_bounds)
        ):
            raise ValueError("Bounds vectors are incompatible.")

    def misfit_bounds(self, coordinates: _numpy.ndarray) -> float:
        """Method to compute the misfit associated with the truncated part of the prior.
        """
        if (
            self.lower_bounds is not None
            and _numpy.any(coordinates < self.lower_bounds)
        ) or (
            self.upper_bounds is not None
            and _numpy.any(coordinates > self.upper_bounds)
        ):
            return _numpy.inf
        return 0.0


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
            _warnings.warn(
                "Neither means or covariance matrix provided. "
                "Generating random means and variances.",
                Warning,
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
                _warnings.warn(
                    "Seems that you only passed a vector as the covariance matrix. "
                    "It will be used as the covariance diagonal.",
                    Warning,
                )
            else:
                raise ValueError("Covariance matrix shape not understood.")
            self.covariance = covariance

        # Precomputing inverses to speed up misfit and gradient computation ------------
        if self.diagonal:
            self.inverse_covariance: _numpy.ndarray = 1.0 / self.covariance
        else:
            self.inverse_covariance: _numpy.ndarray = _numpy.linalg.inv(self.covariance)

        # Process optional bounds ------------------------------------------------------
        self.update_bounds(lower_bounds, upper_bounds)

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        """Method to compute the misfit of a Normal prior distribution.
        """
        if self.diagonal:
            return self.misfit_bounds(coordinates) + 0.5 * (
                (self.means - coordinates).T
                @ (self.inverse_covariance * (self.means - coordinates))
            ).item(0)
        else:
            return self.misfit_bounds(coordinates) + 0.5 * (
                (self.means - coordinates).T
                @ self.inverse_covariance
                @ (self.means - coordinates)
            ).item(0)

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        """Method to compute the gradient of a Normal prior distribution.
        """
        if self.diagonal:
            return -self.inverse_covariance * (
                self.means - coordinates
            ) + self.misfit_bounds(coordinates)
        else:
            return -self.inverse_covariance @ (
                self.means - coordinates
            ) + self.misfit_bounds(coordinates)

    def generate(self) -> _numpy.ndarray:
        raise NotImplementedError("This function is not finished yet")


class Sparse(_AbstractPrior):
    """L1 prior.

    Least absolute deviations, Laplace distribution, LASSO

    TODO: Implement distribution's location other than the 0-vector.
    """

    def __init__(
        self,
        dimensions: int,
        dispersion: float = 1,
        lower_bounds: _numpy.ndarray = None,
        upper_bounds: _numpy.ndarray = None,
    ):
        self.dimensions = dimensions
        self.dispersion = dispersion
        self.update_bounds(lower_bounds, upper_bounds)

    def misfit(self, coordinates) -> float:
        """Method to compute the misfit of a L1 prior distribution.
        """
        return (
            self.misfit_bounds(coordinates)
            + _numpy.sum(_numpy.abs(coordinates)) / self.dispersion
        )

    def gradient(self, coordinates):
        """Method to compute the gradient of a L1 prior distribution.
        """
        # The derivative of the function |x| is simply 1 or -1, depending on the sign
        # of x.
        return (
            self.misfit_bounds(coordinates) + _numpy.sign(coordinates) / self.dispersion
        )

    def generate(self):
        raise NotImplementedError()


class LogNormal(Normal):
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

    def __init__(self, *args, **kwargs):
        self.name = "log Gaussian prior"

        # Re-use the constructor of the superclass (Normal).
        super(LogNormal, self).__init__(*args, **kwargs)

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        """Method to compute the misfit of a log Normal prior distribution.
        """
        # This prior is only non-zero for positive values of the coordinates. ----------
        if _numpy.any(coordinates <= 0):
            return _numpy.inf

        # Compute logarithmic coordinates and misfit -----------------------------------
        logarithmic_coordinates = _numpy.log(coordinates)
        if self.diagonal:
            return (
                _numpy.sum(logarithmic_coordinates).item(0)
                + 0.5
                * (
                    (self.means - logarithmic_coordinates).T
                    @ (self.inverse_covariance * (self.means - logarithmic_coordinates))
                ).item(0)
                + self.misfit_bounds(coordinates)
            )
        else:
            return (
                _numpy.sum(logarithmic_coordinates).item(0)
                + 0.5
                * (
                    (self.means - logarithmic_coordinates).T
                    @ self.inverse_covariance
                    @ (self.means - logarithmic_coordinates)
                ).item(0)
                + self.misfit_bounds(coordinates)
            )

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        """Method to compute the gradient of a log Normal prior distribution.

        TODO Verify these formulas!
        """
        # Compute logarithmic coordinates and gradient ---------------------------------
        logarithmic_coordinates = _numpy.log(coordinates)
        if self.diagonal:
            return (
                (
                    -self.inverse_covariance
                    * (self.means - logarithmic_coordinates)
                    / coordinates
                )
                + _numpy.sum(1.0 / coordinates)
                + self.misfit_bounds(coordinates)
            )
        else:
            return (
                (
                    -self.inverse_covariance
                    @ (self.means - logarithmic_coordinates)
                    / coordinates
                )
                + _numpy.sum(1.0 / coordinates)
                + self.misfit_bounds(coordinates)
            )

    def generate(self) -> _numpy.ndarray:
        raise NotImplementedError("This function is not implemented yet.")


class Uniform(_AbstractPrior):
    """Uniform bounded or unbouded prior in model space.

    Parameters
    ----------
    dimensions : int
        Dimension of the distribution.
    lower_bounds: numpy.ndarray or None
        Numpy array of shape (dimensions, 1) that contains the lower limits of each
        parameter.
    upper_bounds: numpy.ndarray or None
        Numpy array of shape (dimensions, 1) that contains the upper limits of each
        parameter.

    """

    def __init__(
        self,
        dimensions: int,
        lower_bounds: _numpy.ndarray = None,
        upper_bounds: _numpy.ndarray = None,
    ):
        self.name = "uniform prior"
        self.dimensions = dimensions
        self.update_bounds(lower_bounds, upper_bounds)

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        """Method to compute the misfit of a uniform distribution.
        """
        return self.misfit_bounds(coordinates)

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        """Method to compute the gradient of a uniform distribution.
        """
        return _numpy.zeros((self.dimensions, 1)) + self.misfit_bounds(coordinates)

    def generate(self) -> _numpy.ndarray:
        raise NotImplementedError("This function is not implemented yet.")


class CompositePrior(_AbstractPrior):
    """Prior distribution combined from multiple unconditional distributions.

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


    This class can be used when two or more sets of coordinates should be described by
    different priors, e.g. when one set requires a Normal distribution and another a
    uniform distribution.
    """

    def __init__(
        self,
        dimensions: int,
        list_of_priors: _List[_AbstractPrior] = None,
        lower_bounds: _numpy.ndarray = None,
        upper_bounds: _numpy.ndarray = None,
    ):
        self.name = "composite prior"

        if list_of_priors is not None:
            self.separate_priors: _List[_AbstractPrior] = list_of_priors
        else:
            _warnings.warn(
                f"No subpriors were passed, generating {dimensions} random subpriors.",
                Warning,
            )
            available_priors = _AbstractPrior.__subclasses__()
            available_priors.remove(CompositePrior)
            selected_classes = _numpy.random.choice(available_priors, dimensions)
            self.separate_priors = [
                selected_class(1) for selected_class in selected_classes
            ]

        self.enumerated_dimensions: _numpy.ndarray = _numpy.empty(
            (len(self.separate_priors))
        )
        """This object describes how many dimensions each prior has, ordered according
        to ``CompositePrior.separate_priors``. Sums to ``CompositePrior.dimesions``.
        """

        # Assert that the passed priors actually do represent the correct amount of
        # dimensions, and seperately extract the size of each prior
        computed_dimensions: int = 0
        for i_prior, prior in enumerate(self.separate_priors):
            computed_dimensions += prior.dimensions
            self.enumerated_dimensions[i_prior] = prior.dimensions

        print(computed_dimensions)
        assert computed_dimensions == dimensions

        self.enumerated_dimensions_cumulative: _numpy.ndarray = _numpy.cumsum(
            self.enumerated_dimensions, dtype="int"
        )[:-1]
        """This object describes each separate prior index in combined model space. Invoking
        ``numpy.split(m, CompositePrior.enumerated_dimensions_cumulative)[:-1])``
        splits a vector appropriately for all separate priors.
        """

        self.dimensions = dimensions

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        # Split coordinates for all sub-priors -----------------------------------------
        split_coordinates = _numpy.split(
            coordinates, self.enumerated_dimensions_cumulative
        )

        misfit = 0.0

        # Loop over priors and add misfit ----------------------------------------------
        for i_prior, prior in enumerate(self.separate_priors):
            misfit += prior.misfit(split_coordinates[i_prior])

        return misfit + self.misfit_bounds(coordinates)

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        # Split coordinates for all sub-priors -----------------------------------------
        split_coordinates = _numpy.split(
            coordinates, self.enumerated_dimensions_cumulative
        )

        gradients = []

        # Loop over priors and compute gradient ----------------------------------------
        for i_prior, prior in enumerate(self.separate_priors):
            gradients.append(prior.gradient(split_coordinates[i_prior]))

        # Vertically stack gradients ---------------------------------------------------
        gradient = _numpy.vstack(gradients)

        assert gradient.shape == coordinates.shape

        return gradient + self.misfit_bounds(coordinates)

    def generate(self) -> _numpy.ndarray:
        raise NotImplementedError("This function is not implemented yet.")

    def corrector(self, coordinates: _numpy.ndarray, momentum: _numpy.ndarray):
        """Override method to correct an HMC particle for composite prior, which is
        called after every time integration step. Calls all sub-correctors

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

        """
        # Start with bounds of CompositePrior ------------------------------------------
        if self.lower_bounds is not None:
            # Lower bound correction
            too_low = coordinates < self.lower_bounds
            coordinates[too_low] += 2 * (
                self.lower_bounds[too_low] - coordinates[too_low]
            )
            momentum[too_low] *= -1.0
        if self.upper_bounds is not None:
            # Upper bound correction
            too_high = coordinates > self.upper_bounds
            coordinates[too_high] += 2 * (
                self.upper_bounds[too_high] - coordinates[too_high]
            )
            momentum[too_high] *= -1.0

        # Split coordinates and momenta for all sub-priors -----------------------------
        split_coordinates = _numpy.split(
            coordinates, self.enumerated_dimensions_cumulative
        )
        split_momenta = _numpy.split(momentum, self.enumerated_dimensions_cumulative)

        for i_prior, prior in enumerate(self.separate_priors):

            if prior.lower_bounds is not None:
                # Lower bound correction
                too_low = split_coordinates[i_prior] < prior.lower_bounds
                split_coordinates[i_prior][too_low] += 2 * (
                    prior.lower_bounds[too_low] - split_coordinates[i_prior][too_low]
                )
                split_momenta[i_prior][too_low] *= -1.0
            if prior.upper_bounds is not None:
                # Upper bound correction
                too_high = split_coordinates[i_prior] > prior.upper_bounds
                split_coordinates[i_prior][too_high] += 2 * (
                    prior.upper_bounds[too_high] - split_coordinates[i_prior][too_high]
                )
                split_momenta[i_prior][too_high] *= -1.0


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
