"""Collection of essential distributions don't are simple enough not to warrant their
own file.

"""

from abc import abstractmethod as _abstractmethod
from typing import List as _List
from typing import Union as _Union

import numpy as _numpy

from hmclab.Helpers import RandomMatrices as _RandomMatrices
from hmclab.Helpers.BetterABC import abstractattribute as _abstractattribute
from hmclab.Helpers.BetterABC import ABCMeta as _ABCMeta
from hmclab.Helpers import CustomExceptions as _CustomExceptions


from hmclab.Helpers.CustomExceptions import AbstractMethodError as _AbstractMethodError


class _AbstractDistribution(metaclass=_ABCMeta):
    """Abstract base class for distributions.

    This class is used as the template for any distribution supplied in the package or
    made by the user. It ensures key componenets are present (such as abstract methods)
    and takes care of bounded distributions.

    The abstract methods (e.g. functions that *need* to be created by the user) of this
    class are:
    1. :meth:`hmclab.Distributions._AbstractDistribution.misfit`
    2. :meth:`hmclab.Distributions._AbstractDistribution.gradient`
    Make sure the signature of these functions is correct when implementing. Special
    care needs to be given to input and output shapes of NumPy arrays, all of which
    should be column vectors (n×1). Reshaping can be done within the function at will.

    One abstract attribute is also required:
    :meth:`hmclab.Distributions._AbstractDistribution.dimensions`


    """

    name: str = None
    """Name of the distribution."""

    @_abstractattribute
    def dimensions(self) -> int:
        """Dimensionality of misfit space.

        This is an abstract parameter. If it is not defined either in your class
        directly or in its constructor (the __init__ function) then attempting to use
        the class will raise a NotImplementedError.

        Access it like a parameter, not a function: :code:`distribution.dimensions`.
        """
        raise _AbstractMethodError

    lower_bounds: _Union[_numpy.ndarray, None] = None
    """Lower bounds for every parameter. If initialized to None, no bounds are used."""

    upper_bounds: _Union[_numpy.ndarray, None] = None
    """Upper bounds for every parameter. If initialized to None, no bounds are used."""

    normalized: bool = False
    """Boolean describing if the distribution is normalized.
    
    Boolean describing if the distribution is normalized, i.e. if we can use it in
    mixtures of distributions. Is computed typically only after running
    :meth:`hmclab.Distributions._AbstractDistribution.normalize`"""

    @_abstractmethod
    def misfit(self, coordinates: _numpy.ndarray) -> float:
        """Compute misfit of distribution.

        Misfit computation (e.g. log likelihood) of the distribution. This method is
        present in all implemented derived classes, and should be present, with this
        exact signature, in all user-implemented derived classes.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Numpy array shaped as (dimensions, 1) representing a column vector
            containing the coordinates :math:`\\mathbf{m}`.

        Returns
        -------
        misfit : float
            The distribution misfit :math:`\\chi`.


        The distribution misfit is related to the distribution probability density as:

        .. math::

            \\chi_\\text{distribution} (\\mathbf{m}) \propto -\\log p(\\mathbf{m}).


        This method is called many times in an HMC appraisal. It is therefore
        beneficial to optimize the implementation.

        Note that the distribution need not be normalized, except when using mixtures of
        distributions. Therefore, distributions for which the normalization constant is
        intractable should not be used in mixtures. These distributions can be combined
        using Bayes' rule with other mixtures.
        """
        raise _AbstractMethodError()

    def misfit_v(self, coordinates: _numpy.ndarray, vectors_in_dim=None) -> float:

        if coordinates.size == self.dimensions:
            coordinates.shape = (coordinates.size, 1)
            return self.misfit(coordinates)

        if vectors_in_dim is None:
            assert (
                len(coordinates.shape) == 2
            ), "Can only take 1D or 2D arrays as input if not specifying vector input dimension"
            if coordinates.shape[0] == self.dimensions:
                vectors_in_dim = 0
            elif coordinates.shape[1] == self.dimensions:
                vectors_in_dim = 1
            else:
                raise AttributeError(
                    f"Don't know what to do with input of shape {coordinates.shape}."
                )

        _misfits = _numpy.zeros_like(_numpy.take(coordinates, 0, vectors_in_dim))

        it = _numpy.ndenumerate(_misfits)

        for _index, _element in it:
            _index_v = list(_index)
            _index_v.insert(vectors_in_dim, slice(None, None, None))
            _index_v.append(None)
            _index_v = tuple(_index_v)

            _misfits[_index] = self.misfit(coordinates[_index_v])

        return _misfits

    @_abstractmethod
    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        """Compute gradient of distribution.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Numpy array shaped as (dimensions, 1) representing a column vector
            containing the coordinates :math:`\\mathbf{m}`.

        Returns
        -------
        gradient : numpy.ndarray
            The distribution misfit gradient :math:`\\nabla_\\mathbf{m}\\chi`.


        The distribution misfit gradient is related to the distribution probability
        density as:

        .. math::

            \\nabla_\\mathbf{m} \\chi_\\text{distribution} (\\mathbf{m}) = -
            \\nabla_\\mathbf{m} \\log p(\\mathbf{m}).


        This method is called many times in an HMC appraisal. It is therefore
        beneficial to optimize the implementation.
        """
        raise _AbstractMethodError()

    def normalize(self):
        """Normalize distribution.

        Method to compute the normalization constant of a distribution. As this might
        take significant time, it is not done in initialization.

        Raises
        ------
        AttributeError
            An AttributeError is raised if the distribution provides no way to be
            normalized, e.g. when the normalization constant is intractable.


        """
        raise AttributeError("This distribution is not normalizable.")

    @_abstractmethod
    def generate(self, repeat=1, rng=_numpy.random.default_rng()) -> _numpy.ndarray:
        """Draw samples from distribution.

        Returns
        -------
        sample : numpy.ndarray
            A numpy array shaped as (dimensions, repeat) containing a sample of the
            distribution.

        Raises
        ------
        NotImplementedError
            If the distribution does not allow generation of samples.


        This method is mostly a convenience class. The algorithm itself does not
        require the implementation.

        """
        raise _AbstractMethodError()

    @staticmethod
    def create_default(dimensions: int) -> "_AbstractDistribution":
        """Create default instance.

        Method to create a default version of the distribution, given a specific
        dimensionality. Used in automated testing. Can be used on the class instead of
        an instance, e.g.::

           class.create_default(10)

        Parameters
        ----------
        dimensions : int
            Integer corresponding to the amount of free parameters the distribution
            should have.

        Returns
        -------
        distribution : derivative of _AbstractDistribution
            An instance of the derived class with the requested amount of free
            parameters.

        Raises
        ------
        NotImplementedError
            A NotImplementedError is raised if the distribution provides no way create
            a model for the requested dimensionality.

        """
        raise NotImplementedError()

    def corrector(self, coordinates: _numpy.ndarray, momentum: _numpy.ndarray):
        """Correct HMC trajectory.

        Method to correct an HMC particle for bounded distributions, which is called
        after every time integration step.

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
        lower: _Union[_numpy.ndarray, None] = None,
        upper: _Union[_numpy.ndarray, None] = None,
    ):
        """Update bounded distribution.

        This method updates the bounds of a distribution. Note that invocating it,
        does not require both bounds to be passed.

        If both vectors are passed, ensure that all upper bounds are above the
        corresponding lower bounds.

        Parameters
        ----------
        lower : numpy.ndarray or `None`
            Either an array shaped as (dimensions, 1) with floats for the lower bounds,
            or `None` for no bounds. If some dimensions should be bounded, while others
            should not, use ``-numpy.inf`` within the vector as needed.
        upper : numpy.ndarray or `None`
            Either an array shaped as (dimensions, 1) with floats for the upper bounds,
            or `None` for no bounds. If some dimensions should be bounded, while others
            should not, use ``numpy.inf`` within the vector as needed.

        Raises
        ------
        ValueError
           A ValueError is raised if the supplied upper and lower bounds are
           incompatible.

        """

        old_limits = (self.lower_bounds, self.upper_bounds)

        if type(upper) == list:
            upper = _numpy.array(upper)[:, None]

        if type(lower) == list:
            lower = _numpy.array(lower)[:, None]

        # Set the bounds ---------------------------------------------------------------
        self.upper_bounds = upper
        self.lower_bounds = lower

        # Check the types --------------------------------------------------------------
        if lower is not None and type(lower) is not _numpy.ndarray:
            # Lower bound is wrong

            # Reset bounds
            self.lower_bounds, self.upper_bounds = old_limits

            # Raise error
            raise ValueError("Lower bounds object not understood.")

        if upper is not None and type(upper) is not _numpy.ndarray:
            # Upper bound is wrong

            # Reset bounds
            self.lower_bounds, self.upper_bounds = old_limits

            # Raise error
            raise ValueError("Upper bounds object not understood.")

        # Check for both arrays; if they are not None, if the dimension is correct. ----
        if (
            self.lower_bounds is not None
            and self.lower_bounds.shape != (self.dimensions, 1)
        ) or (
            self.upper_bounds is not None
            and self.upper_bounds.shape != (self.dimensions, 1)
        ):
            # Reset bounds
            self.lower_bounds, self.upper_bounds = old_limits

            # Raise error
            raise ValueError("Bounds vectors are of incorrect size.")

        # Check that all upper bounds are (finitely) above lower bounds ----------------
        if (
            self.lower_bounds is not None
            and self.upper_bounds is not None
            and _numpy.any(self.upper_bounds <= self.lower_bounds)
        ):
            # Reset bounds
            self.lower_bounds, self.upper_bounds = old_limits

            # Raise error
            raise ValueError("Bounds vectors are incompatible.")

    def misfit_bounds(self, coordinates: _numpy.ndarray) -> float:
        """Compute misfit of bounded distribution.

        Method to compute the misfit associated with the truncated part of the
        distribution. Used internally."""
        if (
            self.lower_bounds is not None
            and _numpy.any(coordinates < self.lower_bounds)
        ) or (
            self.upper_bounds is not None
            and _numpy.any(coordinates > self.upper_bounds)
        ):
            return _numpy.inf
        return 0.0


class StandardNormal1D(_AbstractDistribution):
    """Standard normal distribution in 1 dimension."""

    mean = 0.0
    """Mean of the standard Normal distribution"""

    std = 1.0
    """Standard deviation of the standard Normal distribution"""

    dimensions = 1
    """Amount of dimensions on which the distribution is defined"""

    name = "Standard normal distribution in 1 dimension."
    """Name of the distribution."""

    def __init__(self, temperature=1.0):
        """ """
        self.temperature = temperature

    def misfit(self, m: _numpy.ndarray) -> float:
        """Compute misfit of distribution.

        Method to compute the misfit of a distribution for a given model m. See,
        :meth:`hmclab.Distributions._AbstractDistribution.misfit` for details.
        """
        _CustomExceptions.Assertions.assert_shape(m, (1, 1))

        return self.misfit_bounds(m) + (0.5 * m[0, 0] ** 2).item() / self.temperature

    def gradient(self, m: _numpy.ndarray) -> _numpy.ndarray:
        """Compute gradient of distribution.

        Method to compute the gradient of a distribution for a given model m. See,
        :meth:`hmclab.Distributions._AbstractDistribution.gradient` for details.
        """
        _CustomExceptions.Assertions.assert_shape(m, (1, 1))

        return m / self.temperature

    @staticmethod
    def create_default(dimensions: int) -> "StandardNormal1D":
        if dimensions == 1:
            return StandardNormal1D()
        else:
            raise _CustomExceptions.InvalidCaseError()

    def generate(self, repeat=1, rng=_numpy.random.default_rng()) -> _numpy.ndarray:
        return rng.normal(0.0, 1.0, (1, repeat))


class Normal(_AbstractDistribution):
    """Normal distribution in model space.


    Parameters
    ----------
    dimensions : int
        Dimension of the distribution.
    means : numpy.ndarray
        Numpy array shaped as (dimensions, 1) containing the means of the
        distribution.
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
        means: _numpy.ndarray,
        covariance: _Union[_numpy.ndarray, float, None],
        inverse_covariance: _Union[_numpy.ndarray, float, None] = None,
        lower_bounds: _numpy.ndarray = None,
        upper_bounds: _numpy.ndarray = None,
    ):

        self.name = "Gaussian (normal) distribution"

        if type(means) == list:
            means = _numpy.array(means)[:, None]
        if type(covariance) == list:
            covariance = _numpy.array(covariance)[:, None]

        # Automatically get dimensionality from means
        if type(means) == float or type(means) == int:
            self.dimensions: int = 1
        else:
            self.dimensions: int = means.size
        """Amount of dimensions on which the distribution is defined, should agree with
        means and covariance, and optionally coordinate_transformation."""

        self.diagonal: bool = False
        """Indicator whether or not the covariance matrix is diagonal, i.e. if the
        distribution is uncorrelated."""

        self.means: _numpy.ndarray = None
        """Means in model space"""

        self.covariance: _numpy.ndarray = None
        """Covariance matrix in model space"""

        self.inverse_covariance: _numpy.ndarray = None
        """Inverse covariance matrix"""

        self.normalization_constant = 0.0
        """Covariance matrix determinant and dimensionality factored in single
        likelihood term. Uncomputed if normalized() is never called."""

        self.generate_ready = False

        # Parse means
        if type(means) == float or type(means) == int:
            means = _numpy.ones((self.dimensions, 1)) * means
        else:
            means.shape = (self.dimensions, 1)
        self.means: _numpy.ndarray = means

        # Parse covariance
        if (
            type(covariance) == float
            or type(covariance) == _numpy.float64
            or type(covariance) == _numpy.float32
            or type(covariance) == int
        ):
            covariance = _numpy.float64(covariance)
            self.diagonal = True
        elif covariance.shape == (means.size, means.size):
            # Supplied a full covariance matrix, could be either NumPy or SciPy
            # matrix.
            self.diagonal = False
        else:
            # Supplied a diagonal of a covariance matrix
            self.diagonal = True
            covariance.shape = (self.dimensions, 1)
        self.covariance = covariance

        # Precomputing inverses to speed up misfit and gradient computation ------------
        if inverse_covariance is not None:
            # There are many ways in which one could optimize the computation of a
            # specific PD-matrix inverse. Let the user compute and provide it if wanted.
            self.inverse_covariance = inverse_covariance
        elif self.diagonal:
            # If the user does not provide one, at least check if the covariance matrix
            # is diagonal, which makes computation of the inverse scale much better.
            self.inverse_covariance: _numpy.ndarray = 1.0 / self.covariance
        else:
            # Else, brute force calculation of the inverse using numpy.
            self.inverse_covariance: _numpy.ndarray = _numpy.linalg.inv(self.covariance)

        if self.dimensions == 1:
            self.diagonal = True

        # Process optional bounds ------------------------------------------------------
        self.update_bounds(lower_bounds, upper_bounds)

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        """Method to compute the misfit of a Normal distribution distribution."""

        if self.diagonal:
            return (
                self.misfit_bounds(coordinates)
                + 0.5
                * (
                    (self.means - coordinates).T
                    @ (self.inverse_covariance * (self.means - coordinates))
                ).flatten()[0]
                + self.normalization_constant
            )
        else:
            return (
                self.misfit_bounds(coordinates)
                + 0.5
                * (
                    (self.means - coordinates).T
                    @ self.inverse_covariance
                    @ (self.means - coordinates)
                ).flatten()[0]
                + self.normalization_constant
            )

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        """Method to compute the gradient of the distribution."""

        if self.diagonal:
            return -self.inverse_covariance * (
                self.means - coordinates
            ) + self.misfit_bounds(coordinates)
        else:
            return -self.inverse_covariance @ (
                self.means - coordinates
            ) + self.misfit_bounds(coordinates)

    def normalize(self):
        if (
            type(self.covariance) == float
            or type(self.covariance) == _numpy.float64
            or type(self.covariance) == _numpy.float32
            or type(self.covariance) == int
        ):
            determinant = self.covariance**self.dimensions
        elif self.covariance.shape == (self.means.size, self.means.size):
            determinant = _numpy.linalg.det(self.covariance)
        elif self.covariance.shape == (self.means.size, 1):
            determinant = _numpy.prod(self.covariance)
        else:
            raise ValueError("Covariance matrix shape not understood.")

        self.normalization_constant = 0.5 * (
            _numpy.log(_numpy.abs(determinant))
            + self.dimensions * _numpy.log(2 * _numpy.pi)
        )

    def generate(self, repeat=1, rng=_numpy.random.default_rng()) -> _numpy.ndarray:

        if not self.generate_ready:
            if self.diagonal:
                self.standard_deviation = self.covariance**0.5
            else:
                # Perform Cholesky decomposition
                self.covariance_cholesky = _numpy.linalg.cholesky(self.covariance)
            self.generate_ready = True

        if self.diagonal:
            samples = (
                rng.normal(size=(self.dimensions, repeat)) * self.standard_deviation
                + self.means
            )
            return samples
        else:

            return (
                self.covariance_cholesky
                @ _numpy.random.default_rng().normal(size=(self.dimensions, repeat))
                + self.means
            )

    @staticmethod
    def create_default(dimensions: int, diagonal=False) -> "Normal":

        # Create random means
        means = _numpy.random.rand(dimensions, 1)

        if not diagonal:
            # Create a PD matrix with some extra definiteness by adding the identity
            correlation = _RandomMatrices.random_correlation_matrix(dimensions)

        # Standard deviations between 1 and 2
        standard_deviations = _numpy.diag(
            _numpy.random.rand(
                dimensions,
            )
            + 1
        )

        if diagonal:
            covariance = _numpy.diag(standard_deviations * standard_deviations)
        else:
            covariance = standard_deviations @ correlation @ standard_deviations

        return Normal(means, covariance)


class Laplace(_AbstractDistribution):
    """Laplace distribution in model space.

    Least absolute deviations, Laplace distribution, LASSO, L1

    """

    def __init__(
        self,
        means: _numpy.ndarray,
        dispersions: _Union[_numpy.ndarray, float, None],
        lower_bounds: _numpy.ndarray = None,
        upper_bounds: _numpy.ndarray = None,
    ):

        if type(means) == list:
            means = _numpy.array(means)[:, None]
        if type(dispersions) == list:
            dispersions = _numpy.array(dispersions)[:, None]

        # Automatically get dimensionality from means
        self.dimensions = means.size

        means.shape = (self.dimensions, 1)
        self.means = means
        """A float or numpy.ndarray of shape (dimensions, 1) of floats describing the
        mean of the uncorrelated multivariate Laplace distribution."""

        dispersions.shape = (self.dimensions, 1)
        self.dispersions = dispersions
        """A positive float or numpy.ndarray of shape (dimensions, 1) of positive floats
        describing the dispersion of the uncorrelated multivariate Laplace
        distribution."""

        self.inverse_dispersions = 1.0 / dispersions
        """A positive float or numpy.ndarray of shape (dimensions, 1) of positive floats
        describing the inverse dispersion of the uncorrelated multivariate Laplace
        distribution. Used to accelerate computations at the cost of memory usage."""

        self.normalization_constant = 0.0

        self.update_bounds(lower_bounds, upper_bounds)

    def misfit(self, coordinates) -> float:
        """Method to compute the misfit the distribution."""

        return (
            self.normalization_constant
            + self.misfit_bounds(coordinates)
            + (
                _numpy.sum(
                    _numpy.abs(coordinates - self.means) * self.inverse_dispersions
                )
            ).flatten()[0]
        )

    def gradient(self, coordinates):
        """Method to compute the gradient the distribution."""

        # The derivative of the function |x| is simply 1 or -1, depending on the sign
        # of x, subsequently scaled by the dispersion.
        return (
            self.misfit_bounds(coordinates)
            + _numpy.sign(coordinates - self.means) * self.inverse_dispersions
        )

    def normalize(self):
        if (
            type(self.dispersions) == float
            or type(self.dispersions) == _numpy.float64
            or type(self.dispersions) == _numpy.float32
            or type(self.dispersions) == int
        ):
            self.normalization_constant = _numpy.log(1.0 / (2.0 * self.dispersions))
        elif self.dispersions.shape == (self.means.size, 1):
            self.normalization_constant = _numpy.log(
                1.0 / (2.0 * (_numpy.prod(self.dispersions) ** (1.0 / self.dimensions)))
            )
        else:
            raise ValueError("Covariance matrix shape not understood.")

    def generate(self, repeat=1, rng=_numpy.random.default_rng()) -> _numpy.ndarray:

        samples = rng.laplace(
            loc=self.means, scale=self.dispersions, size=(self.dimensions, repeat)
        )

        return samples

    @staticmethod
    def create_default(dimensions: int) -> "Laplace":

        # Create random means
        means = _numpy.random.rand(dimensions, 1)

        # Create a PD matrix with some extra definiteness by adding the identity
        dispersions = 10 ** _numpy.random.rand(dimensions, 1)

        return Laplace(means, dispersions)


class Uniform(_AbstractDistribution):
    """Uniform bounded or unbouded distribution in model space.

    Parameters
    ----------
    lower_bounds: numpy.ndarray or None
        Numpy array of shape (dimensions, 1) that contains the lower limits of each
        parameter.
    upper_bounds: numpy.ndarray or None
        Numpy array of shape (dimensions, 1) that contains the upper limits of each
        parameter.

    """

    def __init__(
        self,
        lower_bounds: _numpy.ndarray,
        upper_bounds: _numpy.ndarray,
    ):
        self.name = "uniform distribution"

        lower_bounds = _numpy.asarray(lower_bounds)
        lower_bounds = _numpy.resize(lower_bounds, (lower_bounds.size, 1))
        upper_bounds = _numpy.asarray(upper_bounds)
        upper_bounds = _numpy.resize(upper_bounds, (upper_bounds.size, 1))

        # Automatically get dimensionality from bounds
        dimensions = lower_bounds.size

        self.dimensions = dimensions

        self.update_bounds(lower_bounds, upper_bounds)

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        """Method to compute the misfit of a uniform distribution."""
        return self.misfit_bounds(coordinates)

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        """Method to compute the gradient of a uniform distribution."""
        return _numpy.zeros((self.dimensions, 1)) + self.misfit_bounds(coordinates)

    def generate(self, repeat=1, rng=_numpy.random.default_rng()) -> _numpy.ndarray:

        return rng.uniform(
            self.lower_bounds, self.upper_bounds, (self.dimensions, repeat)
        )

    @staticmethod
    def create_default(dimensions: int) -> "Uniform":

        lower_bounds = _numpy.random.rand(dimensions, 1) * 5 - 10
        upper_bounds = _numpy.random.rand(dimensions, 1) * 5 + 10

        return Uniform(lower_bounds, upper_bounds)


class CompositeDistribution(_AbstractDistribution):
    """Distribution distribution combined from multiple unconditional distributions.

    Parameters
    ==========
    dimensions : int
        Combined dimension of all the separate distributions
    list_of_distributions : List[_AbstractDistribution]
        List of all separate distributions.


    Raises
    ======
    ValueError
        Raised if the passed dimensions do not correspond to the sum of the separate
        dimensions of each distribution.


    This class can be used when two or more sets of coordinates should be described by
    different distributions, e.g. when one set requires a Normal distribution and
    another a uniform distribution.
    """

    def __init__(
        self,
        list_of_distributions: _List[_AbstractDistribution] = None,
        lower_bounds: _numpy.ndarray = None,
        upper_bounds: _numpy.ndarray = None,
    ):
        self.name = "composite distribution"

        self.separate_distributions: _List[
            _AbstractDistribution
        ] = list_of_distributions

        self.enumerated_dimensions: _numpy.ndarray = _numpy.empty(
            (len(self.separate_distributions))
        )
        """This object describes how many dimensions each distribution has, ordered
        according to ``CompositeDistribution.separate_distributions``. Sums to
        ``CompositeDistribution.dimesions``.
        """

        # Assert that the passed distributions actually do represent the correct amount
        # of dimensions, and seperately extract the size of each distribution
        computed_dimensions: int = 0
        for i_distribution, distribution in enumerate(self.separate_distributions):
            computed_dimensions += distribution.dimensions
            self.enumerated_dimensions[i_distribution] = distribution.dimensions

        self.dimensions = computed_dimensions

        self.enumerated_dimensions_cumulative: _numpy.ndarray = _numpy.cumsum(
            self.enumerated_dimensions, dtype="int"
        )[:-1]
        """This object describes each separate distribution index in combined model
        space. Invoking
        ``numpy.split(m, CompositeDistribution.enumerated_dimensions_cumulative)[:-1])``
        splits a vector appropriately for all separate distributions.
        """

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        # Split coordinates for all sub-distributions ----------------------------------
        split_coordinates = _numpy.split(
            coordinates, self.enumerated_dimensions_cumulative
        )

        misfit = 0.0

        # Loop over distributions and add misfit ---------------------------------------
        for i_distribution, distribution in enumerate(self.separate_distributions):
            misfit += distribution.misfit(split_coordinates[i_distribution])

        return misfit + self.misfit_bounds(coordinates)

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        # Split coordinates for all sub-distributions ----------------------------------
        split_coordinates = _numpy.split(
            coordinates, self.enumerated_dimensions_cumulative
        )

        gradients = []

        # Loop over distributions and compute gradient ---------------------------------
        for i_distribution, distribution in enumerate(self.separate_distributions):
            gradients.append(distribution.gradient(split_coordinates[i_distribution]))

        # Vertically stack gradients ---------------------------------------------------
        gradient = _numpy.vstack(gradients)

        assert gradient.shape == coordinates.shape

        return gradient + self.misfit_bounds(coordinates)

    def generate(self, repeat=1, rng=_numpy.random.default_rng()) -> _numpy.ndarray:

        samples = []

        for distribution in self.separate_distributions:
            samples.append(distribution.generate(repeat=repeat, rng=rng))

        return _numpy.vstack(samples)

    def collapse_bounds(self):
        """Method to restructure all composite bounds into top level object."""
        raise NotImplementedError()

    def corrector(self, coordinates: _numpy.ndarray, momentum: _numpy.ndarray):
        """Override method to correct an HMC particle for composite distribution, which
        is called after every time integration step. Calls all sub-correctors only if
        the object does not have bounds itself.

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
        # Start with bounds of CompositeDistribution -----------------------------------
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

        # If they are not set, check subdistributions.
        if self.lower_bounds is None and self.upper_bounds is None:
            # Split coordinates and momenta for all sub-distributions ------------------
            split_coordinates = _numpy.split(
                coordinates, self.enumerated_dimensions_cumulative
            )
            split_momenta = _numpy.split(
                momentum, self.enumerated_dimensions_cumulative
            )

            # And loop over separate distributions to check bounds
            for i_distribution, distribution in enumerate(self.separate_distributions):

                if distribution.lower_bounds is not None:
                    # Lower bound correction
                    too_low = (
                        split_coordinates[i_distribution] < distribution.lower_bounds
                    )
                    split_coordinates[i_distribution][too_low] += 2 * (
                        distribution.lower_bounds[too_low]
                        - split_coordinates[i_distribution][too_low]
                    )
                    split_momenta[i_distribution][too_low] *= -1.0
                if distribution.upper_bounds is not None:
                    # Upper bound correction
                    too_high = (
                        split_coordinates[i_distribution] > distribution.upper_bounds
                    )
                    split_coordinates[i_distribution][too_high] += 2 * (
                        distribution.upper_bounds[too_high]
                        - split_coordinates[i_distribution][too_high]
                    )
                    split_momenta[i_distribution][too_high] *= -1.0

    @staticmethod
    def create_default(dimensions: int) -> "CompositeDistribution":

        # Create a list of all possible distributions
        available_distributions = _AbstractDistribution.__subclasses__()

        # We don't want to recursively create many distributions, so remove those
        for distribution_to_remove in [
            CompositeDistribution,
            AdditiveDistribution,
        ]:
            available_distributions.remove(distribution_to_remove)

        if dimensions != 2:
            # This guy only supports 2 dimensions
            available_distributions.remove(Himmelblau)

        # We select distributions at random
        selected_classes = _numpy.random.choice(available_distributions, dimensions)

        list_of_instances = [d_class.create_default(1) for d_class in selected_classes]

        return CompositeDistribution(list_of_instances)


class AdditiveDistribution(_AbstractDistribution):
    """Distribution generated by summing the characteristic functions of two other
    distributions.

    This is essentially the unnormalized Bayes' rule.
    """

    def __init__(
        self,
        list_of_distributions: _List[_AbstractDistribution],
        lower_bounds: _numpy.ndarray = None,
        upper_bounds: _numpy.ndarray = None,
    ):
        self.name = "additive distribution"

        # Automatically get dimensionality  from first distribution
        self.dimensions = list_of_distributions[0].dimensions

        self.separate_distributions: _List[
            _AbstractDistribution
        ] = list_of_distributions

        # Assert that the passed distributions are of the right dimension
        for i_distribution, distribution in enumerate(self.separate_distributions):
            assert distribution.dimensions == self.dimensions

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        self.collapse_bounds()

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        misfit = 0.0

        # Loop over distributions and add misfit ---------------------------------------
        for distribution in self.separate_distributions:
            misfit += distribution.misfit(coordinates)

        return misfit + self.misfit_bounds(coordinates)

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        gradient = _numpy.zeros((self.dimensions, 1))

        # Loop over distributions and compute gradient ---------------------------------
        for distribution in self.separate_distributions:
            gradient += distribution.gradient(coordinates)

        assert gradient.shape == coordinates.shape

        return gradient + self.misfit_bounds(coordinates)

    def generate(self, repeat=1, rng=_numpy.random.default_rng()) -> _numpy.ndarray:
        raise NotImplementedError(
            "Generating samples from this distribution is not implemented or supported."
        )

    def collapse_bounds(self):
        """Method to restructure all composite bounds into top level object."""
        # Iterate over all subdistributions
        for i_distribution, distribution in enumerate(self.separate_distributions):

            # Assert that every subdistribution has the right shape
            assert distribution.dimensions == self.dimensions

            # If the subdistribution has lower bounds ... act
            if distribution.lower_bounds is not None:

                # Assert the bounds have the right shape
                assert distribution.lower_bounds.shape == (self.dimensions, 1)

                if self.lower_bounds is None:
                    # If the top level distribution doesn't have lower bounds yet,
                    # simply add the new bounds
                    self.lower_bounds = distribution.lower_bounds
                else:
                    # If the top level distribution does already have lower bounds, take
                    #  the maximum of every separate bound
                    self.lower_bounds = _numpy.maximum(
                        self.lower_bounds, distribution.lower_bounds
                    )

            # If the subdistribution has upper bounds ... act
            if distribution.upper_bounds is not None:

                # Assert the bounds have the right shape
                assert distribution.upper_bounds.shape == (self.dimensions, 1)

                if self.upper_bounds is None:
                    # If the top level distribution doesn't have upper bounds yet,
                    # simply add the new bounds
                    self.upper_bounds = distribution.upper_bounds
                else:
                    # If the top level distribution does already have upper bounds, take
                    # the minimum of every separate bound
                    self.upper_bounds = _numpy.minimum(
                        self.upper_bounds, distribution.upper_bounds
                    )

    def add_distribution(self, distribution: _AbstractDistribution):
        """Add a distribution to the object."""
        assert distribution.dimensions == self.dimensions

        self.separate_distributions.append(distribution)

        self.collapse_bounds()

    def corrector(self, coordinates: _numpy.ndarray, momentum: _numpy.ndarray):
        """Override method to correct an HMC particle for additive distribution, which is
        called after every time integration step. Calls all sub-correctors only if the
        object does not have bounds itself.

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
        # Start with bounds of CompositeDistribution -----------------------------------
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

        # # If they are not set, check subdistributions.
        # if self.lower_bounds is None and self.upper_bounds is None:
        #     # Split coordinates and momenta for all sub-distributions ----------------
        #     split_coordinates = _numpy.split(
        #         coordinates, self.enumerated_dimensions_cumulative
        #     )
        #     split_momenta = _numpy.split(
        #         momentum, self.enumerated_dimensions_cumulative
        #     )

        #     # And loop over separate distributions to check bounds
        #     for i_dis, distribution in enumerate(self.separate_distributions):

        #         if distribution.lower_bounds is not None:
        #             # Lower bound correction
        #             too_low = split_coordinates[i_dis] < distribution.lower_bounds
        #             split_coordinates[i_dis][too_low] += 2 * (
        #                 distribution.lower_bounds[too_low]
        #                 - split_coordinates[i_dis][too_low]
        #             )
        #             split_momenta[i_dis][too_low] *= -1.0
        #         if distribution.upper_bounds is not None:
        #             # Upper bound correction
        #             too_high = split_coordinates[i_dis] > distribution.upper_bounds
        #             split_coordinates[i_dis][too_high] += 2 * (
        #                 distribution.upper_bounds[too_high]
        #                 - split_coordinates[i_dis][too_high]
        #             )
        #             split_momenta[i_dis][too_high] *= -1.0

    @staticmethod
    def create_default(dimensions: int) -> "AdditiveDistribution":

        # Create a list of all possible distributions
        available_distributions = _AbstractDistribution.__subclasses__()

        # We don't want to recursively create many distributions, so remove those
        for distribution_to_remove in [
            CompositeDistribution,
            AdditiveDistribution,
        ]:
            available_distributions.remove(distribution_to_remove)

        if dimensions != 2:
            # This guy only supports 2 dimensions
            available_distributions.remove(Himmelblau)

        # We select distributions at random
        selected_classes = _numpy.random.choice(available_distributions, 3)

        list_of_instances = [
            d_class.create_default(dimensions) for d_class in selected_classes
        ]

        return AdditiveDistribution(list_of_instances)


# This creates an alias for AdditiveDistribution
class BayesRule(AdditiveDistribution):
    """A class to apply (the unnormalized) Bayes' rule to two or more distributions."""

    pass


class Himmelblau(_AbstractDistribution):
    """Himmelblau's 2-dimensional function.

    Himmelblau's function is defined as:

    .. math::

        f(x,y)=(x^{2}+y-11)^{2}+(x+y^{2}-7)^{2}
    """

    name: str = "Himmelblau's function"
    dimensions: int = 2
    temperature: float = 1
    """Float representing the temperature (or annealing, :math:`T`) of Himmelblau's
    function.

    Alters the misfit function in the following way:

    .. math::

        f(x,y)_T=\\frac{f(x,y)}{T}
    """

    def __init__(self, temperature: float = 1):
        self.temperature = temperature

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        """Returns the value of Himmelblau's function at the given coordinates."""
        if coordinates.shape != (self.dimensions, 1):
            raise ValueError()
        x = coordinates[0, 0]
        y = coordinates[1, 0]
        return (
            self.misfit_bounds(coordinates)
            + (
                ((x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2) / self.temperature
            ).item()
        )

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        """Returns a numpy.ndarray shaped as (dimensions, 1) containing the gradient of
        Himmelblau's function at the given coordinates."""
        x = coordinates[0]
        y = coordinates[1]
        gradient = _numpy.zeros((self.dimensions, 1))
        gradient[0] = 2 * (2 * x * (x**2 + y - 11) + x + y**2 - 7)
        gradient[1] = 2 * (x**2 + 2 * y * (x + y**2 - 7) + y - 11)
        return gradient / self.temperature

    def generate(self, repeat=1, rng=_numpy.random.default_rng()) -> _numpy.ndarray:
        raise NotImplementedError(
            "Generating samples from this distribution is not implemented or supported."
        )

    @staticmethod
    def create_default(dimensions: int) -> "Himmelblau":

        if dimensions != 2:
            raise _CustomExceptions.InvalidCaseError()

        temperature = 1.0
        return Himmelblau(temperature=temperature)


class Mixture(_AbstractDistribution):
    def __init__(self, distributions, probabilities):

        self.distributions = distributions
        self.dimensions = self.distributions[0].dimensions

        self.probabilities = probabilities

        for dis in self.distributions:

            if not dis.normalized:
                dis.normalize()

            assert dis.dimensions == self.dimensions

    def misfit(self, m):
        misfits = [d.misfit(m) for d in self.distributions]


        with _numpy.errstate(divide = 'ignore'):
            _misfit = self.misfit_bounds(m) - _numpy.log(
                _numpy.sum(_numpy.exp(_numpy.log(self.probabilities) - misfits))
            )

        return _misfit

    def gradient(self, m):

        misfits = [d.misfit(m) for d in self.distributions]
        gradients = _numpy.array([d.gradient(m) for d in self.distributions])
        probs = _numpy.exp(_numpy.log(self.probabilities) - misfits)

        gr = _numpy.sum(
            _numpy.array([prob * (-grad) for prob, grad in zip(probs, gradients)]),
            axis=0,
        )

        return -gr / _numpy.sum(probs)

    @staticmethod
    def create_default(dimensions: int) -> "Mixture":

        Normal1 = Normal.create_default(dimensions)
        Normal2 = Normal.create_default(dimensions)

        return Mixture([Normal1, Normal2], [0.5, 0.5])

    def generate(self, repeat=1, rng=_numpy.random.default_rng()) -> _numpy.ndarray:

        generate_from = _numpy.random.choice(
            _numpy.arange(len(self.probabilities)),
            size=repeat,
            p=self.probabilities,
        )

        samples = []
        for _index, _repeats in _numpy.vstack(
            _numpy.unique(generate_from, return_counts=True)
        ).T:
            samples.append(self.distributions[_index].generate(_repeats))

        return _numpy.hstack(samples)


def EvaluationLimiter_ClassConstructor(
    base,
    limit,
    gradient_count: int = 1,
    throw_interrupt=True,
):
    class EvaluationLimiter(base):
        def __init__(self, *args, **kwargs):
            self.limit = limit
            self.gradient_count = gradient_count
            self.throw_interrupt = throw_interrupt

            if self.limit == 0:
                self.throw_interrupt = False

            self.evaluations = 0

            super().__init__(*args, **kwargs)

        def misfit(self, coordinates: _numpy.ndarray) -> float:

            if self.throw_interrupt and self.evaluations > self.limit:
                self.evaluations = 0
                raise KeyboardInterrupt

            self.evaluations += 1
            return super().misfit(coordinates)

        def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:

            if self.throw_interrupt and self.evaluations > self.limit:
                self.evaluations = 0
                raise KeyboardInterrupt

            self.evaluations += self.gradient_count
            return super().gradient(coordinates)

    return EvaluationLimiter
