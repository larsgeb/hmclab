"""Target (likelihood) classes and associated methods.
"""
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
import numpy as _numpy
from typing import Union as _Union
import scipy as _scipy
import warnings as _warnings


class _AbstractTarget(_ABC):
    """Abstract base class for inverse problem targets. Defines all required
    methods for derived classes.
    """

    name: str = "inverse problem target abstract base class"
    dimensions: int = -1

    def full_name(self) -> str:
        """Returns the full name of the target"""
        return self.name

    @_abstractmethod
    def misfit(self, coordinates: _numpy.ndarray) -> float:
        """Returns the misfit at the given coordinates. This is equal to the negative
        logarithm of the likelihood function: :math:`\\chi(m) = - \\log L(m)=
        - \\log p(m|d)`."""
        pass

    @_abstractmethod
    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        """Returns the gradient of the misfit at the given coordinates: :math:`∇_m
        \\chi(m) = - ∇_m \\log L(m)= - ∇_m \\log p(m|d)`."""
        pass


class LinearMatrix(_AbstractTarget):
    """Target model based on a linear forward model given as
    :math:`G \\mathbf{m} = \\mathbf{d}`
    """

    def __init__(
        self,
        dimensions: int,
        G: _Union[_numpy.ndarray, _scipy.sparse.spmatrix] = None,
        d: _numpy.ndarray = None,
        data_covariance: _Union[float, _numpy.ndarray, _scipy.sparse.spmatrix] = None,
    ):
        """Constructor for linear forward model target.

        Parameters
        ==========
        dimensions : int
            Dimension of the model space.
        G : numpy.ndarray or scipy.sparse.spmatrix
            Numpy ndarray or scipy sparse matrix sized as (datapoints, dimensions)
            containing the linear forward model. Defaults to unit matrix.
        d : numpy.ndarray
            Numpy ndarray shaped as (datapoints, 1) containing observed datapoints.
            Defaults to a vector of zeros.
        data_covariance : float or numpy.ndarray or scipy.sparse.spmatrix
            Optional object representing either scalar data variance, a diagonal data
            variance matrix [numpy.ndarray shaped as (datapoints, 1)], a full data
            covariance matrix [numpy.ndarray shaped as (datapoints, datapoints)] or a
            sparse data covariance matrix [scipy.sparse.spmatrix shaped as
            (datapoints, datapoints)].
        """
        self.dimensions = dimensions

        # Parse forward model matrix ---------------------------------------------------
        if G is None:
            _warnings.warn(
                f"No forward model was supplied. Defaulting to a unit matrix.", Warning
            )
            self.G = _numpy.eye(dimensions)
        elif type(G) == _numpy.ndarray or issubclass(type(G), _scipy.sparse.spmatrix):

            # Assert that the second dimension of the matrix corresponds to model space
            # dimension.
            assert G.shape[1] == dimensions

            self.G = G
        else:
            raise ValueError("The forward model matrix type was not understood.")

        # Parse data vector ------------------------------------------------------------
        if d is None:
            _warnings.warn(
                f"No data was supplied. Defaulting to a zero vector.", Warning
            )
            self.d = _numpy.ones((G.shape[0], 1))
        elif type(d) == _numpy.ndarray:

            # Assert that the data vector is compatible with the matrix.
            assert d.shape == (G.shape[0], 1)

            self.d = d
        else:
            raise ValueError("The data vector type was not understood.")

        # Parse data covariance --------------------------------------------------------
        if data_covariance is None:
            pass
        elif (
            type(data_covariance) is float
        ):  # TODO implement OR for all other possibilities
            self.data_covariance = data_covariance
        else:
            raise ValueError("The data covariance type was not understood.")

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        """
        """
        if type(self.data_covariance) is float:
            # Data covariance is a single scalar, so we move the data covariance
            # operation out of the matrix-vector products
            return (
                _numpy.linalg.norm((self.G @ coordinates - self.d), ord=2) ** 2
                / self.data_covariance
            )
        else:
            # TODO implement other cases
            raise NotImplementedError("This class is not production ready.")

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        """
        """
        if type(self.data_covariance) is float:
            # Data covariance is a single scalar, so we move the data covariance
            # operation out of the matrix-vector products
            return self.G.T @ (self.G @ coordinates - self.d) / self.data_covariance
        else:
            raise NotImplementedError("This class is not production ready.")


class Himmelblau(_AbstractTarget):
    """Himmelblau's 2-dimensional function.

    Himmelblau's function is defined as:

    .. math::

        f(x,y)=(x^{2}+y-11)^{2}+(x+y^{2}-7)^{2}
    """

    name: str = "Himmelblau's function"
    dimensions: int = 2
    annealing: float = 1
    """Float representing the annealing (:math:`T`) of Himmelblau's function.
    
    Alters the misfit function in the following way:

    .. math::

        f(x,y)_T=\\frac{f(x,y)}{T}
    """

    def __init__(self, dimensions: int = -1, annealing: float = 1):
        self.annealing = annealing

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        """Returns the value of Himmelblau's function at the given coordinates."""
        if coordinates.shape != (self.dimensions, 1):
            raise ValueError()
        x = coordinates[0, 0]
        y = coordinates[1, 0]
        return ((x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2) / self.annealing

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        """Returns a numpy.ndarray shaped as (dimensions, 1) containing the gradient of
        Himmelblau's function at the given coordinates."""
        x = coordinates[0]
        y = coordinates[1]
        gradient = _numpy.zeros((self.dimensions, 1))
        gradient[0] = 2 * (2 * x * (x ** 2 + y - 11) + x + y ** 2 - 7)
        gradient[1] = 2 * (x ** 2 + 2 * y * (x + y ** 2 - 7) + y - 11)
        return gradient / self.annealing


class Empty(_AbstractTarget):
    """Null target function.


    Has zero misfit and gradient for all parameters
    everywhere. Defined as:

    .. math::

        f(\\mathbf{m})=0


    """

    def __init__(self, dimensions: int):
        self.name = "empty target"
        self.dimensions = dimensions

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        """Returns zero for all arguments."""
        return 0.0

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        """Returns a vector of zeros for all arguments."""
        return _numpy.zeros((self.dimensions, 1))
