from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
import numpy as _numpy


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
