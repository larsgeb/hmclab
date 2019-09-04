from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
import numpy as _numpy


class _AbstractTarget(_ABC):
    """Abstract base class for inverse problem targets. Defines all required
    methods for derived classes.

    """

    name: str = "inverse problem target abstract base class"
    dimensions: int = -1

    def full_name(self):
        """

        Returns
        -------

        """
        return self.name

    @_abstractmethod
    def misfit(self, coordinates: _numpy.ndarray) -> float:
        """

        Parameters
        ----------
        coordinates
        """
        pass

    @_abstractmethod
    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        """

        Parameters
        ----------
        coordinates
        """
        pass


class Himmelblau(_AbstractTarget):

    name = "Himmelblau's function"
    dimensions = 2
    annealing = 1

    def __init__(self, dimensions: int = -1, annealing: float = 1):
        """

        Parameters
        ----------
        annealing
        """
        self.annealing = annealing

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        """

        Parameters
        ----------
        coordinates

        Returns
        -------

        """
        if coordinates.shape != (self.dimensions, 1):
            raise ValueError()
        x = coordinates[0, 0]
        y = coordinates[1, 0]
        return ((x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2) / self.annealing

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        """

        Parameters
        ----------
        coordinates

        Returns
        -------

        """
        x = coordinates[0]
        y = coordinates[1]
        gradient = _numpy.zeros((self.dimensions, 1))
        gradient[0] = 2 * (2 * x * (x ** 2 + y - 11) + x + y ** 2 - 7)
        gradient[1] = 2 * (x ** 2 + 2 * y * (x + y ** 2 - 7) + y - 11)
        return gradient / self.annealing


class Empty(_AbstractTarget):
    def __init__(self, dimensions: int):
        """

        Parameters
        ----------
        dimensions
        """
        self.name = "empty target"
        self.dimensions = dimensions

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        return 0.0

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:
        return _numpy.zeros((self.dimensions, 1))
