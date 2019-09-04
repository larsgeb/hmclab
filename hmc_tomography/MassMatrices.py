from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
import numpy as _numpy


class _AbstractMassMatrix(_ABC):
    """Abstract base class for mass matrix. Defines all required methods for
    derived classes.

    """

    name: str = "mass matrix abstract base class"
    dimensions: int = -1

    def full_name(self) -> str:
        return self.name

    @_abstractmethod
    def kinetic_energy(self, momentum: _numpy.ndarray) -> float:
        """Abstract method for computing kinetic energy for a given momentum.

        Parameters
        ----------
        momentum
        """
        float()

    @_abstractmethod
    def kinetic_energy_gradient(
        self, momentum: _numpy.ndarray
    ) -> _numpy.ndarray:
        """Abstract method for computing kinetic energy gradient for a given
        momentum.

        Parameters
        ----------
        momentum
        """
        return _numpy.ndarray(())

    @_abstractmethod
    def generate_momentum(self) -> _numpy.ndarray:
        return _numpy.ndarray(())


class Unit(_AbstractMassMatrix):
    def __init__(self, dimensions: int):
        """Constructor for unit mass matrices

        """
        self.name = "unit mass matrix"
        self.dimensions = dimensions

    def kinetic_energy(self, momentum: _numpy.ndarray) -> float:
        """

        Parameters
        ----------
        momentum

        Returns
        -------

        """
        if momentum.shape != (self.dimensions, 1):
            raise ValueError(
                f"The passed momentum vector is not of the right dimensions, "
                f"which would be ({self.dimensions, 1})."
            )
        return 0.5 * (momentum.T @ momentum).item(0)

    def kinetic_energy_gradient(
        self, momentum: _numpy.ndarray
    ) -> _numpy.ndarray:
        """

        Parameters
        ----------
        momentum

        Returns
        -------

        """
        return momentum

    def generate_momentum(self) -> _numpy.ndarray:
        """

        Returns
        -------

        """
        return _numpy.random.randn(self.dimensions, 1)

    @property
    def matrix(self) -> _numpy.ndarray:
        """

        Returns
        -------

        """
        return _numpy.eye(self.dimensions)


class Diagonal(_AbstractMassMatrix):
    def __init__(self, dimensions: int, diagonal: _numpy.ndarray = None):
        """Constructor for diagonal mass matrices.

        """
        self.name = "diagonal mass matrix"
        self.dimensions = dimensions

        if diagonal is None:
            self.diagonal = _numpy.ones((self.dimensions, 1))
        else:
            if diagonal.shape != (self.dimensions, 1):
                raise ValueError(
                    f"The passed diagonal vector is not of the right"
                    f"dimensions, which would be ({self.dimensions, 1})."
                )
            self.diagonal = diagonal
        self.inverse_diagonal = 1.0 / self.diagonal

    def kinetic_energy(self, momentum: _numpy.ndarray) -> float:
        """

        Parameters
        ----------
        momentum

        Returns
        -------

        """
        return 0.5 * (momentum.T @ (self.inverse_diagonal * momentum)).item(0)

    def kinetic_energy_gradient(
        self, momentum: _numpy.ndarray
    ) -> _numpy.ndarray:
        """

        Parameters
        ----------
        momentum

        Returns
        -------

        """
        return self.inverse_diagonal * momentum

    def generate_momentum(self) -> _numpy.ndarray:
        """

        Returns
        -------

        """
        return _numpy.sqrt(self.diagonal) * _numpy.random.randn(
            self.dimensions, 1
        )

    @property
    def matrix(self) -> _numpy.ndarray:
        return _numpy.diagflat(self.diagonal)


class LBFGS(_AbstractMassMatrix):
    def __init__(self, dimensions: int):
        """Constructor for LBFGS-style mass matrices.

        """
        self.name = "LBFGS-style mass matrix"
        self.dimensions = dimensions

    def kinetic_energy(self, momentum: _numpy.ndarray) -> float:
        raise NotImplementedError("This function is not finished yet")

    def kinetic_energy_gradient(
        self, momentum: _numpy.ndarray
    ) -> _numpy.ndarray:
        raise NotImplementedError("This function is not finished yet")

    def generate_momentum(self) -> _numpy.ndarray:
        raise NotImplementedError("This function is not finished yet")
