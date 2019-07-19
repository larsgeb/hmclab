from abc import ABC, abstractmethod
import numpy


class MassMatrix(ABC):
    """Abstract base class for mass matrix. Defines all required methods for
    derived classes.

    """

    name: str = "mass matrix abstract base class"
    dimensions: int = -1

    def full_name(self) -> str:
        return self.name

    @abstractmethod
    def kinetic_energy(self, momentum: numpy.ndarray) -> float:
        """Abstract method for computing kinetic energy for a given momentum.

        Parameters
        ----------
        momentum
        """
        float()

    @abstractmethod
    def kinetic_energy_gradient(self, momentum: numpy.ndarray) -> numpy.ndarray:
        """Abstract method for computing kinetic energy gradient for a given
        momentum.

        Parameters
        ----------
        momentum
        """
        return numpy.ndarray(())

    @abstractmethod
    def generate_momentum(self) -> numpy.ndarray:
        return numpy.ndarray(())


class Unit(MassMatrix):
    def __init__(self, dimensions: int):
        """Constructor for unit mass matrices

        """
        self.name = "unit mass matrix"
        self.dimensions = dimensions

    def kinetic_energy(self, momentum: numpy.ndarray) -> float:
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

    def kinetic_energy_gradient(self, momentum: numpy.ndarray) -> numpy.ndarray:
        """

        Parameters
        ----------
        momentum

        Returns
        -------

        """
        return momentum

    def generate_momentum(self) -> numpy.ndarray:
        """

        Returns
        -------

        """
        return numpy.random.randn(self.dimensions, 1)

    @property
    def matrix(self) -> numpy.ndarray:
        """

        Returns
        -------

        """
        return numpy.eye(self.dimensions)


class Diagonal(MassMatrix):
    def __init__(self, dimensions: int, diagonal: numpy.ndarray = None):
        """Constructor for diagonal mass matrices.

        """
        self.name = "diagonal mass matrix"
        self.dimensions = dimensions

        if diagonal is None:
            self.diagonal = numpy.ones((self.dimensions, 1))
        else:
            if diagonal.shape != (self.dimensions, 1):
                raise ValueError(
                    f"The passed diagonal vector is not of the right"
                    f"dimensions, which would be ({self.dimensions, 1})."
                )
            self.diagonal = diagonal
        self.inverse_diagonal = 1.0 / self.diagonal

    def kinetic_energy(self, momentum: numpy.ndarray) -> float:
        """

        Parameters
        ----------
        momentum

        Returns
        -------

        """
        return 0.5 * (momentum.T @ (self.inverse_diagonal * momentum)).item(0)

    def kinetic_energy_gradient(self, momentum: numpy.ndarray) -> numpy.ndarray:
        """

        Parameters
        ----------
        momentum

        Returns
        -------

        """
        return self.inverse_diagonal * momentum

    def generate_momentum(self) -> numpy.ndarray:
        """

        Returns
        -------

        """
        return numpy.sqrt(self.diagonal) * numpy.random.randn(
            self.dimensions, 1
        )

    @property
    def matrix(self) -> numpy.ndarray:
        return numpy.diagflat(self.diagonal)


class LBFGS(MassMatrix):
    def __init__(self, dimensions: int):
        """Constructor for LBFGS-style mass matrices.

        """
        self.name = "LBFGS-style mass matrix"
        self.dimensions = dimensions

    def kinetic_energy(self, momentum: numpy.ndarray) -> float:
        raise NotImplementedError("This function is not finished yet")

    def kinetic_energy_gradient(self, momentum: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError("This function is not finished yet")

    def generate_momentum(self) -> numpy.ndarray:
        raise NotImplementedError("This function is not finished yet")
