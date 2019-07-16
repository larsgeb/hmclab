from abc import ABC, abstractmethod
import numpy


class MassMatrix(ABC):
    """Abstract base class for mass matrix. Defines all required methods for
    derived classes.

    """

    name = "mass matrix abstract base class"

    def full_name(self):
        return self.name

    @abstractmethod
    def kinetic_energy(self, momentum: numpy.ndarray):
        """Abstract method for computing kinetic energy for a given momentum.

        Parameters
        ----------
        momentum
        """
        pass

    @abstractmethod
    def generate_momentum(self) -> numpy.ndarray:
        """

        """
        pass


class Unit(MassMatrix):
    def __init__(self, dimensions):
        """Constructor for unit mass matrices

        """
        self.name = "unit mass matrix"
        self.dimensions = dimensions

    def kinetic_energy(self, momentum: numpy.ndarray):
        if momentum.shape != (self.dimensions, 1):
            raise ValueError(
                f"The passed momentum vector is not of the right dimensions, "
                f"which would be ({self.dimensions, 1})."
            )
        return 0.5 * numpy.linalg.norm(momentum)

    def generate_momentum(self) -> numpy.ndarray:
        return numpy.random.randn(self.dimensions, 1)


class Diagonal(MassMatrix):
    def __init__(self, dimensions, diagonal=None):
        """Constructor for diagonal mass matrices.

        """
        self.name = "diagonal mass matrix"
        self.dimensions = dimensions
        if diagonal.shape != (self.dimensions, 1):
            raise ValueError(
                f"The passed diagonal vector is not of the right dimensions, "
                f"which would be ({self.dimensions, 1})."
            )
        self.diagonal = diagonal

    def kinetic_energy(self, momentum: numpy.ndarray):
        pass

    def generate_momentum(self) -> numpy.ndarray:
        return numpy.sqrt(self.diagonal) * numpy.random.randn(self.dimensions, 1)


class LBFGS(MassMatrix):
    def __init__(self, dimensions):
        """Constructor for LBFGS-style mass matrices.

        """
        self.name = "LBFGS-style mass matrix"
        self.dimensions = dimensions

    def kinetic_energy(self, momentum: numpy.ndarray):
        pass

    def generate_momentum(self) -> numpy.ndarray:
        raise NotImplementedError("This function is not finished yet")
        # return numpy.random.randn(self.dimensions, 1)
