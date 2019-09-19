from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
import numpy as _numpy
from scipy.sparse.linalg import spsolve


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
        return 0.5 * _numpy.vdot(momentum, self.inverse_diagonal * momentum)

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
    def __init__(
        self,
        dimensions: int,
        number_of_vectors: int,
        starting_position: _numpy.ndarray,
        starting_gradient: _numpy.ndarray,
        max_determinant_change: float,
    ):
        """Constructor for LBFGS-style mass matrices.

        """
        self.name = "LBFGS-style mass matrix"
        self.dimensions = dimensions
        self.number_of_vectors = number_of_vectors
        self.current_number_of_gradients = 0

        self.current_position = starting_position
        self.current_gradient = starting_gradient

        self.s = _numpy.empty((dimensions, number_of_vectors))
        self.y = _numpy.empty((dimensions, number_of_vectors))
        self.u = _numpy.empty((dimensions, number_of_vectors))
        self.v = _numpy.empty((dimensions, number_of_vectors))
        self.vTu = _numpy.empty(number_of_vectors)

        self.max_determinant_change = max_determinant_change

    def kinetic_energy(self, momentum: _numpy.ndarray) -> float:
        return 0.5 * _numpy.vdot(momentum, self.Hinv(momentum))

    def kinetic_energy_gradient(
        self, momentum: _numpy.ndarray
    ) -> _numpy.ndarray:
        return self.Hinv(momentum)

    def generate_momentum(self) -> _numpy.ndarray:
        return self.S(_numpy.random.randn(self.dimensions, 1))

    def update(self, m, g):
        if self.current_number_of_gradients == self.number_of_vectors:
            return

        s = m - self.current_position
        y = g - self.current_gradient

        assert s.shape == (self.dimensions, 1)
        assert y.shape == (self.dimensions, 1)

        self.current_position = m
        self.current_gradient = g

        # Compute auxiliary stuff

        rho = 1.0 / _numpy.vdot(s, y)

        # Do nothing unless rho is positive.
        if rho > 0:

            Hinv_y = self.Hinv(y)
            gamma2 = rho ** 2 * _numpy.vdot(y, y) + rho
            beta = gamma2 * _numpy.vdot(s, self.H(s))
            theta = _numpy.sqrt(rho / (beta * gamma2))

            a = _numpy.sqrt(gamma2) * s
            b = (rho / _numpy.sqrt(gamma2)) * Hinv_y

            u = a
            v = -self.H(b + theta * a)

            assert u.shape == (self.dimensions, 1)
            assert v.shape == (self.dimensions, 1)

            sigma_threshold = (1.0 / (1.0 + _numpy.vdot(u, v))) ** 2

            if sigma_threshold < self.max_determinant_change:
                r = (1.0 - self.max_determinant_change) / (
                    self.max_determinant_change * _numpy.vdot(u, v)
                )
                v = r * v

            if self.current_number_of_gradients < self.number_of_vectors:
                self.current_number_of_gradients += 1
            else:
                # todo get rid of this abysmal thing
                self.s = _numpy.roll(self.s, -1, axis=1)
                self.y = _numpy.roll(self.y, -1, axis=1)
                self.u = _numpy.roll(self.u, -1, axis=1)
                self.v = _numpy.roll(self.v, -1, axis=1)
                self.vTu = _numpy.roll(self.vTu, -1)

            assert s.shape == (self.dimensions, 1)
            assert y.shape == (self.dimensions, 1)
            assert u.shape == (self.dimensions, 1)
            assert v.shape == (self.dimensions, 1)

            self.s[:, self.current_number_of_gradients - 1] = s.flatten()
            self.y[:, self.current_number_of_gradients - 1] = y.flatten()
            self.u[:, self.current_number_of_gradients - 1] = u.flatten()
            self.v[:, self.current_number_of_gradients - 1] = v.flatten()
            self.vTu[self.current_number_of_gradients - 1] = 1.0 + _numpy.vdot(
                v, u
            )

    def S(self, h):

        for i in range(self.current_number_of_gradients):
            h = (
                h
                - self.v[:, i, None]
                * _numpy.vdot(self.u[:, i, None], h)
                / self.vTu[i]
            )
        assert h.shape == (self.dimensions, 1)
        return h

    def ST(self, h):

        for i in range(self.current_number_of_gradients - 1, -1, -1):
            h = (
                h
                - self.u[:, i, None]
                * _numpy.vdot(self.v[:, i, None], h)
                / self.vTu[i]
            )
        assert h.shape == (self.dimensions, 1)
        return h

    def Sinv(self, h):

        for i in range(self.current_number_of_gradients - 1, -1, -1):
            h = h + self.v[:, i, None] * _numpy.vdot(self.u[:, i, None], h)
        assert h.shape == (self.dimensions, 1)
        return h

    def SinvT(self, h):

        for i in range(self.current_number_of_gradients):
            h = h + self.u[:, i, None] * _numpy.vdot(self.v[:, i, None], h)
        assert h.shape == (self.dimensions, 1)
        return h

    def H(self, h):

        h = self.ST(h)
        assert h.shape == (self.dimensions, 1)
        return self.S(h)

    def Hinv(self, h):

        h = self.Sinv(h)
        assert h.shape == (self.dimensions, 1)
        return self.SinvT(h)

    def logdet(self):

        logdet = 0.0
        for i in range(self.current_number_of_gradients):
            alpha = 1.0 / (1.0 + _numpy.dot(self.u[:, i], self.v[:, i]))
            logdet += _numpy.log(alpha ** 2)

        return logdet


class SparseDecomposed(_AbstractMassMatrix):
    def __init__(self, decomposition):
        self.decomposition = decomposition
        self.dimensions = self.decomposition.shape[0]

    def kinetic_energy(self, momentum: _numpy.ndarray) -> float:
        return (
            0.5 * _numpy.linalg.norm(spsolve(self.decomposition, momentum)) ** 2
        ).item()

    def kinetic_energy_gradient(
        self, momentum: _numpy.ndarray
    ) -> _numpy.ndarray:
        return spsolve(self.decomposition, momentum)[:, _numpy.newaxis]

    def generate_momentum(self) -> _numpy.ndarray:
        return (
            self.decomposition
            @ _numpy.random.randn(self.decomposition.shape[0])[
                :, _numpy.newaxis
            ]
        )
