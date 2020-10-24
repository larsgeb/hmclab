"""Mass matrix classes and associated methods.

The classes in this module describe the metric used in HMC algorithms. Changing the
metric alters the shape of trajectories the HMC algorithm generates, thereby impacting
convergence performance.

All of the classes inherit from :class:`._AbstractMassMatrix`; a base class outlining
required methods and their signatures (required in- and outputs).

.. note::

    The mass matrix is vitally important for the performance of HMC algorithms A
    tutorial on the tuning parameters of HMC can be found at
    :ref:`/examples/0.2 - Tuning Hamiltonian Monte Carlo.ipynb`.

"""
import warnings as _warnings
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod

import numpy as _numpy

from hmc_tomography.Helpers.CustomExceptions import (
    AbstractMethodError as _AbstractMethodError,
)

from hmc_tomography.Helpers.CustomExceptions import Assertions


class _AbstractMassMatrix(_ABC):
    """Abstract base class for mass matrices.

    Defines all required methods for derived classes.

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
    def kinetic_energy_gradient(self, momentum: _numpy.ndarray) -> _numpy.ndarray:
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

    @staticmethod
    def create_default(dimensions: int) -> "_AbstractMassMatrix":
        raise _AbstractMethodError()


class Unit(_AbstractMassMatrix):
    """The unit mass matrix.

    This mass matrix or metric does not perform any scaling on the target distribution.
    It is the default setting for the HMC algorithms and is optimal when all parameters
    of your target distribution are expected to have the same variance and no
    trade-offs.

    """

    def __init__(self, dimensions: int = -1):
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
        if momentum.shape != (momentum.size, 1):
            raise ValueError(
                f"The passed momentum vector is not of the right dimensions, "
                f"which would be ({momentum.size, 1})."
            )
        return 0.5 * (momentum.T @ momentum).item(0)

    def kinetic_energy_gradient(self, momentum: _numpy.ndarray) -> _numpy.ndarray:
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

    @staticmethod
    def create_default(dimensions: int) -> "Unit":
        return Unit(dimensions)


class Diagonal(_AbstractMassMatrix):
    """The diagonal mass matrix.

    This mass matrix or metric does only performs scaling on each dimension separately.
    It is optimal when all parameters of your target distribution are expected to be
    independent (not have trade-offs) but still varying scales of disperion / variance.

    """

    def __init__(self, diagonal: _numpy.ndarray):
        """Constructor for diagonal mass matrices.

        """
        self.name = "diagonal mass matrix"

        diagonal = _numpy.asarray(diagonal)

        if diagonal is None or type(diagonal) != _numpy.ndarray:
            raise ValueError("The diagonal mass matrix did not receive a diagonal")

        diagonal.shape = (diagonal.size, 1)

        assert diagonal.shape == (diagonal.size, 1)

        self.dimensions = diagonal.size
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

    def kinetic_energy_gradient(self, momentum: _numpy.ndarray) -> _numpy.ndarray:
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
        return _numpy.sqrt(self.diagonal) * _numpy.random.randn(self.dimensions, 1)

    @property
    def matrix(self) -> _numpy.ndarray:
        return _numpy.diagflat(self.diagonal)

    @staticmethod
    def create_default(dimensions: int) -> "Diagonal":
        diagonal = _numpy.ones((dimensions, 1))
        return Diagonal(diagonal)


class LBFGS(_AbstractMassMatrix):
    """The experimental adaptive LBFGS mass matrix.

    .. warning::

        This mass matrix is not guaranteed to produce valid Markov chains.

    """

    def __init__(
        self,
        dimensions: int,
        number_of_vectors: int = 10,
        starting_position: _numpy.ndarray = None,
        starting_gradient: _numpy.ndarray = None,
        max_determinant_change: float = 0.1,
        update_interval: int = 1,
    ):
        """Constructor for LBFGS-style mass matrices.

        """

        if starting_position is None or starting_gradient is None:
            _warnings.warn(
                f"The LBFGS-style mass matrix did either not receive a starting "
                f"coordinate or a starting gradient. We will use a unit initial "
                f"point (m=1) and gradient (g=1).",
                Warning,
            )
            starting_gradient = _numpy.ones((dimensions, 1))
            starting_position = _numpy.ones((dimensions, 1))

        self.name = "LBFGS-style mass matrix"
        self.dimensions = dimensions
        self.number_of_vectors = number_of_vectors
        self.currently_stored_gradients = 0

        self.current_position = starting_position
        self.current_gradient = starting_gradient

        self.update_attempt = 0
        self.update_interval = update_interval

        self.s = _numpy.empty((dimensions, number_of_vectors))
        self.y = _numpy.empty((dimensions, number_of_vectors))
        self.u = _numpy.empty((dimensions, number_of_vectors))
        self.v = _numpy.empty((dimensions, number_of_vectors))
        self.vTu = _numpy.empty(number_of_vectors)

        self.max_determinant_change = max_determinant_change

    def kinetic_energy(self, momentum: _numpy.ndarray) -> float:
        return 0.5 * _numpy.vdot(momentum, self.Hinv(momentum))

    def kinetic_energy_gradient(self, momentum: _numpy.ndarray) -> _numpy.ndarray:
        return self.Hinv(momentum)

    def generate_momentum(self) -> _numpy.ndarray:
        return self.S(_numpy.random.randn(self.dimensions, 1))

    def update(self, m, g):

        self.update_attempt += 1

        # if (
        #     self.currently_stored_gradients == self.number_of_vectors
        #     or self.update_attempt % self.update_interval
        # ):
        #     # Do nothing if we shouldn't do anything
        #     return

        # Calculate separation and gradient change
        s_update = m - self.current_position
        y_update = g - self.current_gradient

        assert s_update.shape == (self.dimensions, 1), Assertions.v_shape
        assert y_update.shape == (self.dimensions, 1), Assertions.v_shape

        rho = 1.0 / _numpy.vdot(s_update, y_update)
        # Do nothing unless rho is positive.

        if rho > 0 and not _numpy.isnan(rho) and not _numpy.isinf(rho):

            self.current_position = m
            self.current_gradient = g

            Hinv_y = self.Hinv(y_update)

            gamma2 = rho ** 2 * _numpy.vdot(y_update, y_update) + rho

            beta = gamma2 * _numpy.vdot(s_update, self.H(s_update))
            theta = _numpy.sqrt(rho / (beta * gamma2))

            a = _numpy.sqrt(gamma2) * s_update
            b = (rho / _numpy.sqrt(gamma2)) * Hinv_y

            u_update = a
            v_update = -self.H(b + theta * a)

            assert u_update.shape == (self.dimensions, 1), Assertions.v_shape
            assert v_update.shape == (self.dimensions, 1), Assertions.v_shape

            sigma_threshold = (1.0 / (1.0 + _numpy.vdot(u_update, v_update))) ** 2

            if sigma_threshold < self.max_determinant_change:
                r = (1.0 - self.max_determinant_change) / (
                    self.max_determinant_change * _numpy.vdot(u_update, v_update)
                )
                v_update = r * v_update

            if self.currently_stored_gradients < self.number_of_vectors:
                self.currently_stored_gradients += 1
            else:
                # self.s[:, 1:-1] = self.s[:, 2:]
                # self.y[:, 1:-1] = self.y[:, 2:]
                # self.u[:, 1:-1] = self.u[:, 2:]
                # self.v[:, 1:-1] = self.v[:, 2:]
                # self.vTu[1:-1] = self.vTu[2:]

                # Next two blocks are equivalent

                # 1
                # self.s[:, :-1] = self.s[:, 1:]
                # self.y[:, :-1] = self.y[:, 1:]
                # self.u[:, :-1] = self.u[:, 1:]
                # self.v[:, :-1] = self.v[:, 1:]
                # self.vTu[:-1] = self.vTu[1:]

                # 2
                self.s = _numpy.roll(self.s, -1, axis=1)
                self.y = _numpy.roll(self.y, -1, axis=1)
                self.u = _numpy.roll(self.u, -1, axis=1)
                self.v = _numpy.roll(self.v, -1, axis=1)
                self.vTu = _numpy.roll(self.vTu, -1, axis=0)

            assert s_update.shape == (self.dimensions, 1), Assertions.v_shape
            assert y_update.shape == (self.dimensions, 1), Assertions.v_shape
            assert u_update.shape == (self.dimensions, 1), Assertions.v_shape
            assert v_update.shape == (self.dimensions, 1), Assertions.v_shape

            self.s[:, self.currently_stored_gradients - 1] = s_update.flatten()
            self.y[:, self.currently_stored_gradients - 1] = y_update.flatten()
            self.u[:, self.currently_stored_gradients - 1] = u_update.flatten()
            self.v[:, self.currently_stored_gradients - 1] = v_update.flatten()
            self.vTu[self.currently_stored_gradients - 1] = 1.0 + _numpy.vdot(
                v_update, u_update
            )
        else:
            print(f"Not updating. Rho: {rho}")

    def S(self, h):

        for i in range(self.currently_stored_gradients):
            h = (
                h
                - self.v[:, i, None] * _numpy.vdot(self.u[:, i, None], h) / self.vTu[i]
            )
        assert h.shape == (self.dimensions, 1), Assertions.v_shape
        return h

    def ST(self, h):

        for i in range(self.currently_stored_gradients - 1, -1, -1):
            h = (
                h
                - self.u[:, i, None] * _numpy.vdot(self.v[:, i, None], h) / self.vTu[i]
            )
        assert h.shape == (self.dimensions, 1), Assertions.v_shape
        return h

    def Sinv(self, h):

        for i in range(self.currently_stored_gradients - 1, -1, -1):
            h = h + self.v[:, i, None] * _numpy.vdot(self.u[:, i, None], h)
        assert h.shape == (self.dimensions, 1), Assertions.v_shape
        return h

    def SinvT(self, h):

        for i in range(self.currently_stored_gradients):
            h = h + self.u[:, i, None] * _numpy.vdot(self.v[:, i, None], h)
        assert h.shape == (self.dimensions, 1), Assertions.v_shape
        return h

    def H(self, h):

        h = self.ST(h)
        assert h.shape == (self.dimensions, 1), Assertions.v_shape
        return self.S(h)

    def Hinv(self, h):

        h = self.Sinv(h)
        assert h.shape == (self.dimensions, 1), Assertions.v_shape
        return self.SinvT(h)

    def logdet(self):

        logdet = 0.0
        for i in range(self.currently_stored_gradients):
            alpha = 1.0 / (1.0 + _numpy.dot(self.u[:, i], self.v[:, i]))
            logdet += _numpy.log(alpha ** 2)

        return logdet

    @staticmethod
    def create_default(dimensions: int) -> "LBFGS":
        return LBFGS(dimensions)
