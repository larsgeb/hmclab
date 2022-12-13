"""Mass matrix classes and associated methods.

The classes in this module describe the metric used in HMC algorithms. Changing the
metric alters the shape of trajectories the HMC algorithm generates, thereby impacting
convergence performance.

All of the classes inherit from :class:`._AbstractMassMatrix`; a base class outlining
required methods and their signatures (required in- and outputs).

.. note::

    The mass matrix is vitally important for the performance of HMC algorithms A
    tutorial on the tuning parameters of HMC can be found at
    :ref:`/notebooks/tutorials/1 - Tuning Hamiltonian Monte Carlo.ipynb`.

"""
import warnings as _warnings
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
from scipy.linalg import cho_factor as _cho_factor, cho_solve as _cho_solve
import numpy as _numpy

from hmclab.Helpers.CustomExceptions import AbstractMethodError as _AbstractMethodError
from hmclab.Helpers.CustomExceptions import Assertions


class _AbstractMassMatrix(_ABC):
    """Abstract base class for mass matrices.

    Defines all required methods for derived classes.

    """

    name: str = "mass matrix abstract base class"
    dimensions: int = -1
    rng: _numpy.random.Generator = _numpy.random.default_rng()

    def full_name(self) -> str:
        raise _AbstractMethodError()

    @_abstractmethod
    def kinetic_energy(self, momentum: _numpy.ndarray) -> float:
        """Abstract method for computing kinetic energy for a given momentum.

        Parameters
        ----------
        momentum
        """
        raise _AbstractMethodError()

    @_abstractmethod
    def kinetic_energy_gradient(
        self,
        momentum: _numpy.ndarray,
        position: _numpy.ndarray = None,
        g: _numpy.ndarray = None,
    ) -> _numpy.ndarray:
        """Abstract method for computing kinetic energy gradient for a given
        momentum.

        Parameters
        ----------
        momentum
        """
        raise _AbstractMethodError()

    @_abstractmethod
    def generate_momentum(self) -> _numpy.ndarray:
        raise _AbstractMethodError()

    @staticmethod
    def create_default(dimensions: int) -> "_AbstractMassMatrix":
        raise _AbstractMethodError()

    def accept(self):
        pass

    def reject(self):
        pass


class Unit(_AbstractMassMatrix):
    """The unit mass matrix.

    This mass matrix or metric does not perform any scaling on the target distribution.
    It is the default setting for the HMC algorithms and is optimal when all parameters
    of your target distribution are expected to have the same variance and no
    trade-offs.

    """

    def __init__(self, dimensions: int = -1, rng: _numpy.random.Generator = None):
        """Constructor for unit mass matrices"""
        self.name = "unit mass matrix"
        self.dimensions = dimensions

        if rng is not None:
            self.rng = rng

    def kinetic_energy(self, momentum: _numpy.ndarray) -> float:
        """

        Parameters
        ----------
        momentum

        Returns
        -------

        """
        if momentum.shape != (self.dimensions, 1):
            raise ValueError()
        return 0.5 * (momentum.T @ momentum).item(0)

    def kinetic_energy_gradient(
        self,
        momentum: _numpy.ndarray,
        position: _numpy.ndarray = None,
        g: _numpy.ndarray = None,
    ) -> _numpy.ndarray:
        """

        Parameters
        ----------
        momentum

        Returns
        -------

        """
        if momentum.shape != (self.dimensions, 1):
            raise ValueError()
        return momentum

    def generate_momentum(self) -> _numpy.ndarray:
        """

        Returns
        -------

        """
        return self.rng.normal(size=(self.dimensions, 1))

    @property
    def matrix(self) -> _numpy.ndarray:
        """

        Returns
        -------

        """
        return _numpy.eye(self.dimensions)

    @staticmethod
    def create_default(dimensions: int, rng: _numpy.random.Generator = None) -> "Unit":
        return Unit(dimensions, rng)


class Diagonal(_AbstractMassMatrix):
    """The diagonal mass matrix.

    This mass matrix or metric does only performs scaling on each dimension separately.
    It is optimal when all parameters of your target distribution are expected to be
    independent (not have trade-offs) but still varying scales of disperion / variance.

    """

    def __init__(self, diagonal: _numpy.ndarray, rng: _numpy.random.Generator = None):
        """Constructor for diagonal mass matrices."""
        self.name = "diagonal mass matrix"

        diagonal = _numpy.asarray(diagonal)

        diagonal.shape = (diagonal.size, 1)

        assert diagonal.shape == (diagonal.size, 1)

        self.dimensions = diagonal.size
        self.diagonal = diagonal
        self.inverse_diagonal = 1.0 / self.diagonal

        if rng is not None:
            self.rng = rng

    def kinetic_energy(self, momentum: _numpy.ndarray) -> float:
        """

        Parameters
        ----------
        momentum

        Returns
        -------

        """
        if momentum.shape != (self.dimensions, 1):
            raise ValueError()
        return 0.5 * _numpy.vdot(momentum, self.inverse_diagonal * momentum)

    def kinetic_energy_gradient(
        self,
        momentum: _numpy.ndarray,
        position: _numpy.ndarray = None,
        g: _numpy.ndarray = None,
    ) -> _numpy.ndarray:
        """

        Parameters
        ----------
        momentum

        Returns
        -------

        """
        if momentum.shape != (self.dimensions, 1):
            raise ValueError()
        return self.inverse_diagonal * momentum

    def generate_momentum(self) -> _numpy.ndarray:
        """

        Returns
        -------

        """
        return _numpy.sqrt(self.diagonal) * self.rng.normal(size=(self.dimensions, 1))

    @property
    def matrix(self) -> _numpy.ndarray:
        return _numpy.diagflat(self.diagonal)

    @staticmethod
    def create_default(
        dimensions: int, rng: _numpy.random.Generator = None
    ) -> "Diagonal":
        diagonal = _numpy.ones((dimensions, 1))
        return Diagonal(diagonal, rng=rng)


class Full(_AbstractMassMatrix):
    """The full mass matrix.

    This mass matrix or metric performs scaling on all dimensions, as well as rotations.
    It is ideal when it is know a-priori that parameters might be heavily correlated.

    """

    def __init__(
        self,
        full: _numpy.ndarray,
        rng: _numpy.random.Generator = None,
        do_hermitian_check=True,
    ):
        """Constructor for diagonal mass matrices."""
        self.name = "diagonal mass matrix"

        self.mass_matrix = _numpy.asarray(full)

        if do_hermitian_check:
            assert _numpy.allclose(self.mass_matrix, self.mass_matrix.T)

        self.cholesky, self.cholesky_lower = _cho_factor(self.mass_matrix, lower=True)
        self.cholesky = _numpy.tril(self.cholesky)
        self.dimensions = self.mass_matrix.shape[0]

        if rng is not None:
            self.rng = rng

    def kinetic_energy(self, momentum: _numpy.ndarray) -> float:
        """

        Parameters
        ----------
        momentum

        Returns
        -------

        """
        if momentum.shape != (self.dimensions, 1):
            raise ValueError()
        return 0.5 * _numpy.vdot(
            momentum, _cho_solve((self.cholesky, self.cholesky_lower), momentum)
        )

    def kinetic_energy_gradient(
        self,
        momentum: _numpy.ndarray,
        position: _numpy.ndarray = None,
        g: _numpy.ndarray = None,
    ) -> _numpy.ndarray:
        """

        Parameters
        ----------
        momentum

        Returns
        -------

        """
        if momentum.shape != (self.dimensions, 1):
            raise ValueError()
        return _cho_solve((self.cholesky, self.cholesky_lower), momentum)

    def generate_momentum(self, repeat=1) -> _numpy.ndarray:
        """

        Returns
        -------

        """
        return self.cholesky @ self.rng.normal(size=(self.dimensions, repeat))

    @property
    def matrix(self) -> _numpy.ndarray:
        return self.mass_matrix

    @staticmethod
    def create_default(dimensions: int, rng: _numpy.random.Generator = None) -> "Full":
        mass_matrix = (
            _numpy.eye(dimensions)
            + 0.1 * _numpy.eye(dimensions, k=-1)
            + 0.1 * _numpy.eye(dimensions, k=1)
        )
        return Full(mass_matrix, rng=rng)


class BFGS(_AbstractMassMatrix):
    ms = []
    gs = []
    succesful_updates = 0
    attempted_updates = 0

    succesful_updates_current = 0
    attempted_updates_current = 0

    def __init__(
        self,
        dimensions: int,
        m: _numpy.ndarray,
        g: _numpy.ndarray,
        Minv: _numpy.ndarray = None,
        greedy: bool = False,
        rng: _numpy.random.Generator = None,
    ):
        """
        Initialise the BFGS iteration.

        :param dimensions: number of model-space dimensions
        :param Minv: initial mass matrix inverse
        :param m: current model vector
        :param g: current gradient

        The matrix Minv plays the role of the inverse mass matrix, which ideally is the inverse Hessian, i.e., the covariance matrix.
        """

        self.greedy = greedy

        if rng is not None:
            self.rng = rng

        self.dimensions = dimensions

        assert m.shape == (self.dimensions, 1), f"{m.shape}"
        assert g.shape == (self.dimensions, 1), f"{g.shape}"

        if Minv is None:
            self.Minv = _numpy.eye(dimensions)
        else:
            self.Minv = Minv

        LT = _numpy.linalg.cholesky(self.Minv).transpose()
        self.LTinv = _numpy.linalg.inv(LT)

        self.m = m
        self.g = g

        self.save()

    def kinetic_energy(self, momentum: _numpy.ndarray) -> float:
        if momentum.shape != (self.dimensions, 1):
            raise ValueError()

        return 0.5 * (momentum.T @ self.Minv @ momentum).item()

    def kinetic_energy_gradient(
        self,
        momentum: _numpy.ndarray,
        position: _numpy.ndarray = None,
        g: _numpy.ndarray = None,
    ) -> _numpy.ndarray:

        if momentum.shape != (self.dimensions, 1):
            raise ValueError()

        if position is not None and g is not None:
            self._update(position, g)

        return self.Minv @ momentum

    def accept(self):

        self.consolidate_updates()
        self.save()

        self.attempted_updates += self.attempted_updates_current
        self.succesful_updates += self.succesful_updates_current

        self.attempted_updates_current = 0
        self.succesful_updates_current = 0

    def freeze(self, cast=False):

        if cast:
            return Full(_numpy.linalg.inv(self.Minv), rng=self.rng)

        def nothing(*args, **kwargs):
            pass

        self.accept = nothing
        self.reject = nothing
        self.update = nothing
        self._update = nothing

    def reject(self):

        self.reset()

        self.attempted_updates_current = 0
        self.succesful_updates_current = 0

    def save(self):
        self.backup = self.Minv.copy(), self.m.copy(), self.g.copy()

    def reset(self):
        self.Minv, self.m, self.g = self.backup
        self.ms, self.gs = [], []

    def update(self, m, g):

        self.ms.append(m)
        self.gs.append(g)

        if self.greedy:
            self.consolidate_updates()

    def consolidate_updates(self):

        for m, g in zip(self.ms, self.gs):
            self._update(m, g)

        ms = []
        gs = []

    def _update(self, m, g):
        """
        Update BFGS matrix and perform Cholesky decomposition.

        :param m: current model vector
        :param g: current gradient
        """

        backup = self.Minv.copy(), self.m.copy(), self.g.copy()

        # Compute differences and update vectors.
        s = m - self.m
        y = g - self.g

        self.m = m
        self.g = g

        # Compute update of BFGS matrix.
        if (s.T @ y) > 0.0:
            rho = 1.0 / (s.T @ y)
            I = _numpy.identity(self.dimensions)
            sy = rho * (s @ y.T)

            l = I - sy
            r = I - sy.T

            ss = rho * (s @ s.T)

            self.Minv = ((l @ self.Minv) @ r) + ss

        # Compute Cholesky decomposition.

        self.attempted_updates_current += 1
        try:
            LT = _numpy.linalg.cholesky(self.Minv).transpose()
            self.LTinv = _numpy.linalg.inv(LT)
            self.succesful_updates_current += 1
        except _numpy.linalg.LinAlgError as e:
            print(e)
            print("Relevant quantities (s.T @ y, m):")
            print(s.T @ y, m)
            self.Minv, self.m, self.g = backup

    def generate_momentum(self, repeat=1) -> _numpy.ndarray:

        momentum = self.rng.normal(size=(self.dimensions, repeat))
        momentum = self.LTinv.dot(momentum)

        assert momentum.shape == (self.dimensions, repeat)

        return momentum

    @property
    def matrix(self):
        return _numpy.linalg.inv(self.Minv)

    @staticmethod
    def create_default(dimensions: int, rng: _numpy.random.Generator = None) -> "Full":
        minv = _numpy.eye(dimensions)
        return BFGS(
            dimensions,
            _numpy.zeros((dimensions, 1)),
            _numpy.ones((dimensions, 1)),
            minv,
            rng=rng,
        )


"""
class LBFGS(_AbstractMassMatrix):
    ""The experimental adaptive LBFGS mass matrix.


    UNTESTED


    .. warning::

        This mass matrix is not guaranteed to produce valid Markov chains.

    ""

    def __init__(
        self,
        dimensions: int,
        number_of_vectors: int = 10,
        starting_position: _numpy.ndarray = None,
        starting_gradient: _numpy.ndarray = None,
        max_determinant_change: float = 0.1,
        update_interval: int = 1,
        rng: _numpy.random.Generator = None,
    ):
        ""Constructor for LBFGS-style mass matrices.""

        if starting_position is None or starting_gradient is None:
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

        if rng is not None:
            self.rng = rng

    def kinetic_energy(self, momentum: _numpy.ndarray) -> float:
        if momentum.shape != (self.dimensions, 1):
            raise ValueError()
        return 0.5 * _numpy.vdot(momentum, self.Hinv(momentum))

    def kinetic_energy_gradient(
        self,
        momentum: _numpy.ndarray,
        position: _numpy.ndarray = None,
        g: _numpy.ndarray = None,
    ) -> _numpy.ndarray:
        if momentum.shape != (self.dimensions, 1):
            raise ValueError()
        return self.Hinv(momentum)
    def generate_momentum(self) -> _numpy.ndarray:
        return self.S(self.rng.normal(size=(self.dimensions, 1)))

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

        print(rho)

        if rho > 0 and not _numpy.isnan(rho) and not _numpy.isinf(rho):

            self.current_position = m
            self.current_gradient = g

            Hinv_y = self.Hinv(y_update)

            gamma2 = rho**2 * _numpy.vdot(y_update, y_update) + rho

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
            pass
            # print(f"Not updating. Rho: {rho}")

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
            logdet += _numpy.log(alpha**2)

        return logdet

    @property
    def matrix(self):

        matrix = _numpy.empty((self.dimensions, self.dimensions))

        return matrix

    @staticmethod
    def create_default(dimensions: int, rng: _numpy.random.Generator = None) -> "LBFGS":
        return LBFGS(dimensions, rng=rng)
"""
