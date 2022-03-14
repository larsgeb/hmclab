import numpy as _numpy
import jax as _jax
from functools import partial
from hmclab.Distributions import _AbstractDistribution
from hmclab.Distributions import Normal as _Normal


class TransformToLogSpace(_AbstractDistribution):
    def __init__(self, distribution: _AbstractDistribution, base: float = 10):

        assert issubclass(type(distribution), _AbstractDistribution)

        self.base: float = base
        self.dimensions: int = distribution.dimensions
        self.distribution: _AbstractDistribution = distribution

        self.grad_logdetjac = _jax.jit(_jax.grad(self._grad_logdetjac))

    def misfit(self, m):

        assert m.shape == (m.size, 1)

        _m = self.transform_forward(m)

        if _numpy.any(_numpy.isnan(_m)):
            return _numpy.inf

        return self.distribution.misfit(_m) - _numpy.log(
            _numpy.linalg.det(self.jacobian(m))
        )

    def gradient(self, m):

        assert m.shape == (m.size, 1)

        _m = self.transform_forward(m)

        J = self.jacobian(m)

        return (
            self.distribution.gradient(_m).T @ J
            - _numpy.asarray(self.grad_logdetjac(m)).T
        ).T

    @staticmethod
    def create_default(dimensions: int) -> "TransformToLogSpace":
        return TransformToLogSpace(_Normal.create_default(dimensions))

    def generate(self):
        pass

    def generate(self, repeat=1, rng=_numpy.random.default_rng()) -> _numpy.ndarray:
        return self.distribution.generate(repeat, rng)

    def transform_backward(self, m):
        assert m.shape[0] == self.dimensions
        return _numpy.power(self.base, m)

    def transform_forward(self, m):
        assert m.shape[0] == self.dimensions
        return _numpy.log(m) / _numpy.log(self.base)

    def jacobian(self, m):
        assert m.shape == (m.size, 1)
        return _numpy.diag((1.0 / m.flatten()) / _numpy.log(self.base))

    def _grad_logdetjac(self, m):
        return _jax.numpy.log(
            _jax.numpy.linalg.det(
                _jax.numpy.diag((1.0 / m.flatten()) / _jax.numpy.log(self.base))
            )
        )
