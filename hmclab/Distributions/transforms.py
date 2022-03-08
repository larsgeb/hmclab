from typing import Union as _Union, Tuple as _Tuple

import numpy as _numpy
import matplotlib.pyplot as _plt

from hmclab.Distributions import _AbstractDistribution
from hmclab.Distributions import Normal as _Normal
from hmclab.Helpers.CustomExceptions import InvalidCaseError as _InvalidCaseError
import math as _math


class TransformToLogSpace(_AbstractDistribution):
    def __init__(self, distribution: _AbstractDistribution, base: float = 10):

        assert issubclass(type(distribution), _AbstractDistribution)

        self.base: float = base
        self.dimensions: int = distribution.dimensions
        self.distribution: _AbstractDistribution = distribution

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
        H = self.hessian(m)

        return (
            self.distribution.gradient(_m).T @ J
            - ((_numpy.einsum("jk,ijk->i", J.T, H))[None, :])
        ).T

    @staticmethod
    def create_default(dimensions: int) -> "TransformToLogSpace":
        return TransformToLogSpace(_Normal.create_default(dimensions))

    def generate(self, repeat=1, rng=_numpy.random.default_rng()) -> _numpy.ndarray:
        return self.distribution.generate(repeat, rng)

    def transform_backward(self, m):

        assert m.shape == (m.size, 1)

        return _numpy.power(self.base, m)

    def transform_forward(self, m):

        assert m.shape == (m.size, 1)

        return _numpy.log(m) / _numpy.log(self.base)

    def jacobian(self, m):

        assert m.shape == (m.size, 1)

        return _numpy.diag((1.0 / m.flatten()) / _numpy.log(self.base))

    def hessian(self, m):

        H = _numpy.zeros((self.dimensions, self.dimensions, self.dimensions))

        _numpy.fill_diagonal(H, (1.0 / m.flatten() ** 2) / _numpy.log(self.base))

        return H

    def det_inverse_jacobian(self, m):
        return _numpy.abs(_numpy.prod(self.inverse_jacobian(m)))

    def inverse_jacobian(self, m):
        return _numpy.power(self.base, m) * _numpy.log(self.base)
