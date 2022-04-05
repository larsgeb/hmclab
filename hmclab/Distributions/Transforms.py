import numpy as _numpy
from hmclab.Distributions import _AbstractDistribution
from hmclab.Distributions import Normal as _Normal


class TransformToLogSpace(_AbstractDistribution):

    grad_logdetjac_jax = None

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

        return (
            self.distribution.misfit(_m)
            # - _numpy.log(_numpy.linalg.det(self.jacobian(m)))
            # For diagonal jacobians:
            - _numpy.sum(_numpy.log(_numpy.diag(self.jacobian(m))))
            + self.misfit_bounds(m)
        )

    def gradient(self, m):

        assert m.shape == (m.size, 1)

        _m = self.transform_forward(m)

        J = self.jacobian(m)

        return (
            self.distribution.gradient(_m).T @ J - self.manual_grad_logdetjac(m).T
        ).T + self.misfit_bounds(m)

    @staticmethod
    def create_default(dimensions: int) -> "TransformToLogSpace":
        return TransformToLogSpace(_Normal.create_default(dimensions))

    def generate(self, repeat=1, rng=_numpy.random.default_rng()) -> _numpy.ndarray:
        return self.transform_backward(self.distribution.generate(repeat, rng))

    def transform_backward(self, m):
        assert m.shape[0] == self.dimensions
        return _numpy.power(self.base, m)

    def transform_forward(self, m):
        assert m.shape[0] == self.dimensions
        return _numpy.log(m) / _numpy.log(self.base)

    def jacobian(self, m):
        assert m.shape == (m.size, 1)
        return _numpy.diag((1.0 / m.flatten()) / _numpy.log(self.base))

    def inv_jacobian(self, m):
        assert m.shape == (m.size, 1)
        return _numpy.diag(m.flatten() * _numpy.log(self.base))

    def hessian(self, m):
        assert m.shape == (m.size, 1)
        return _numpy.diag((-1.0 / m.flatten() ** 2) / _numpy.log(self.base))

    def manual_grad_logdetjac(self, m):

        # This is computed using Jacobi's rule.
        # Additionally, taking the matrix inverse using NumPy is unstable, making it
        # prudent to find the inverse jacobian analytically.
        # Formula: d/dm log det jac = tr ( jac^-1 @ (d jac / dm) )
        # The term d jac / dm is taken to be the Hessian. Note that the tensor order
        # of the Hessian should actually be 3, but with a diagonal transformation, the
        # cross terms are not important, so we get away with simpler expressions.

        return (_numpy.diag((self.inv_jacobian(m)) @ self.hessian(m)))[:, None]

    """
    The following are two functions that don't require one to find analytical
    expressions of the gradient of the logarithm of the determinant of the Jacobian.
    This gradient is computed using automatic differentiation implemented in JAX. Note
    That JAX doesn't play well with the parallel HMC sampler, leading to many a bug.

    def _gradient_using_jax(self, m):

        import jax as _jax

        if self.grad_logdetjac_jax is None:
            self.grad_logdetjac_jax = _jax.jit(_jax.grad(self._jax_logdetjac))

        assert m.shape == (m.size, 1)

        _m = self.transform_forward(m)

        J = self.jacobian(m)

        return (
            self.distribution.gradient(_m).T @ J
            - _numpy.asarray(self.grad_logdetjac(m)).T
        ).T + self.misfit_bounds(m)

    def _jax_logdetjac(self, m):

        return _jax.numpy.log(
            _jax.numpy.linalg.det(
                _jax.numpy.diag((1.0 / m.flatten()) / _jax.numpy.log(self.base))
            )
        )
    """
