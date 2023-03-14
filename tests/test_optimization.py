"""A collection of optimization tests.
"""
from re import A
import pytest as _pytest, numpy as _numpy

import hmclab as _hmclab
from hmclab.Helpers.CustomExceptions import InvalidCaseError as _InvalidCaseError
from hmclab.Optimizers import gradient_descent as _gradient_descent

_ad = _hmclab.Distributions._AbstractDistribution

distribution_classes = _ad.__subclasses__()
optimizer_methods = [_gradient_descent]


@_pytest.mark.parametrize("optimizer_method", optimizer_methods)
@_pytest.mark.parametrize("distribution_class", distribution_classes)
@_pytest.mark.parametrize("dimensions", [1, 2, 5, 100])
@_pytest.mark.parametrize("iterations", [1, 100])
@_pytest.mark.parametrize("epsilon", [0.1])
@_pytest.mark.parametrize("strictly_monotonic", [True, False])
def test_basic_optimization(
    optimizer_method,
    distribution_class: _ad,
    dimensions: int,
    iterations: int,
    epsilon: float,
    strictly_monotonic: bool,
):
    """Test optimization algorithms in general"""

    try:
        distribution: _ad = distribution_class.create_default(dimensions)
    except _InvalidCaseError:
        return _pytest.skip("Invalid case")

    try:
        initial_model = distribution.generate()
    except:
        initial_model = _numpy.ones((distribution.dimensions, 1))

    m, x, ms, xs = optimizer_method(
        target=distribution,
        initial_model=initial_model,
        iterations=iterations,
        epsilon=epsilon,
        strictly_monotonic=strictly_monotonic,
        disable_progressbar=True,
    )

    assert len(ms) == len(xs)

    if strictly_monotonic:
        assert xs[0] >= xs[-1]


@_pytest.mark.parametrize("distribution_class", distribution_classes)
@_pytest.mark.parametrize("dimensions", [1, 2, 5, 100])
@_pytest.mark.parametrize("iterations", [1, 100])
@_pytest.mark.parametrize("epsilon", [0.1])
@_pytest.mark.parametrize("regularization", [1e-3, 1e0, 1e3])
@_pytest.mark.parametrize("strictly_monotonic", [True, False])
def test_gradient_descent(
    distribution_class: _ad,
    dimensions: int,
    iterations: int,
    regularization: float,
    epsilon: float,
    strictly_monotonic: bool,
):
    """Test all settings of the gradient descent algorithm"""

    try:
        distribution: _ad = distribution_class.create_default(dimensions)
    except _InvalidCaseError:
        return _pytest.skip("Invalid case")

    try:
        initial_model = distribution.generate()
    except:
        initial_model = _numpy.ones((distribution.dimensions, 1))

    m, x, ms, xs = _gradient_descent(
        target=distribution,
        initial_model=initial_model,
        iterations=iterations,
        epsilon=epsilon,
        regularization=regularization,
        strictly_monotonic=strictly_monotonic,
        disable_progressbar=True,
    )

    assert m.shape == (distribution.dimensions, 1)
    for i_m in ms:
        assert m.shape == i_m.shape
    assert isinstance(x, (float, _numpy.floating))
    assert len(ms) == len(xs)
    if strictly_monotonic:
        assert xs[0] >= xs[-1]
