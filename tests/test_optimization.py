"""A collection of optimization tests.
"""
import pytest as _pytest, numpy as _numpy

import hmclab as _hmclab
from hmclab.Helpers.CustomExceptions import InvalidCaseError as _InvalidCaseError
from hmclab.Optimizers import gradient_descent as _gradient_descent

_ad = _hmclab.Distributions._AbstractDistribution
_ao = _hmclab.Optimizers._AbstractOptimizer

distribution_classes = _ad.__subclasses__()
optimizer_classes = _ao.__subclasses__()


@_pytest.mark.parametrize("optimizer_class", optimizer_classes)
@_pytest.mark.parametrize("distribution_class", distribution_classes)
@_pytest.mark.parametrize("dimensions", [1, 2, 5, 100])
@_pytest.mark.parametrize("iterations", [1, 100])
@_pytest.mark.parametrize("epsilon", [0.1])
@_pytest.mark.parametrize("strictly_monotonic", [True, False])
def test_basic_optimization(
    optimizer_class: _ao,
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
        return 0

    optimizer_instance = optimizer_class()

    assert isinstance(optimizer_instance, _ao)

    m, x, ms, xs = optimizer_instance.iterate(
        target=distribution,
        initial_model=None,
        iterations=iterations,
        epsilon=epsilon,
        strictly_monotonic=strictly_monotonic,
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
        return 0

    optimizer_instance = _gradient_descent()

    assert isinstance(optimizer_instance, _ao)

    m, x, ms, xs = optimizer_instance.iterate(
        target=distribution,
        initial_model=None,
        iterations=iterations,
        epsilon=epsilon,
        regularization=regularization,
        strictly_monotonic=strictly_monotonic,
    )

    assert m.shape == (distribution.dimensions, 1)
    for i_m in ms:
        assert m.shape == i_m.shape
    assert isinstance(x, (float, _numpy.floating))
    assert len(ms) == len(xs)
    if strictly_monotonic:
        assert xs[0] >= xs[-1]
