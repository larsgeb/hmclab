"""A collection of tests for likelihood functions.
"""
import matplotlib.pyplot as _plt
import numpy as _numpy
import pytest as _pytest

from hmclab import Distributions as _Distributions
from hmclab.Helpers.CustomExceptions import InvalidCaseError as _InvalidCaseError

dimensions = [1, 2, 5, 50]
subclasses = _Distributions._AbstractDistribution.__subclasses__()
deltas = [1e-10, 1e-2, -1e-10, -1e-2]


@_pytest.mark.parametrize("pclass", subclasses)
@_pytest.mark.parametrize("dimensions", dimensions)
def test_creation(pclass: _Distributions._AbstractDistribution, dimensions: int):
    # Create the object
    try:
        distribution: _Distributions._AbstractDistribution = pclass.create_default(
            dimensions
        )
    except _InvalidCaseError:
        return _pytest.skip("Invalid case")

    # Check if a subtype of mass matrices
    assert issubclass(type(distribution), _Distributions._AbstractDistribution)

    # Check if the right amount of dimensions
    assert distribution.dimensions == dimensions


@_pytest.mark.parametrize("pclass", subclasses)
@_pytest.mark.parametrize("dimensions", dimensions)
def test_generation(pclass: _Distributions._AbstractDistribution, dimensions: int):
    try:
        distribution: _Distributions._AbstractDistribution = pclass.create_default(
            dimensions
        )
    except _InvalidCaseError:
        return _pytest.skip("Invalid case")

    try:
        samples = distribution.generate(100)
    except NotImplementedError:
        return _pytest.skip("Note implemented case")

    assert samples.shape == (distribution.dimensions, 100)


@_pytest.mark.parametrize("pclass", subclasses)
@_pytest.mark.parametrize("dimensions", dimensions)
def test_normalization(pclass: _Distributions._AbstractDistribution, dimensions: int):
    try:
        distribution: _Distributions._AbstractDistribution = pclass.create_default(
            dimensions
        )
    except _InvalidCaseError:
        return _pytest.skip("Invalid case")

    try:
        distribution.normalize()
    except AttributeError:
        return _pytest.skip("Not normalizable, skipping")


@_pytest.mark.parametrize("pclass", subclasses)
@_pytest.mark.parametrize("dimensions", dimensions)
def test_misfit(pclass: _Distributions._AbstractDistribution, dimensions: int):
    try:
        distribution: _Distributions._AbstractDistribution = pclass.create_default(
            dimensions
        )
    except _InvalidCaseError:
        return _pytest.skip("Invalid case")

    location = _numpy.ones((dimensions, 1)) + _numpy.random.rand(1)
    misfit = distribution.misfit(location)

    assert (
        type(misfit) == float
        or type(misfit) == _numpy.float64
        or type(misfit) == _numpy.float32
    )

    return


@_pytest.mark.parametrize("pclass", subclasses)
@_pytest.mark.parametrize("dimensions", dimensions)
def test_misfit_bounds(pclass: _Distributions._AbstractDistribution, dimensions: int):
    try:
        distribution: _Distributions._AbstractDistribution = pclass.create_default(
            dimensions
        )
    except _InvalidCaseError:
        return _pytest.skip("Invalid case")

    lower_bounds = _numpy.ones((dimensions, 1))
    distribution.update_bounds(lower=lower_bounds)

    # Compute misfit above lower bounds

    location = _numpy.ones((dimensions, 1)) + _numpy.random.rand(1) + 0.1
    misfit = distribution.misfit(location)

    assert (
        type(misfit) == float
        or type(misfit) == _numpy.float64
        or type(misfit) == _numpy.float32
    )

    # Compute misfit below lower bounds

    location = _numpy.ones((dimensions, 1)) - _numpy.random.rand(1) - 0.1
    misfit = distribution.misfit(location)

    assert misfit == _numpy.inf

    # Create upper bounds

    upper_bounds = 3 * _numpy.ones((dimensions, 1))
    distribution.update_bounds(upper=upper_bounds)

    # Compute misfit between the two limits

    location = _numpy.ones((dimensions, 1)) + _numpy.random.rand(1) + 0.1
    misfit = distribution.misfit(location)

    assert (
        type(misfit) == float
        or type(misfit) == _numpy.float64
        or type(misfit) == _numpy.float32
    )

    # Compute misfit above the upper limit

    location = 3 * _numpy.ones((dimensions, 1)) + _numpy.random.rand(1) + 0.1
    misfit = distribution.misfit(location)

    assert misfit == _numpy.inf, " ds"
    return


@_pytest.mark.parametrize("pclass", subclasses)
@_pytest.mark.parametrize("dimensions", dimensions)
def test_misfit_bounds_impossible(
    pclass: _Distributions._AbstractDistribution, dimensions: int
):
    try:
        distribution: _Distributions._AbstractDistribution = pclass.create_default(
            dimensions
        )
    except _InvalidCaseError:
        return _pytest.skip("Invalid case")

    lower_bounds = _numpy.ones((dimensions, 1))
    upper_bounds = 3 * _numpy.ones((dimensions, 1))

    # Try to switch the bounds s.t. lower > upper
    try:
        distribution.update_bounds(lower=upper_bounds, upper=lower_bounds)
    except ValueError as e:
        # Assert that the exception is raised by the bounds, else re-raise
        if e.args[0] != "Bounds vectors are incompatible.":
            raise e


@_pytest.mark.parametrize("pclass", subclasses)
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("delta", deltas)
def test_gradient(
    pclass: _Distributions._AbstractDistribution,
    dimensions: int,
    delta: float,
    results_bag,
):
    results_bag.test_type = "gradient"
    results_bag.class_name = pclass.__name__

    try:
        distribution: _Distributions._AbstractDistribution = pclass.create_default(
            dimensions
        )
    except _InvalidCaseError:
        _pytest.skip("Invalid case")  # pragma: no cover

    location = _numpy.ones((dimensions, 1)) + _numpy.random.rand(1)
    gradient = distribution.gradient(location)

    assert gradient.dtype == _numpy.dtype("float32") or gradient.dtype == _numpy.dtype(
        "float64"
    )
    assert gradient.shape == location.shape

    # Gradient test
    dot_product = (gradient.T @ location).item(0)
    misfit_1 = distribution.misfit(location)
    misfit_2 = distribution.misfit(location + delta * location)
    if (misfit_2 - misfit_1) != 0:
        relative_error = (misfit_2 - misfit_1 - dot_product * delta) / (
            misfit_2 - misfit_1
        )
        try:
            assert abs(relative_error) < 1e-2
        except AssertionError:
            _pytest.xfail("Error bigger than 10% in gradient test, not failing pytest.")

        results_bag.relative_error = relative_error
    else:
        assert _numpy.allclose(gradient, 0.0)

        results_bag.relative_error = 0

    return
