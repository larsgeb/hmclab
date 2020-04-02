"""A collection of tests for likelihood functions.
"""
from hmc_tomography import Distributions as _Distributions
from hmc_tomography.Helpers.CustomExceptions import (
    AbstractMethodError as _AbstractMethodError,
)
from hmc_tomography.Helpers.CustomExceptions import (
    InvalidCaseError as _InvalidCaseError,
)
import pytest as _pytest
import numpy as _numpy


@_pytest.mark.parametrize(
    "pclass", _Distributions._AbstractDistribution.__subclasses__()
)
@_pytest.mark.parametrize("dimensions", [1, 2, 10, 100, 1000])
def test_creation(pclass: _Distributions._AbstractDistribution, dimensions: int):
    # Create the object
    try:
        distribution: _Distributions._AbstractDistribution = pclass.create_default(
            dimensions
        )
    except _InvalidCaseError:
        return 0

    # Check if a subtype of mass matrices
    assert issubclass(type(distribution), _Distributions._AbstractDistribution)

    # Check if the right amount of dimensions
    assert distribution.dimensions == dimensions

    return True


@_pytest.mark.parametrize(
    "pclass", _Distributions._AbstractDistribution.__subclasses__()
)
@_pytest.mark.parametrize("dimensions", [1, 2, 10, 100, 1000])
def test_misfit(pclass: _Distributions._AbstractDistribution, dimensions: int):
    try:
        distribution: _Distributions._AbstractDistribution = pclass.create_default(
            dimensions
        )
    except _InvalidCaseError:
        return 0

    location = _numpy.ones((dimensions, 1))

    misfit = distribution.misfit(location)

    assert type(misfit) == float

    return True


@_pytest.mark.parametrize(
    "pclass", _Distributions._AbstractDistribution.__subclasses__()
)
@_pytest.mark.parametrize("dimensions", [1, 2, 10, 100, 1000])
@_pytest.mark.parametrize("stepsize_delta", [1e-10, 1e-5, 1e-3, -1e-10, -1e-5, -1e-4])
def test_gradient(
    pclass: _Distributions._AbstractDistribution, dimensions: int, stepsize_delta: float
):
    try:
        distribution: _Distributions._AbstractDistribution = pclass.create_default(
            dimensions
        )
    except _InvalidCaseError:
        return 0

    location = _numpy.ones((dimensions, 1))
    gradient = distribution.gradient(location)

    assert gradient.dtype == _numpy.dtype("float")
    assert gradient.shape == location.shape

    # Gradient test
    dot_product = (gradient.T @ location).item(0)
    misfit_1 = distribution.misfit(location)
    misfit_2 = distribution.misfit(location + stepsize_delta * location)
    if (misfit_2 - misfit_1) != 0:
        relative_error = (misfit_2 - misfit_1 - dot_product * stepsize_delta) / (
            misfit_2 - misfit_1
        )
        assert abs(relative_error) < 1e-2
    else:
        assert _numpy.allclose(gradient, 0.0)

    return True
