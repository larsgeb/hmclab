"""A collection of tests for likelihood functions.
"""
from hmc_tomography import Likelihoods as _Targets
import pytest as _pytest
import numpy as _numpy


@_pytest.mark.parametrize("pclass", _Targets._AbstractLikelihood.__subclasses__())
@_pytest.mark.parametrize("dimensions", [2, 10, 100, 1000])
def test_creation(pclass: _Targets._AbstractLikelihood, dimensions: int):
    """Test for the creation of targets.

    Parameters
    ==========
    pclass : hmc_tomography._Targets._AbstractLikelihood
        A target class.
    dimensions : int
        Dimensions to check the target.


    This test checks if we can create a given target. Using pytest, it will loop over
    all available subclasses of targets, with variable amount of dimensions.
    """

    # Create the object
    try:
        target: _Targets._AbstractLikelihood = pclass(dimensions)
    except NotImplementedError:
        return 0

    # Check if a subtype of mass matrices
    assert issubclass(type(target), _Targets._AbstractLikelihood)

    # Check if the right amount of dimensions
    assert target.dimensions == dimensions

    return True


@_pytest.mark.parametrize("pclass", _Targets._AbstractLikelihood.__subclasses__())
@_pytest.mark.parametrize("dimensions", [2, 10, 100, 1000])
def test_misfit(pclass: _Targets._AbstractLikelihood, dimensions: int):
    """Test for the computation of target misfits.

    Parameters
    ==========
    pclass : hmc_tomography._Targets._AbstractLikelihood
        A target class.
    dimensions : int
        Dimensions to check the target.


    This test checks if we can compute the misfit of a given target. Using pytest, it
    will loop over all available subclasses of targets, with variable amount of
    dimensions.
    """

    try:
        target: _Targets._AbstractLikelihood = pclass(dimensions)
    except NotImplementedError:
        return 0

    location = _numpy.ones((dimensions, 1))

    misfit = target.misfit(location)

    assert type(misfit) == float or type(misfit) == _numpy.dtype("float")

    return True


@_pytest.mark.parametrize("pclass", _Targets._AbstractLikelihood.__subclasses__())
@_pytest.mark.parametrize("dimensions", [2, 10, 100, 1000])
@_pytest.mark.parametrize("stepsize_delta", [1e-10, 1e-5, 1e-3, -1e-10, -1e-5, -1e-3])
def test_gradient(
    pclass: _Targets._AbstractLikelihood, dimensions: int, stepsize_delta: float
):
    """Test for the computation of target gradients.

    Parameters
    ==========
    pclass : hmc_tomography._Targets._AbstractLikelihood
        A target class.
    dimensions : int
        Dimensions to check the target.


    This test checks if we can compute the gradient of a given target. Using pytest, it
    will loop over all available subclasses of targets, with variable amount of
    dimensions.
    """

    try:
        target: _Targets._AbstractLikelihood = pclass(dimensions)
    except NotImplementedError:
        return 0

    location = 0.75 * _numpy.ones((dimensions, 1))
    gradient = target.gradient(location)

    assert gradient.dtype == _numpy.dtype("float")
    assert gradient.shape == location.shape

    # Gradient test
    dot_product = (gradient.T @ location).item(0)
    misfit_1 = target.misfit(location)
    misfit_2 = target.misfit(location + stepsize_delta * location)
    if (misfit_2 - misfit_1) != 0:
        relative_error = (misfit_2 - misfit_1 - dot_product * stepsize_delta) / (
            misfit_2 - misfit_1
        )
        assert abs(relative_error) < 1e-2
    else:
        assert _numpy.allclose(gradient, 0.0)

    return True
