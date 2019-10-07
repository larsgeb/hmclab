"""A collection of tests for mass matrices.
"""
from hmc_tomography import Priors as _Priors
import pytest as _pytest
import numpy as _numpy


@_pytest.mark.parametrize("pclass", _Priors._AbstractPrior.__subclasses__())
@_pytest.mark.parametrize("dimensions", [1, 10, 100, 1000])
def test_creation(pclass: _Priors._AbstractPrior, dimensions: int):
    """Test for the creation of prior.

    Parameters
    ==========
    pclass : hmc_tomography._Priors._AbstractPrior
        A prior class.
    dimensions : int
        Dimensions to check the prior.


    This test checks if we can create a given prior. Using pytest, it will loop over
    all available subclasses of priors, with variable amount of dimensions.
    """

    # Create the object
    prior: _Priors._AbstractPrior = pclass(dimensions)

    # Check if a subtype of mass matrices
    assert issubclass(type(prior), _Priors._AbstractPrior)

    # Check if the right amount of dimensions
    assert prior.dimensions == dimensions

    return True


@_pytest.mark.xfail(raises=NotImplementedError)
@_pytest.mark.parametrize("pclass", _Priors._AbstractPrior.__subclasses__())
@_pytest.mark.parametrize("dimensions", [1, 10, 100, 1000])
def test_generate(pclass: _Priors._AbstractPrior, dimensions: int):
    """Test for the generation of prior samples.

    Parameters
    ==========
    pclass : hmc_tomography._Priors._AbstractPrior
        A prior class.
    dimensions : int
        Dimensions to check the prior.


    This test checks if we can generate a sample of a given prior. Using pytest, it
    will loop over all available subclasses of priors, with variable amount of
    dimensions.
    """

    # Create the object
    prior: _Priors._AbstractPrior = pclass(dimensions)

    sample = prior.generate()

    assert sample.shape == (dimensions, 1)

    assert sample.dtype == _numpy.dtype("float")

    return True


@_pytest.mark.parametrize("pclass", _Priors._AbstractPrior.__subclasses__())
@_pytest.mark.parametrize("dimensions", [1, 10, 100, 1000])
def test_misfit(pclass: _Priors._AbstractPrior, dimensions: int):
    """Test for the computation of prior misfits.

    Parameters
    ==========
    pclass : hmc_tomography._Priors._AbstractPrior
        A prior class.
    dimensions : int
        Dimensions to check the prior.


    This test checks if we can compute the misfit of a given prior. Using pytest, it
    will loop over all available subclasses of priors, with variable amount of
    dimensions.
    """

    # Create the object
    prior: _Priors._AbstractPrior = pclass(dimensions)

    location = _numpy.ones((dimensions, 1))

    misfit = prior.misfit(location)

    assert type(misfit) == float or type(misfit) == _numpy.dtype("float")

    return True


@_pytest.mark.parametrize("pclass", _Priors._AbstractPrior.__subclasses__())
@_pytest.mark.parametrize("dimensions", [1, 10, 100, 1000])
def test_gradient(pclass: _Priors._AbstractPrior, dimensions: int):
    """Test for the computation of prior gradients.

    Parameters
    ==========
    pclass : hmc_tomography._Priors._AbstractPrior
        A prior class.
    dimensions : int
        Dimensions to check the prior.


    This test checks if we can compute the gradient of a given prior. Using pytest, it
    will loop over all available subclasses of priors, with variable amount of
    dimensions.
    """

    # Create the object
    prior: _Priors._AbstractPrior = pclass(dimensions)

    location = _numpy.ones((dimensions, 1))

    gradient = prior.gradient(location)

    assert gradient.dtype == _numpy.dtype("float")

    assert gradient.shape == location.shape

    return True


@_pytest.mark.parametrize("pclass", _Priors._AbstractPrior.__subclasses__())
@_pytest.mark.parametrize("dimensions", [1, 10, 100, 1000])
def test_bounds(pclass: _Priors._AbstractPrior, dimensions: int):
    """Test for correct implementation of prior bounds.

    Parameters
    ==========
    pclass : hmc_tomography._Priors._AbstractPrior
        A prior class.
    dimensions : int
        Dimensions to check the prior.


    This test checks all properties of ``_Priors._AbstractPrior.update_bounds()`` and
    its consequences. Using pytest, it will loop over all available subclasses of
    priors, with variable amount of dimensions.
    """

    # Create the object
    prior: _Priors._AbstractPrior = pclass(dimensions)

    location = _numpy.ones((dimensions, 1))

    # Compute a finite misfit
    misfit = prior.misfit(location)

    assert misfit < _numpy.inf

    # Create a lower bound above the current location ----------------------------------
    prior.update_bounds(2 * _numpy.ones_like(location), None)
    # Compute infinite misfit
    misfit = prior.misfit(location)
    assert misfit == _numpy.inf
    # Move the bound below the location
    prior.update_bounds(_numpy.zeros_like(location), None)
    # Compute a finite misfit
    misfit = prior.misfit(location)
    assert misfit < _numpy.inf
    # Reset bounds
    prior.update_bounds(None, None)

    # Create an upper bound below the current location ---------------------------------
    prior.update_bounds(None, _numpy.zeros_like(location))
    # Compute infinite misfit
    misfit = prior.misfit(location)
    assert misfit == _numpy.inf
    # Move the bound below the location
    prior.update_bounds(None, 2 * _numpy.ones_like(location))
    # Compute a finite misfit
    misfit = prior.misfit(location)
    assert misfit < _numpy.inf
    # Reset bounds
    prior.update_bounds(None, None)

    # Assert that we can't add upper_bounds below lower_bounds -------------------------
    with _pytest.raises(ValueError):
        lower = _numpy.arange(dimensions) + 2
        upper = _numpy.arange(dimensions)
        lower.shape = (dimensions, 1)
        upper.shape = (dimensions, 1)
        prior.update_bounds(lower, upper)

    return True
