from hmc_tomography import MassMatrices as _MassMatrices
import pytest as _pytest
import numpy as _numpy


@_pytest.mark.xfail(raises=NotImplementedError)
@_pytest.mark.parametrize("mmclass", _MassMatrices._AbstractMassMatrix.__subclasses__())
@_pytest.mark.parametrize("dimensions", [1, 10, 100, 1000])
def test_creation(mmclass: _MassMatrices._AbstractMassMatrix, dimensions: int):
    """Test for the creation of mass matrices.
    """

    # Create the object
    mass_matrix: _MassMatrices._AbstractMassMatrix = mmclass(dimensions)

    # Check if a subtype of mass matrices
    assert issubclass(type(mass_matrix), _MassMatrices._AbstractMassMatrix)

    # Check if the right amount of dimensions
    assert mass_matrix.dimensions == dimensions

    return True


@_pytest.mark.xfail(raises=NotImplementedError)
@_pytest.mark.parametrize("mmclass", _MassMatrices._AbstractMassMatrix.__subclasses__())
@_pytest.mark.parametrize("dimensions", [1, 10, 100, 1000])
def test_generate(mmclass: _MassMatrices._AbstractMassMatrix, dimensions: int):
    """Test for the generation of momenta from mass matrices.

    Parameters
    ==========
    mmclass : hmc_tomography.MassMatrices._AbstractMassMatrix
        A mass matrix class.
    dimensions : int
        Dimensions to check the mass matrix.

    This test checks if we can generate momentum from a given mass matrix. Using pytest,
    it will loop over all available subclasses of the mass matrix, with variable amount
    of dimensions.
    """

    # Create the object
    mass_matrix: _MassMatrices._AbstractMassMatrix = mmclass(dimensions)

    # Generate momentum
    momentum = mass_matrix.generate_momentum()

    # Assert column vector shape
    assert momentum.shape == (dimensions, 1)

    # Assert float type
    assert momentum.dtype is _numpy.dtype("float")

    return True
