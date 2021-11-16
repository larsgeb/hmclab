"""A collection of tests for mass matrices.
"""
import numpy as _numpy
import pytest as _pytest
import os as _os

from hmclab import MassMatrices as _MassMatrices, Samplers as _Samplers
from hmclab.Distributions import Normal as _Normal

dimensions = [1, 10, 100]
subclasses = _MassMatrices._AbstractMassMatrix.__subclasses__()


@_pytest.mark.parametrize("mmclass", subclasses)
@_pytest.mark.parametrize("dimensions", dimensions)
def test_creation(mmclass: _MassMatrices._AbstractMassMatrix, dimensions: int):

    # Create the object
    mass_matrix: _MassMatrices._AbstractMassMatrix = mmclass.create_default(dimensions)

    # Check if a subtype of mass matrices
    assert issubclass(type(mass_matrix), _MassMatrices._AbstractMassMatrix)

    # Check if the right amount of dimensions
    assert mass_matrix.dimensions == dimensions

    return True


@_pytest.mark.parametrize("mmclass", subclasses)
@_pytest.mark.parametrize("dimensions", dimensions)
def test_generate(mmclass: _MassMatrices._AbstractMassMatrix, dimensions: int):

    # Create the object
    mass_matrix: _MassMatrices._AbstractMassMatrix = mmclass.create_default(dimensions)

    # Generate momentum
    momentum = mass_matrix.generate_momentum()

    # Assert column vector shape
    assert momentum.shape == (dimensions, 1)

    # Assert float type
    assert momentum.dtype is _numpy.dtype("float")

    return True


@_pytest.mark.parametrize("mmclass", subclasses)
@_pytest.mark.parametrize("dimensions", dimensions)
def test_kinetic_energy(mmclass: _MassMatrices._AbstractMassMatrix, dimensions: int):

    # Create the object
    mass_matrix: _MassMatrices._AbstractMassMatrix = mmclass.create_default(dimensions)

    # Generate momentum
    momentum = mass_matrix.generate_momentum()
    kinetic_energy = mass_matrix.kinetic_energy(momentum)

    # Assert float type
    assert type(kinetic_energy) == float or type(kinetic_energy) == _numpy.float64

    return True


@_pytest.mark.parametrize("mmclass", subclasses)
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("stepsize_delta", [1e-10, 1e-5, 1e-2, -1e-10, -1e-5, -1e-2])
def test_kinetic_energy_gradient(
    mmclass: _MassMatrices._AbstractMassMatrix, dimensions: int, stepsize_delta: float
):

    # Create the object
    mass_matrix: _MassMatrices._AbstractMassMatrix = mmclass.create_default(dimensions)

    # Generate momentum
    momentum = mass_matrix.generate_momentum()
    kinetic_energy_gradient = mass_matrix.kinetic_energy_gradient(momentum)

    # Assert column vector shape
    assert kinetic_energy_gradient.shape == (dimensions, 1)

    # Assert float type
    assert kinetic_energy_gradient.dtype is _numpy.dtype("float")

    # Gradient test
    dot_product = (kinetic_energy_gradient.T @ momentum).item(0)

    kinetic_1 = mass_matrix.kinetic_energy(momentum)
    kinetic_2 = mass_matrix.kinetic_energy(momentum + stepsize_delta * momentum)
    if (kinetic_2 - kinetic_1) != 0:
        relative_error = (kinetic_2 - kinetic_1 - dot_product * stepsize_delta) / (
            kinetic_2 - kinetic_1
        )
        assert relative_error < 1e-2
    else:
        assert _numpy.allclose(kinetic_energy_gradient, 0.0)

    return True


@_pytest.mark.parametrize("dimensions", dimensions)
def test_basic_sampling(
    dimensions: int,
):

    means = _numpy.zeros((dimensions, 1))
    covariance = _numpy.eye(dimensions)
    distribution = _Normal(means, covariance)

    sampler_instance = _Samplers.HMC()

    filename = "temporary_file.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)

    proposals = 1000

    sampler_instance.sample(
        filename,
        distribution,
        proposals=proposals,
        online_thinning=10,
        ram_buffer_size=int(proposals / _numpy.random.rand() * 10),
        max_time=1.0,
        autotuning=False,
    )
    if sampler_instance.amount_of_writes > 0:
        # 10 percent burn_in
        burn_in = int(0.1 * sampler_instance.amount_of_writes)
        sampler_instance.load_results(burn_in=burn_in)

    # Check if the file was created. If it wasn't, fail
    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")

    # Remove the file
    _os.remove(filename)
