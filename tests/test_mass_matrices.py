"""A collection of tests for mass matrices.
"""
import numpy as _numpy
import pytest as _pytest
import os as _os
import uuid as _uuid


from hmclab import MassMatrices as _MassMatrices
from hmclab import Samplers as _Samplers
from hmclab import Distributions as _Distributions
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

    assert mass_matrix.matrix.shape == (dimensions, dimensions)

    return


@_pytest.mark.parametrize("mmclass", subclasses)
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("rng", [None, _numpy.random.default_rng()])
def test_generate(
    mmclass: _MassMatrices._AbstractMassMatrix,
    dimensions: int,
    rng: _numpy.random.Generator,
):
    # Create the object
    mass_matrix: _MassMatrices._AbstractMassMatrix = mmclass.create_default(
        dimensions, rng=rng
    )

    # Generate momentum
    momentum = mass_matrix.generate_momentum()

    # Assert column vector shape
    assert momentum.shape == (dimensions, 1)

    # Assert float type
    assert momentum.dtype is _numpy.dtype("float")

    return


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

    # Try to compute kinetic energy which SHOULD fail
    momentum = _numpy.vstack((momentum, _numpy.ones((1, 1))))
    with _pytest.raises(ValueError):
        kinetic_energy = mass_matrix.kinetic_energy(momentum)

    return


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
        assert _numpy.allclose(kinetic_energy_gradient, 0.0)  # pragma: no cover

    # Try to compute kinetic energy gradient which SHOULD fail
    momentum = _numpy.vstack((momentum, _numpy.ones((1, 1))))
    with _pytest.raises(ValueError):
        kinetic_energy_gradient = mass_matrix.kinetic_energy_gradient(momentum)

    return


@_pytest.mark.parametrize("dimensions", dimensions)
def test_basic_sampling(
    dimensions: int,
):
    means = _numpy.zeros((dimensions, 1))
    covariance = _numpy.eye(dimensions)
    distribution = _Normal(means, covariance)

    sampler_instance = _Samplers.HMC()

    unique_name = _uuid.uuid4().hex.upper()
    filename = f"temporary_file_{unique_name}.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)  # pragma: no cover  # pragma: no cover

    proposals = 1000

    sampler_instance.sample(
        filename,
        distribution,
        proposals=proposals,
        online_thinning=10,
        max_time=0.1,
        autotuning=False,
        disable_progressbar=True,
    )
    if sampler_instance.amount_of_writes > 0:
        # 10 percent burn_in
        burn_in = int(0.1 * sampler_instance.amount_of_writes)
        sampler_instance.load_results(burn_in=burn_in)

    # Check if the file was created. If it wasn't, fail
    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")  # pragma: no cover

    # Remove the file
    _os.remove(filename)


def test_full_massmatrix():
    """Test all parts of the full mass matrix that aren't hit yet."""
    non_symmetric_matrix = _numpy.tri(10)

    with _pytest.raises(AssertionError):
        _MassMatrices.Full(non_symmetric_matrix)
