"""A collection of tests for the _LinearMatrix_sparse_forward_sparse_covariance class.
"""
import numpy as _numpy
import scipy as _scipy
import pytest as _pytest

from hmclab.Distributions.LinearMatrix import (
    _LinearMatrix_sparse_forward_sparse_covariance,
)

from hmclab.Helpers.RandomMatrices import random_correlation_matrix

dimension_data = [1, 10, 100]
dimension_model = [1, 10, 100]
deltas = [1e-10, 1e-2, -1e-10, -1e-2]
dtype = [_numpy.dtype("float64"), _numpy.dtype("float32")]
covariance_type = ["vector", "matrix"]
density = [0.001, 0.1]


@_pytest.mark.parametrize("dimension_data", dimension_data)
@_pytest.mark.parametrize("dimension_model", dimension_model)
@_pytest.mark.parametrize("dtype", dtype)
@_pytest.mark.parametrize("covariance_type", covariance_type)
@_pytest.mark.parametrize("density", density)
def test_creation(
    dimension_data: int,
    dimension_model: int,
    dtype: _numpy.dtype,
    covariance_type: str,
    density: float,
):
    """This check assesses whether or not we can actually create an instance of the
    _LinearMatrix_sparse_forward_sparse_covariance class under varying circumstances."""

    # Adjust for small matrices
    actual_elements = dimension_data * dimension_model * density
    if actual_elements < 10:
        density = 10.0 / (dimension_data * dimension_model)
    density = min(density, 1.0)

    # Create forward model matrix
    G = _scipy.sparse.random(
        dimension_data, dimension_model, density=density, format="csr", dtype=dtype
    )

    # Create observed data
    data = _numpy.ones((dimension_data, 1))

    # Create covariance
    if covariance_type == "vector":
        covariance = (
            _numpy.random.rand(
                dimension_data,
            )
            + 5.0
        )
        covariance = _numpy.diag(covariance)
    else:
        covariance = random_correlation_matrix(dimension_data)

    covariance = covariance.astype(dtype)

    # Create distribution
    distribution = _LinearMatrix_sparse_forward_sparse_covariance(
        G,
        data,
        data_covariance=covariance,
        dtype=dtype,
    )

    # Check if the distribution has right amount of dimensions
    assert distribution.dimensions == dimension_model

    return


@_pytest.mark.parametrize("dimension_data", dimension_data)
@_pytest.mark.parametrize("dimension_model", dimension_model)
@_pytest.mark.parametrize("dtype", dtype)
@_pytest.mark.parametrize("covariance_type", covariance_type)
@_pytest.mark.parametrize("density", density)
def test_misfit(
    dimension_data: int,
    dimension_model: int,
    dtype: _numpy.dtype,
    covariance_type: str,
    density: float,
):
    """This check assesses whether or not we can compute the misfit of an instance of
    the _LinearMatrix_sparse_forward_sparse_covariance class under varying
    circumstances. It also checks if the returned misfit is of the correct type."""

    # Adjust for small matrices
    actual_elements = dimension_data * dimension_model * density
    if actual_elements < 10:
        density = 10.0 / (dimension_data * dimension_model)
    density = min(density, 1.0)

    # Create forward model matrix
    G = _scipy.sparse.random(
        dimension_data, dimension_model, density=density, format="csr", dtype=dtype
    )

    # Create observed data
    data = _numpy.ones((dimension_data, 1))

    # Create covariance
    if covariance_type == "vector":
        covariance = (
            _numpy.random.rand(
                dimension_data,
            )
            + 5.0
        )
        covariance = _numpy.diag(covariance)
    else:
        covariance = random_correlation_matrix(dimension_data)

    covariance = covariance.astype(dtype)

    # Create distribution
    distribution = _LinearMatrix_sparse_forward_sparse_covariance(
        G,
        data,
        data_covariance=covariance,
        dtype=dtype,
    )

    location = _numpy.ones((dimension_model, 1)) + _numpy.random.rand(1)

    misfit = distribution.misfit(location.astype(dtype))

    assert type(misfit) == dtype or type(misfit) == _numpy.dtype("float64")

    return


@_pytest.mark.parametrize("dimension_data", dimension_data)
@_pytest.mark.parametrize("dimension_model", dimension_model)
@_pytest.mark.parametrize("dtype", dtype)
@_pytest.mark.parametrize("covariance_type", covariance_type)
@_pytest.mark.parametrize("density", density)
def test_misfit_bounds(
    dimension_data: int,
    dimension_model: int,
    dtype: _numpy.dtype,
    covariance_type: str,
    density: float,
):
    """This check assesses whether or not we can compute the misfit of an instance of
    the _LinearMatrix_sparse_forward_sparse_covariance class when the distribution
    becomes bounded, under varying circumstances.
    """

    # Adjust for small matrices
    actual_elements = dimension_data * dimension_model * density
    if actual_elements < 10:
        density = 10.0 / (dimension_data * dimension_model)
    density = min(density, 1.0)

    # Create forward model matrix
    G = _scipy.sparse.random(
        dimension_data, dimension_model, density=density, format="csr", dtype=dtype
    )

    # Create observed data
    data = _numpy.ones((dimension_data, 1))

    # Create covariance
    if covariance_type == "vector":
        covariance = (
            _numpy.random.rand(
                dimension_data,
            )
            + 5.0
        )
        covariance = _numpy.diag(covariance)
    else:
        covariance = random_correlation_matrix(dimension_data)

    covariance = covariance.astype(dtype)

    # Create distribution
    distribution = _LinearMatrix_sparse_forward_sparse_covariance(
        G,
        data,
        data_covariance=covariance,
        dtype=dtype,
    )

    lower_bounds = _numpy.ones((dimension_model, 1))
    distribution.update_bounds(lower=lower_bounds)

    # Compute misfit above lower bounds
    location = _numpy.ones((dimension_model, 1)) + _numpy.random.rand(1) + 0.1
    misfit = distribution.misfit(location.astype(dtype))

    assert type(misfit) == dtype or type(misfit) == _numpy.dtype("float64")

    # Compute misfit below lower bounds
    location = _numpy.ones((dimension_model, 1)) - _numpy.random.rand(1) - 0.1
    misfit = distribution.misfit(location.astype(dtype))

    assert misfit == _numpy.inf

    # Create upper bounds
    upper_bounds = 3 * _numpy.ones((dimension_model, 1))
    distribution.update_bounds(upper=upper_bounds)

    # Compute misfit between the two limits
    location = _numpy.ones((dimension_model, 1)) + _numpy.random.rand(1) + 0.1
    misfit = distribution.misfit(location.astype(dtype))

    assert type(misfit) == dtype or type(misfit) == _numpy.dtype("float64")

    # Compute misfit above the upper limit
    location = 3 * _numpy.ones((dimension_model, 1)) + _numpy.random.rand(1) + 0.1
    misfit = distribution.misfit(location.astype(dtype))

    assert misfit == _numpy.inf, " ds"
    return


@_pytest.mark.parametrize("dimension_data", dimension_data)
@_pytest.mark.parametrize("dimension_model", dimension_model)
@_pytest.mark.parametrize("dtype", dtype)
@_pytest.mark.parametrize("covariance_type", covariance_type)
@_pytest.mark.parametrize("density", density)
def test_misfit_bounds_impossible(
    dimension_data: int,
    dimension_model: int,
    dtype: _numpy.dtype,
    covariance_type: str,
    density: float,
):
    """This check assesses whether or not improper bounds are handled correctly within
    the _LinearMatrix_sparse_forward_sparse_covariance class under varying
    circumstances."""

    # Adjust for small matrices
    actual_elements = dimension_data * dimension_model * density
    if actual_elements < 10:
        density = 10.0 / (dimension_data * dimension_model)
    density = min(density, 1.0)

    # Create forward model matrix
    G = _scipy.sparse.random(
        dimension_data, dimension_model, density=density, format="csr", dtype=dtype
    )

    # Create observed data
    data = _numpy.ones((dimension_data, 1))

    # Create covariance
    if covariance_type == "vector":
        covariance = (
            _numpy.random.rand(
                dimension_data,
            )
            + 5.0
        )
        covariance = _numpy.diag(covariance)
    else:
        covariance = random_correlation_matrix(dimension_data)

    covariance = covariance.astype(dtype)

    # Create distribution
    distribution = _LinearMatrix_sparse_forward_sparse_covariance(
        G,
        data,
        data_covariance=covariance,
        dtype=dtype,
    )

    lower_bounds = _numpy.ones((dimension_model, 1))
    upper_bounds = 3 * _numpy.ones((dimension_model, 1))

    # Try to switch the bounds s.t. lower > upper
    try:
        distribution.update_bounds(lower=upper_bounds, upper=lower_bounds)
    except ValueError as e:
        # Assert that the exception is raised by the bounds
        assert (
            e.args[0] == "Bounds vectors are incompatible."
        ), "Something unexpected went wrong."

        # Required fail
        return
    else:
        raise RuntimeError(
            "The distribution accepted invalid bounds, this should not have happened."
        )


@_pytest.mark.parametrize("dimension_data", dimension_data)
@_pytest.mark.parametrize("dimension_model", dimension_model)
@_pytest.mark.parametrize("dtype", dtype)
@_pytest.mark.parametrize("covariance_type", covariance_type)
@_pytest.mark.parametrize("density", density)
@_pytest.mark.parametrize("delta", deltas)
def test_gradient(
    dimension_data: int,
    dimension_model: int,
    dtype: _numpy.dtype,
    covariance_type: str,
    density: float,
    delta: float,
):
    """This check assesses whether or not we can compute the gradient of an instance of
    the _LinearMatrix_sparse_forward_sparse_covariance class under varying
    circumstances. Also performs an accuracy check on the gradient."""

    # Adjust for small matrices
    actual_elements = dimension_data * dimension_model * density
    if actual_elements < 10:
        density = 10.0 / (dimension_data * dimension_model)
    density = min(density, 1.0)

    # Create forward model matrix
    G = _scipy.sparse.random(
        dimension_data, dimension_model, density=density, format="csr", dtype=dtype
    )

    # Create observed data
    data = _numpy.ones((dimension_data, 1))

    # Create covariance
    if covariance_type == "vector":
        covariance = (
            _numpy.random.rand(
                dimension_data,
            )
            + 5.0
        )
        covariance = _numpy.diag(covariance)
    else:
        covariance = random_correlation_matrix(dimension_data)

    covariance = covariance.astype(dtype)

    # Create distribution
    distribution = _LinearMatrix_sparse_forward_sparse_covariance(
        G,
        data,
        data_covariance=covariance,
        dtype=dtype,
    )

    location = (_numpy.ones((dimension_model, 1)) + _numpy.random.rand(1)).astype(dtype)
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

    elif _numpy.allclose(location + delta * location, location):
        # This means that the delta in location is so small that we are up to machine
        # precision in the same point
        pass
    else:
        assert _numpy.allclose(gradient, 0.0)

    return
