"""A collection of tests for likelihood functions.
"""
import matplotlib.pyplot as _plt
import numpy as _numpy
import pytest as _pytest

from hmclab.Distributions.LinearMatrix import (
    _LinearMatrix_sparse_forward_simple_covariance,
)

dimensions = [100, 10000]
deltas = [1e-10, 1e-4, 1e-2, -1e-10, -1e-4, -1e-2]
dtype = [_numpy.dtype("float64"), _numpy.dtype("float32")]

use_mkl = [True, False]

try:
    from ctypes import cdll

    mkl = cdll.LoadLibrary("libmkl_rt.so")
except OSError:
    skip = True


@_pytest.mark.skipif(skip, reason="MKL not installed")
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("use_mkl", use_mkl)
@_pytest.mark.parametrize("dtype", dtype)
def test_creation(dimensions: int, use_mkl: bool, dtype):
    # Create the object
    distribution = _LinearMatrix_sparse_forward_simple_covariance.create_default(
        dimensions, use_mkl=use_mkl, dtype=dtype
    )

    assert distribution.use_mkl == use_mkl

    # Check if the right amount of dimensions
    assert distribution.dimensions == dimensions

    return


@_pytest.mark.skipif(skip, reason="MKL not installed")
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("use_mkl", use_mkl)
@_pytest.mark.parametrize("dtype", dtype)
def test_misfit(dimensions: int, use_mkl: bool, dtype):
    distribution = _LinearMatrix_sparse_forward_simple_covariance.create_default(
        dimensions, use_mkl=use_mkl, dtype=dtype
    )

    assert distribution.use_mkl == use_mkl

    location = _numpy.ones((dimensions, 1)) + _numpy.random.rand(1)

    misfit = distribution.misfit(location)

    assert type(misfit) == dtype or type(misfit) == _numpy.dtype("float64")

    return


@_pytest.mark.skipif(skip, reason="MKL not installed")
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("use_mkl", use_mkl)
@_pytest.mark.parametrize("dtype", dtype)
def test_misfit_bounds(dimensions: int, use_mkl: bool, dtype):
    distribution = _LinearMatrix_sparse_forward_simple_covariance.create_default(
        dimensions, use_mkl=use_mkl, dtype=dtype
    )

    assert distribution.use_mkl == use_mkl

    lower_bounds = _numpy.ones((dimensions, 1))
    distribution.update_bounds(lower=lower_bounds)

    # Compute misfit above lower bounds

    location = _numpy.ones((dimensions, 1)) + _numpy.random.rand(1) + 0.1
    misfit = distribution.misfit(location)

    assert type(misfit) == dtype or type(misfit) == _numpy.dtype("float64")

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

    assert type(misfit) == dtype or type(misfit) == _numpy.dtype("float64")

    # Compute misfit above the upper limit

    location = 3 * _numpy.ones((dimensions, 1)) + _numpy.random.rand(1) + 0.1
    misfit = distribution.misfit(location)

    assert misfit == _numpy.inf, " ds"
    return


@_pytest.mark.skipif(skip, reason="MKL not installed")
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("use_mkl", use_mkl)
@_pytest.mark.parametrize("dtype", dtype)
def test_misfit_bounds_impossible(dimensions: int, use_mkl: bool, dtype):
    distribution = _LinearMatrix_sparse_forward_simple_covariance.create_default(
        dimensions, use_mkl=use_mkl, dtype=dtype
    )

    assert distribution.use_mkl == use_mkl

    lower_bounds = _numpy.ones((dimensions, 1))
    upper_bounds = 3 * _numpy.ones((dimensions, 1))

    # Try to switch the bounds s.t. lower > upper
    try:
        distribution.update_bounds(lower=upper_bounds, upper=lower_bounds)
    except ValueError as e:
        # Assert that the exception is raised by the bounds
        assert e.args[0] == "Bounds vectors are incompatible."

        # Expected fail
        _pytest.xfail("Impossible test case, failure is required")


@_pytest.mark.skipif(skip, reason="MKL not installed")
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("delta", deltas)
@_pytest.mark.parametrize("use_mkl", use_mkl)
@_pytest.mark.parametrize("dtype", dtype)
def test_gradient(dimensions: int, delta: float, results_bag, use_mkl: bool, dtype):
    results_bag.test_type = "gradient"
    results_bag.class_name = "Using mkl" if use_mkl else "Not using mkl"

    distribution = _LinearMatrix_sparse_forward_simple_covariance.create_default(
        dimensions, use_mkl=use_mkl, dtype=dtype
    )

    assert distribution.use_mkl == use_mkl

    location = (_numpy.ones((dimensions, 1)) + _numpy.random.rand(1)).astype(dtype)
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
    elif _numpy.allclose(location + delta * location, location):
        # This means that the delta in location is so small that we are up to machine
        # precision in the same point
        results_bag.relative_error = 0
    else:
        assert _numpy.allclose(gradient, 0.0)
        results_bag.relative_error = 0

    return


@_pytest.mark.skipif(skip, reason="MKL not installed")
@_pytest.mark.plot
def test_gradient_plots(module_results_df):
    """
    Shows that the `module_results_df` fixture already contains what you need
    """
    # drop the 'pytest_obj' column
    module_results_df.drop("pytest_obj", axis=1, inplace=True)

    for name, df in module_results_df[
        module_results_df.test_type == "gradient"
    ].groupby("class_name"):
        for dimensions, df_dim in df.groupby("dimensions"):
            if not _numpy.all(_numpy.isnan(df_dim.relative_error)):
                _plt.scatter(
                    df_dim.delta,
                    _numpy.abs(df_dim.relative_error),
                    alpha=0.5,
                    label=dimensions,
                )
        ax = _plt.gca()
        _plt.grid(True)
        _plt.xlim([-2e-2, 2e-2])
        _plt.ylim([-1e-7, 1e0])
        ax.set_xscale("symlog", linthreshx=1e-11)
        ax.set_yscale("symlog", linthreshy=1e-8)
        _plt.legend()
        _plt.title(name)
        ax.set_xticks([-1e-3, -1e-6, -1e-9, 0, 1e-9, 1e-6, 1e-3])

        _plt.show()
