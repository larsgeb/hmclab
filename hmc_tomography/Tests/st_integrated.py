"""A collection of integrated tests.
"""
import hmc_tomography
import pytest as _pytest
import numpy as _numpy
import matplotlib.pyplot as _plt


@_pytest.mark.parametrize("dimensions", [5, 50, 100])
@_pytest.mark.parametrize("data_covariance_scalar", [1e-5, 1e0, 1e5])
def test_diagonal_matrix_optimality_float_covariance(
    dimensions: int, data_covariance_scalar: float, plot: bool = False
):
    """Test for the diagonal of mass matrix efficiency.

    Parameters
    ==========
    dimensions : int
        Dimensions to check the diagonal mass matrix.


    This test checks if the diagonal mass matrix is more optimal than a unit mass
    matrix. Optionally generates a plot of error in variances if `plot=True` is passed.
    """

    G = _numpy.eye(dimensions)

    G[-1, -1] = 4.464e4
    G[0, 0] = 3.468e-4

    d = _numpy.zeros((dimensions, 1))

    # This is the exact posterior covariance if G is diagonal and data covariance is a
    # scalar
    inverse_covariance_diagonal = (G.T @ G).diagonal()[:, None] / data_covariance_scalar

    target = hmc_tomography.Likelihoods.LinearMatrix(
        dimensions, G=G, d=d, data_covariance=data_covariance_scalar
    )

    prior = hmc_tomography.Priors.Uniform(dimensions)

    unit_mass_matrix = hmc_tomography.MassMatrices.Unit(dimensions)

    unit_mm_sampler = hmc_tomography.Samplers.HMC(
        target=target, mass_matrix=unit_mass_matrix, prior=prior
    )

    unit_mm_sampler.sample(
        "unit_mass_matrix_samples.h5",
        time_step=(data_covariance_scalar ** 0.5)
        / G.max(),  # A good standard choice if G is diagonal
        proposals=5000,
        overwrite_samples=True,
    )

    with hmc_tomography.Post.Samples(
        "unit_mass_matrix_samples.h5", burn_in=0
    ) as samples:
        # Note that parameter -1 is NOT a dimension of the posterior. It is the misfit.
        if plot:
            hmc_tomography.Post.Visualization.marginal_grid(
                samples, [0, 1, -3, -2], show=True, bins=25
            )

        variance_errors_unit = _numpy.log(
            _numpy.var(samples[:-1, :], axis=1)[:, None]
        ) - _numpy.log(1.0 / inverse_covariance_diagonal)

    diagonal_mass_matrix = hmc_tomography.MassMatrices.Diagonal(
        dimensions, inverse_covariance_diagonal
    )

    diagonal_mm_sampler = hmc_tomography.Samplers.HMC(
        target=target, mass_matrix=diagonal_mass_matrix, prior=prior
    )

    diagonal_mm_sampler.sample(
        "diagonal_mass_matrix_samples.h5",
        time_step=0.1,
        proposals=5000,
        overwrite_samples=True,
    )

    with hmc_tomography.Post.Samples(
        "diagonal_mass_matrix_samples.h5", burn_in=0
    ) as samples:
        # Note that parameter -1 is NOT a dimension of the posterior. It is the misfit.
        if plot:
            hmc_tomography.Post.Visualization.marginal_grid(
                samples, [0, 1, -3, -2], show=True, bins=25
            )
        variance_errors_diagonal = _numpy.log(
            _numpy.var(samples[:-1, :], axis=1)[:, None]
        ) - _numpy.log(1.0 / inverse_covariance_diagonal)

    if plot:
        _plt.subplot(211)
        _plt.title("Unit mass matrix variance log error")
        _plt.hist(variance_errors_unit, color="black")
        _plt.subplot(212)
        _plt.title("Diagonal mass matrix variance log error")
        _plt.hist(variance_errors_diagonal, color="black")
        _plt.tight_layout()
        _plt.show()

    # Assert that the diagonal mass matrix estimates the order of posterior variance
    # magnitude well enough
    assert all(_numpy.abs(variance_errors_diagonal) < 1.0)

    return 0


@_pytest.mark.parametrize("dimensions", [5, 50, 100])
@_pytest.mark.parametrize("data_covariance_scalar", [1e-5, 1e0, 1e5])
def test_diagonal_matrix_optimality_float_covariance_prior(
    dimensions: int, data_covariance_scalar: float, plot: bool = False
):
    """Test for the diagonal of mass matrix efficiency.

    Parameters
    ==========
    dimensions : int
        Dimensions to check the diagonal mass matrix.


    This test checks if the diagonal mass matrix is more optimal than a unit mass
    matrix. Optionally generates a plot of error in variances if `plot=True` is passed.
    """

    if dimensions < 5:
        raise NotImplementedError("For this test, dimensions has to be at least 5.")

    G = _numpy.eye(dimensions)

    G[-1, -1] = 4.464e4
    G[0, 0] = 3.468e-4

    d = _numpy.zeros((dimensions, 1))

    # This is the exact posterior covariance if G is diagonal and data covariance is a
    # scalar
    inverse_likelihood_covariance_diagonal = (G.T @ G).diagonal()[
        :, None
    ] / data_covariance_scalar

    target = hmc_tomography.Likelihoods.LinearMatrix(
        dimensions, G=G, d=d, data_covariance=data_covariance_scalar
    )

    prior_covariance = _numpy.ones((dimensions, 1))
    prior_covariance[-3] = 4.897e10
    prior_covariance[3] = 7.897e-10

    prior = hmc_tomography.Priors.Normal(
        dimensions, means=_numpy.zeros((dimensions, 1)), covariance=prior_covariance
    )

    inverse_prior_covariance_diagonal = 1.0 / prior_covariance
    inverse_posterior_covariance_diagonal = (
        inverse_prior_covariance_diagonal + inverse_likelihood_covariance_diagonal
    )

    unit_mass_matrix = hmc_tomography.MassMatrices.Unit(dimensions)

    unit_mm_sampler = hmc_tomography.Samplers.HMC(
        target=target, mass_matrix=unit_mass_matrix, prior=prior
    )

    unit_mm_sampler.sample(
        "unit_mass_matrix_samples.h5",
        time_step=(data_covariance_scalar ** 0.5)
        / G.max(),  # A good standard choice if G is diagonal
        proposals=5000,
        overwrite_samples=True,
    )

    with hmc_tomography.Post.Samples(
        "unit_mass_matrix_samples.h5", burn_in=0
    ) as samples:
        # Note that parameter -1 is NOT a dimension of the posterior. It is the misfit.
        if plot:
            hmc_tomography.Post.Visualization.marginal_grid(
                samples, [0, 1, -3, -2], show=True, bins=25
            )

        variance_errors_unit = _numpy.log(
            _numpy.var(samples[:-1, :], axis=1)[:, None]
        ) - _numpy.log(1.0 / inverse_posterior_covariance_diagonal)

    diagonal_mass_matrix = hmc_tomography.MassMatrices.Diagonal(
        dimensions, inverse_posterior_covariance_diagonal
    )

    diagonal_mm_sampler = hmc_tomography.Samplers.HMC(
        target=target, mass_matrix=diagonal_mass_matrix, prior=prior
    )

    diagonal_mm_sampler.sample(
        "diagonal_mass_matrix_samples.h5",
        time_step=0.1,
        proposals=5000,
        overwrite_samples=True,
    )

    with hmc_tomography.Post.Samples(
        "diagonal_mass_matrix_samples.h5", burn_in=0
    ) as samples:
        # Note that parameter -1 is NOT a dimension of the posterior. It is the misfit.
        if plot:
            hmc_tomography.Post.Visualization.marginal_grid(
                samples, [0, 1, -3, -2], show=True, bins=25
            )
        variance_errors_diagonal = _numpy.log(
            _numpy.var(samples[:-1, :], axis=1)[:, None]
        ) - _numpy.log(1.0 / inverse_posterior_covariance_diagonal)

    if plot:
        _plt.subplot(211)
        _plt.title("Unit mass matrix variance log error")
        _plt.hist(variance_errors_unit, color="black")
        _plt.subplot(212)
        _plt.title("Diagonal mass matrix variance log error")
        _plt.hist(variance_errors_diagonal, color="black")
        _plt.tight_layout()
        _plt.show()

    # Assert that the diagonal mass matrix estimates the order of posterior variance
    # magnitude well enough
    assert all(_numpy.abs(variance_errors_diagonal) < 1.0)

    return 0
