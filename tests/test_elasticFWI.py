import os as _os

import numpy as _numpy
import matplotlib.pyplot as _plt
import pytest as _pytest

import hmc_tomography as _hmc_tomography

import sys

installed = "psvWave" in sys.modules


@_pytest.mark.skipif(
    not installed, reason="Skipping test for which required packages are not installed."
)
def test_elasticFWI_creation():
    _hmc_tomography.Distributions.ElasticFullWaveform2D.create_default(
        4800, "tests/configurations/default_testing_configuration.ini",
    )


@_pytest.mark.skipif(
    not installed, reason="Skipping test for which required packages are not installed."
)
def test_elasticFWI_gradient():
    likelihood = _hmc_tomography.Distributions.ElasticFullWaveform2D.create_default(
        4800, "tests/configurations/default_testing_configuration.ini",
    )

    likelihood.temperature = 1.0

    print(f"Free parameters: {likelihood.dimensions}")

    starting_model = likelihood.get_model_vector()

    X1 = likelihood.misfit(starting_model)
    print(f"Misfit 1: {X1:.2f}")
    g = likelihood.gradient(starting_model)
    X2 = likelihood.misfit(starting_model - 0.1 * g)
    print(f"Misfit 2: {X2:.2f}")


@_pytest.mark.skipif(
    not installed, reason="Skipping test for which required packages are not installed."
)
def test_elasticFWI_sampling():
    likelihood = _hmc_tomography.Distributions.ElasticFullWaveform2D.create_default(
        4800, "tests/configurations/default_testing_configuration.ini",
    )
    likelihood.temperature = 100.0

    template = _numpy.ones((int(likelihood.dimensions / 3), 1))

    lower_vp = template * 1800
    lower_vs = template * 600
    lower_rho = template * 1300
    lower_bounds = _numpy.vstack((lower_vp, lower_vs, lower_rho))

    upper_vp = template * 2200
    upper_vs = template * 1000
    upper_rho = template * 1700
    upper_bounds = _numpy.vstack((upper_vp, upper_vs, upper_rho))

    prior = _hmc_tomography.Distributions.Uniform(lower_bounds, upper_bounds)

    posterior = _hmc_tomography.Distributions.BayesRule([prior, likelihood])

    filename = "temporary_file.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)

    _hmc_tomography.Samplers.HMC.sample(
        filename,
        posterior,
        proposals=10,
        ram_buffer_size=1,
        amount_of_steps=2,
        initial_model=(upper_bounds + lower_bounds) / 2.0,
        time_step=0.03,
    )

    # Remove the file
    _os.remove(filename)

