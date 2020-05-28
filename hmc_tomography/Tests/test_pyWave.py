import os as _os

import numpy as _numpy
import matplotlib.pyplot as _plt
import pytest as _pytest

import hmc_tomography as _hmc_tomography


def test_pyWave_creation():
    dist = _hmc_tomography.Distributions.pyWave.create_default()

    dist.temperature = 1.0

    print(f"Free parameters: {dist.dimensions}")

    starting_model = dist.get_model_vector()

    X1 = dist.misfit(starting_model)
    print(f"Misfit 1: {X1:.2f}")
    g = dist.gradient(starting_model)
    X2 = dist.misfit(starting_model - 0.1 * g)
    print(f"Misfit 2: {X2:.2f}")


def test_pyWave_sampling():
    likelihood = _hmc_tomography.Distributions.pyWave.create_default()
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

    starting_model = likelihood.get_model_vector()

    _hmc_tomography.Samplers.HMC.sample(
        "samplesFWI",
        posterior,
        proposals=1000,
        ram_buffer_size=1,
        amount_of_steps=10,
        initial_model=starting_model,
        time_step=0.025,
        overwrite_existing_file=False,
    )
