import os as _os

import numpy as _numpy
import matplotlib.pyplot as _plt
import pytest as _pytest

import hmc_tomography as _hmc_tomography


def test_pyWave_creation():
    dist = _hmc_tomography.Distributions.pyWave.create_default()

    dist.temperature = 10.0

    print(f"Free parameters: {dist.dimensions}")

    starting_model = dist.get_model_vector()

    X1 = dist.misfit(starting_model)
    print(f"Misfit 1: {X1:.2f}")
    g = dist.gradient(starting_model)
    X2 = dist.misfit(starting_model - 0.1 * g)
    print(f"Misfit 2: {X2:.2f}")


def test_pyWave_sampling():
    dist = _hmc_tomography.Distributions.pyWave.create_default()
    dist.temperature = 10.0

    starting_model = dist.get_model_vector()

    _hmc_tomography.Samplers.HMC.sample(
        "samplesFWI",
        dist,
        amount_of_steps=2,
        initial_model=starting_model,
        time_step=0.01,
    )
