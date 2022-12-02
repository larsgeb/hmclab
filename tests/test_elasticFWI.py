import os as _os

import numpy as _numpy
import pytest as _pytest
import uuid as _uuid


import hmclab as _hmclab
from hmclab.Helpers.CustomExceptions import InvalidCaseError
import sys

installed = "psvWave" in sys.modules


@_pytest.mark.skipif(
    not installed, reason="Skipping test for which required packages are not installed."
)
def test_elasticFWI_creation():
    likelihood = _hmclab.Distributions.ElasticFullWaveform2D.create_default(
        4800,
        "tests/configurations/default_testing_configuration.ini",
    )

    # This should fail with a InvalidCaseError
    try:
        likelihood.generate()
    except InvalidCaseError:
        pass

    _hmclab.Distributions.ElasticFullWaveform2D(likelihood, temperature=2)

    # This should fail with a ValueError
    try:
        _hmclab.Distributions.ElasticFullWaveform2D(42)
    except ValueError as e:
        print(e)

    # This should fail with a ValueError
    try:
        _hmclab.Distributions.ElasticFullWaveform2D(
            "tests/configurations/default_testing_configuration.ini",
        )
    except ValueError as e:
        print(e)

    ux, uz = likelihood.fdModel.get_observed_data()
    _hmclab.Distributions.ElasticFullWaveform2D(
        "tests/configurations/default_testing_configuration.ini",
        ux_obs=ux,
        uz_obs=uz,
    )


@_pytest.mark.skipif(
    not installed, reason="Skipping test for which required packages are not installed."
)
def test_elasticFWI_gradient():
    likelihood = _hmclab.Distributions.ElasticFullWaveform2D.create_default(
        4800,
        "tests/configurations/default_testing_configuration.ini",
    )

    likelihood.temperature = 1.0

    print(f"Free parameters: {likelihood.dimensions}")

    starting_model = likelihood.get_model_vector()

    X1 = likelihood.misfit(starting_model)
    print(f"Misfit 1: {X1:.2f}")
    g = likelihood.gradient(starting_model)
    X2 = likelihood.misfit(starting_model - 0.1 * g)
    print(f"Misfit 2: {X2:.2f}")

    # This is to trigger the 'if self.forward_up_to_date' up to date line.
    X2 = likelihood.misfit(starting_model - 0.1 * g)


@_pytest.mark.skipif(
    not installed, reason="Skipping test for which required packages are not installed."
)
def test_elasticFWI_sampling():
    likelihood = _hmclab.Distributions.ElasticFullWaveform2D.create_default(
        4800,
        "tests/configurations/default_testing_configuration.ini",
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

    prior = _hmclab.Distributions.Uniform(lower_bounds, upper_bounds)

    posterior = _hmclab.Distributions.BayesRule([prior, likelihood])

    unique_name = _uuid.uuid4().hex.upper()
    filename = f"temporary_file_{unique_name}.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)  # pragma: no cover

    _hmclab.Samplers.HMC().sample(
        filename,
        posterior,
        proposals=10,
        amount_of_steps=2,
        initial_model=(upper_bounds + lower_bounds) / 2.0,
        stepsize=0.03,
    )

    # Remove the file
    _os.remove(filename)
