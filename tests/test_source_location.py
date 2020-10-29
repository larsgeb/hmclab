import hmc_tomography as _hmc_tomography
import numpy as _numpy
import numpy.matlib
import os as _os

# from line_profiler import LineProfiler

SourceLocation = _hmc_tomography.Distributions.SourceLocation

# profile = LineProfiler()
# profile.add_function(_hmc_tomography.Distributions.SourceLocation.gradient)
# profile.add_function(_hmc_tomography.Distributions.SourceLocation.forward)
# profile.add_function(_hmc_tomography.Distributions.SourceLocation.forward_gradient)


# @profile
def test_basic():

    stations_x = _numpy.array([-10.0, 5.0, 10.0])[None, :]
    stations_z = _numpy.array([0.0, 0.0, 0.0])[None, :]

    # Create the true model and observations -------------------------------------------

    true_model = _numpy.array([0, 10, 0, 0, 12, 0, 2.5])[:, None]

    x, z, T, v = SourceLocation.split_vector(true_model)

    fake_observed_data = SourceLocation.forward(x, z, T, v, stations_x, stations_z)

    # Create the likelihood ------------------------------------------------------------

    data_dispersion = 0.25 * _numpy.ones_like(fake_observed_data)

    likelihood = SourceLocation(
        stations_x, stations_z, fake_observed_data, data_dispersion, infer_velocity=True
    )

    # Creating the prior ---------------------------------------------------------------

    prior_event = _hmc_tomography.Distributions.Uniform([-20, 0, 0], [20, 25, 100])

    prior_velocity = _hmc_tomography.Distributions.Normal(2.5, 2.5)

    composite_prior = _hmc_tomography.Distributions.CompositeDistribution(
        [prior_event, prior_event, prior_velocity]
    )

    # Applying Bayes rule --------------------------------------------------------------

    posterior = _hmc_tomography.Distributions.BayesRule([composite_prior, likelihood])

    # Sampling -------------------------------------------------------------------------

    filename = "temporary_file.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)

    _hmc_tomography.Samplers.RWMH().sample(
        filename,
        posterior,
        initial_model=true_model,
        proposals=1000,
        stepsize=0.1,
        online_thinning=1,
        overwrite_existing_file=True,
    )

    _os.remove(filename)
