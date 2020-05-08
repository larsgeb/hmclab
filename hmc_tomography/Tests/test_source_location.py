import hmc_tomography as _hmc_tomography
import numpy as _numpy
import numpy.matlib


SourceLocation = _hmc_tomography.Distributions.SourceLocation


def test_basic():

    stations_x = _numpy.array([-10.0, 5.0, 10.0])[None, :]
    stations_z = _numpy.array([0.0, 0.0, 0.0])[None, :]

    # x, z, T, v
    true_model = _numpy.array([0, 10, 0, 0, 12, 0, 2.5])[:, None]

    x, z, T, v = SourceLocation.split_vector(true_model)

    fake_observed_data = SourceLocation.forward(x, z, T, v, stations_x, stations_z)
    data_dispersion = _numpy.ones_like(fake_observed_data)

    likelihood = SourceLocation(
        stations_x, stations_z, fake_observed_data, data_dispersion, infer_velocity=True
    )

    print()
    print(likelihood.gradient(true_model))

    samples_filename = "samples_source_location.h5"

    # Creating the prior

    prior_event = _hmc_tomography.Distributions.Uniform([-20, 0, 0], [20, 25, 100])

    prior_velocity = _hmc_tomography.Distributions.Normal(2.5, 0.5)

    composite_prior = _hmc_tomography.Distributions.CompositeDistribution(
        [prior_event, prior_event, prior_velocity]
    )

    posterior = _hmc_tomography.Distributions.BayesRule([composite_prior, likelihood])

    step_size = 0.005 * _numpy.matlib.repmat(
        prior_event.upper_bounds - prior_event.lower_bounds, 2, 1
    )
    step_size = _numpy.append(step_size, [[0.25]], axis=0)

    _hmc_tomography.Samplers.HMC.sample(
        samples_filename,
        posterior,
        time_step=0.1,
        amount_of_steps=100,
        initial_model=true_model,
        proposals=1000,
        overwrite_existing_file=True,
    )

    with _hmc_tomography.Post.Samples(samples_filename) as samples:
        _hmc_tomography.Post.Visualization.marginal_grid(samples, [0, 1, 2, 3, 4, 5, 6])
