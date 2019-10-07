import hmc_tomography
import numpy

target = hmc_tomography.Targets.Empty(2)

# Create prior 1, a 1d uniform distribution on [0, 1]
prior_1 = hmc_tomography.Priors.Uniform(
    1, lower_bounds=numpy.zeros((1, 1)), upper_bounds=numpy.ones((1, 1))
)

# Create prior 1, a 1d normal distribution with mu = 0, sigma = 1
prior_2 = hmc_tomography.Priors.Normal(
    1,
    means=numpy.zeros((1, 1)),
    covariance=numpy.ones((1, 1)),
    upper_bounds=numpy.ones((1, 1)),
)

prior = hmc_tomography.Priors.CompositePrior(2, [prior_1, prior_2])

mass_matrix = hmc_tomography.MassMatrices.Unit(2)

sampler = hmc_tomography.Samplers.HMC(target, mass_matrix, prior)

filename = "samples_composite.h5"


sampler.sample(filename, proposals=50000, online_thinning=1, time_step=1.0)

samples = hmc_tomography.Post.Samples(filename)

hmc_tomography.Post.Visualization.visualize_2_dimensions(samples, bins=25, show=True)
