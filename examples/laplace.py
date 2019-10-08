import hmc_tomography
import numpy

G = numpy.array([[1, 0.5], [1, 0.5]])
d = G @ numpy.array([[0], [1]])

# target = hmc_tomography.Targets.LinearMatrix(2, G, d, data_covariance=1.0)
target = hmc_tomography.Targets.Empty(2)

# Create prior 1, a 1d uniform distribution on [0, 1]
prior = hmc_tomography.Priors.L05(
    2,
    lower_bounds=-6 * numpy.ones((2, 1)),
    upper_bounds=6 * numpy.ones((2, 1)),
    dispersion=1,
)

mass_matrix = hmc_tomography.MassMatrices.Unit(2)

sampler = hmc_tomography.Samplers.HMC(target, mass_matrix, prior)

filename = "samples_composite.h5"

# sampler.sample(filename, proposals=100000, online_thinning=1, time_step=0.5)

samples = hmc_tomography.Post.Samples(filename)

hmc_tomography.Post.Visualization.marginal_grid(samples, [0, 1,])
