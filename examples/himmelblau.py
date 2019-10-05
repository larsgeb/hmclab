import hmc_tomography

target = hmc_tomography.Targets.Himmelblau(annealing=100)
prior = hmc_tomography.Priors.UnboundedUniform(target.dimensions)
mass_matrix = hmc_tomography.MassMatrices.Unit(target.dimensions)

sampler = hmc_tomography.Samplers.HMC(target, mass_matrix, prior)

filename = "samples_himmelblau.h5"

sampler.sample(filename, proposals=10000, online_thinning=1, time_step=1.1)

samples = hmc_tomography.Post.Samples(filename)

hmc_tomography.Post.Visualization.visualize_2_dimensions(samples, bins=50, show=True)
