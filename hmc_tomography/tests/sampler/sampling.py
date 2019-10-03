"""
Test which encompasses all aspects of sampling.
"""
import numpy

from hmc_tomography import Samplers, Targets, MassMatrices, Priors

raise NotImplementedError("This test is not finished yet")

print("\r\nStarting sampling test ...\r\n")

# target = Targets.Himmelblau(annealing=100)
target = Targets.Empty(100)
diagonal = numpy.ones((target.dimensions, 1))
mass_matrix = MassMatrices.Diagonal(target.dimensions, diagonal=diagonal)

# Create prior
means = -1.0 * numpy.ones((target.dimensions, 1))
vars = numpy.ones((target.dimensions, 1))
prior = Priors.Normal(target.dimensions, means, vars)

# Create sampler
sampler = Samplers.HMC(target, mass_matrix, prior)

# Sample
sampler.sample(
    "samples.hdf5",
    time_step=0.1,
    proposals=10001,
    integration_steps=10,
    online_thinning=1,
    randomize_integration_steps=True,
    randomize_time_step=True,
    sample_ram_buffer_size=234,
)

# Visualize
# figure, _ = Visualization.visualize_2_dimensions(
#     sampler.samples, 0, 1, bins=15
# )
# pyplot.show()
exit(0)
