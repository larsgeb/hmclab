"""
Test which encompasses all aspects of sampling.
"""
import sys
import matplotlib.pyplot as pyplot
import numpy

sys.path.append("..")
from hmc_tomography import Samplers, Targets, MassMatrices, Priors
from post_processing import Visualization

print("\r\nStarting sampling test ...\r\n")

# target = Targets.Himmelblau(annealing=100)
target = Targets.Empty(2)
diagonal = numpy.ones((target.dimensions, 1))
mass_matrix = MassMatrices.Diagonal(target.dimensions, diagonal=diagonal)

# Create prior
means = -1.0 * numpy.ones((target.dimensions, 1))
vars = 1.0 * numpy.ones((target.dimensions, target.dimensions))
means[0] = 0.0
means[1] = 0.0
vars[0, 0] = 1
vars[1, 1] = 1
vars[1, 0] = 0.9
vars[0, 1] = 0.9
prior = Priors.Normal(target.dimensions, means, vars, lower_bounds=means)

# Create sampler
sampler = Samplers.HMC("../tests/sampling.test.yml", target, mass_matrix, prior)

# Sample
sampler.sample(
    time_step=0.5, proposals=100000, iterations=15, online_thinning=1
)

# Visualize
figure, _ = Visualization.visualize_2_dimensions(sampler.samples, 0, 1)
pyplot.show()
