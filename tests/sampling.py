"""
Test which encompasses all aspects of sampling.
"""
import sys
import matplotlib.pyplot as pyplot
import numpy

sys.path.append("..")
from hmc_tomography import Samplers, Targets, MassMatrices, Priors
from mcmc_visualization import Processing

print("\r\nStarting sampling test ...\r\n")

# target = Targets.Himmelblau(annealing=100)
target = Targets.Empty(2)
diagonal = numpy.ones((2, 1))
mass_matrix = MassMatrices.Diagonal(target.dimensions, diagonal=diagonal)
# prior = Priors.UnboundedUniform(target.dimensions)

means = -1.0 * numpy.ones((target.dimensions, 1))
vars = 1.0 * numpy.ones((target.dimensions, 1))
means[0] = 0.5
means[1] = -0.5
vars[0] = 0.75
vars[1] = 1.8
prior = Priors.LogNormal(target.dimensions, means, vars)

sampler = Samplers.HMC("../tests/sampling.test.yml", target, mass_matrix, prior)

sampler.sample(time_step=0.01, proposals=500000, iterations=50)

figure_analysis = pyplot.figure(figsize=(16, 8))
axis_2d_histogram = figure_analysis.add_axes([0.025, 0.52, 0.2, 0.4])

axis_1d_histogram_x = figure_analysis.add_axes(
    [0.025, 0.08, 0.2, 0.4], sharex=axis_2d_histogram
)
axis_1d_histogram_y = figure_analysis.add_axes(
    [0.025 + 0.2 + 0.03, 0.52, 0.2, 0.4], sharey=axis_2d_histogram
)
axis_1d_traceplot = figure_analysis.add_axes(
    [
        0.025 + 0.2 + 0.03 + 0.2 + 0.03,
        0.52,
        1 - (0.025 + 0.2 + 0.03 + 0.2 + 0.03) - 0.025,
        0.4,
    ],
    sharey=axis_2d_histogram,
)
axis_autocorrelation = figure_analysis.add_axes(
    [
        0.025 + 0.2 + 0.03 + 0.2 + 0.03,
        0.08,
        1 - (0.025 + 0.2 + 0.03 + 0.2 + 0.03) - 0.025,
        0.4,
    ]
)

sampler.samples = numpy.log(sampler.samples)
print(numpy.mean(sampler.samples[0, :]))
print(numpy.median(sampler.samples[0, :]))
print(numpy.std(sampler.samples[0, :]))
print(numpy.var(sampler.samples[0, :]))
print(numpy.mean(sampler.samples[1, :]))
print(numpy.median(sampler.samples[1, :]))
print(numpy.std(sampler.samples[1, :]))
print(numpy.var(sampler.samples[1, :]))

axis_2d_histogram.hist2d(sampler.samples[0, :], sampler.samples[1, :], 100)
axis_1d_histogram_x.hist(sampler.samples[0, :], 100)
axis_1d_histogram_y.hist(sampler.samples[1, :], 100, orientation="horizontal")
axis_1d_traceplot.plot(sampler.samples[1, :], "--")
axis_1d_traceplot.set_xlim([0, sampler.samples[1, :].size])
axis_autocorrelation.plot(
    Processing.autocorrelation(sampler.samples[0, :]), "r", label="Dimension 0"
)
axis_autocorrelation.plot(
    Processing.autocorrelation(sampler.samples[1, :]), "k", label="Dimension 1"
)
axis_autocorrelation.legend()
pyplot.show()

print("Test successful.\r\n")
