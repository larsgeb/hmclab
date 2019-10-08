import helper_xrayTomography as xrt
import numpy
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import hmc_tomography


# Some quick settings ------------------------------------------------------------------
resample = True
plot_rays = False

# Set up the straight ray tomography problem -------------------------------------------

# Give model dimension in 2D
nx = 5
ny = 10

# Create a 'true model' (we can use realtive slowness, because the forward problem is
# linear)
true_slowness = numpy.zeros((nx, ny))
true_slowness[2, 2] = 100
true_slowness[0, 0] = 100

# Create source-receiver setup
paths = numpy.array(
    [
        [0, 0.1, 1.0, 0.1],  # x-src, y-src, x-rec, y-rec
        [0, 0.15, 0.9, 2.0],
        [0, 1.5, 1.0, 1.4],
        [0.3, 1.7, 0.3, 2.0],
        [0.7, 0.0, 0.7, 2.0],
    ]
)

# Compute daomain size
xmin = 0
xmax = 1
ymin = 0
ymax = 2
extent = (xmin, xmax, ymin, ymax)
xlength = xmax - xmin
ylength = ymax - ymin

# The aspect ratio of the cells has to be 1
assert xlength / ylength == nx / ny

# Create forward model matrix and data
d, G = xrt.tracer(true_slowness, paths, extent=extent)

# Add some noise to the data
d += numpy.random.normal(loc=0, scale=1.0, size=d.shape)

# Plotting paths -----------------------------------------------------------------------
if plot_rays:
    for i_path, path in enumerate(paths):
        xrt.displayModel(
            G[i_path, :].reshape([nx, ny]),
            cmap=plt.cm.hot_r,
            paths=[path],
            extent=extent,
        )
        plt.show()

# LSMR recovery ------------------------------------------------------------------------
model_recovered = scipy.sparse.linalg.lsmr(G, d, damp=0.1)[0]

# Bayesian recovery --------------------------------------------------------------------

# Target from G, d, and data covariance
target = hmc_tomography.Targets.LinearMatrix(
    true_slowness.size, G, d[:, None], 1.0 ** 2
)

# Laplace prior
prior_laplace = hmc_tomography.Priors.Sparse(
    true_slowness.size,
    lower_bounds=-200 * numpy.ones((true_slowness.size, 1)),
    upper_bounds=200 * numpy.ones((true_slowness.size, 1)),
    dispersion=2,
)

# Gaussian prior
prior_gauss = hmc_tomography.Priors.Normal(
    true_slowness.size,
    lower_bounds=-200 * numpy.ones((true_slowness.size, 1)),
    upper_bounds=200 * numpy.ones((true_slowness.size, 1)),
    means=numpy.zeros((true_slowness.size, 1)),
    covariance=2 * numpy.ones((true_slowness.size, 1)),
)

mass_matrix = hmc_tomography.MassMatrices.Unit(true_slowness.size)

# Sample the Laplace-conditioned posterior
sampler = hmc_tomography.Samplers.HMC(target, mass_matrix, prior_laplace)
filename = "samples_simple_linear_tomography_Laplace.h5"
if resample:
    sampler.sample(
        filename,
        proposals=10000,
        online_thinning=1,
        time_step=1.0,
        initial_model=numpy.ones((true_slowness.size, 1)),
    )
samples_laplace = hmc_tomography.Post.Samples(filename, burn_in=0)

# Sample the Normal-conditioned posterior
sampler = hmc_tomography.Samplers.HMC(target, mass_matrix, prior_gauss)
filename = "samples_simple_linear_tomography_Gaussian.h5"
if resample:
    sampler.sample(
        filename,
        proposals=10000,
        online_thinning=1,
        time_step=1.0,
        initial_model=numpy.ones((true_slowness.size, 1)),
    )
samples_gaussian = hmc_tomography.Post.Samples(filename, burn_in=0)


# Compute statistics
mean_laplace = numpy.mean(samples_laplace[:-1, :], axis=1)
mean_gaussian = numpy.mean(samples_gaussian[:-1, :], axis=1)

# Plot results -------------------------------------------------------------------------
plt.figure(figsize=(16, 5))
plt.subplot(141)
plt.title("True model")
xrt.displayModel(
    true_slowness, paths=paths, clim=(-200, 200), extent=extent, cmap=plt.cm.seismic
)
plt.subplot(142)
plt.title("LSMR recovery")
xrt.displayModel(
    model_recovered.reshape([nx, ny]),
    paths=paths,
    clim=(-200, 200),
    extent=extent,
    cmap=plt.cm.seismic,
)

plt.subplot(143)
plt.title("Bayesian recovery\n(Laplace prior)")
xrt.displayModel(
    mean_laplace.reshape([nx, ny]),
    paths=paths,
    clim=(-200, 200),
    extent=extent,
    cmap=plt.cm.seismic,
)
plt.subplot(144)
plt.title("Bayesian recovery\n(Gaussian prior)")
xrt.displayModel(
    mean_gaussian.reshape([nx, ny]),
    paths=paths,
    clim=(-200, 200),
    extent=extent,
    cmap=plt.cm.seismic,
)
plt.tight_layout()
plt.show()
