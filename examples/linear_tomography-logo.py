import helper_xrayTomography as xrt
import helper_generate_rays as pth
import numpy
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import hmc_tomography


# Some quick settings ------------------------------------------------------------------
resample = False
plot_rays = False

# Set up the straight ray tomography problem -------------------------------------------

# Create a 'true model' (we can use realtive slowness, because the forward problem is
# linear)
logo = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, -1, -1, -1, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, -1, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, -1, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]
true_slowness = numpy.fliplr(numpy.array(logo).T) * 100
nx = true_slowness.shape[0]
ny = true_slowness.shape[1]

# Compute daomain size
xmin = 0.0
xmax = 30.0
ymin = 0.0
ymax = 18.0
extent = (xmin, xmax, ymin, ymax)
xlength = xmax - xmin
ylength = ymax - ymin

# The aspect ratio of the cells has to be 1
assert xlength / ylength == nx / ny

# Generate source-receiver setup randomly (but repeatedly)
paths = pth.generate_rays(extent, n_rays=500)

# Create forward model matrix and data
d, G = xrt.tracer(true_slowness, paths, extent=extent)

# Add some noise to the data
d += numpy.random.normal(loc=0, scale=50.0, size=d.shape)


# plt.title("True model")
# xrt.displayModel(
#     true_slowness, paths=paths, clim=(-200, 200), extent=extent, cmap=plt.cm.seismic
# )
# plt.show()
# exit(0)

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
model_recovered = scipy.sparse.linalg.lsmr(G, d, damp=2.5)[0]

# Bayesian recovery --------------------------------------------------------------------

# Target from G, d, and data covariance
target = hmc_tomography.Targets.LinearMatrix(
    true_slowness.size, G, d[:, None], 50.0 ** 2
)

# Laplace prior
prior_laplace = hmc_tomography.Priors.Sparse(
    true_slowness.size,
    lower_bounds=-200 * numpy.ones((true_slowness.size, 1)),
    upper_bounds=200 * numpy.ones((true_slowness.size, 1)),
    dispersion=1,
)

# Gaussian prior
prior_gauss = hmc_tomography.Priors.Normal(
    true_slowness.size,
    lower_bounds=-200 * numpy.ones((true_slowness.size, 1)),
    upper_bounds=200 * numpy.ones((true_slowness.size, 1)),
    means=numpy.zeros((true_slowness.size, 1)),
    covariance=5 * numpy.ones((true_slowness.size, 1)),
)

mass_matrix = hmc_tomography.MassMatrices.Unit(true_slowness.size)

# Sample the Laplace-conditioned posterior
sampler = hmc_tomography.Samplers.HMC(target, mass_matrix, prior_laplace)
filename = "samples_simple_linear_tomography_Laplace.h5"
if resample:
    sampler.sample(
        filename,
        proposals=100000,
        online_thinning=1,
        time_step=0.05,
        initial_model=100 * numpy.ones((true_slowness.size, 1)),
    )
samples_laplace = hmc_tomography.Post.Samples(filename, burn_in=5000)

# Sample the Normal-conditioned posterior
sampler = hmc_tomography.Samplers.HMC(target, mass_matrix, prior_gauss)
filename = "samples_simple_linear_tomography_Gaussian.h5"
if resample:
    sampler.sample(
        filename,
        proposals=100000,
        online_thinning=1,
        time_step=0.01,
        initial_model=100 * numpy.ones((true_slowness.size, 1)),
    )
samples_gaussian = hmc_tomography.Post.Samples(filename, burn_in=5000)


# Compute statistics
mean_laplace = numpy.mean(samples_laplace[:-1, :], axis=1)
mean_gaussian = numpy.mean(samples_gaussian[:-1, :], axis=1)

std_laplace = numpy.std(samples_laplace[:-1, :], axis=1)
std_gaussian = numpy.std(samples_gaussian[:-1, :], axis=1)

dimensions = [
    ny * 4 + 4,
    ny * 4 + 3,
    ny * 1 + 4,
    ny * 1 + 3,
    ny * 2 + 4,
    ny * 2 + 3,
    ny * 3 + 4,
    ny * 3 + 3,
]

hmc_tomography.Post.Visualization.marginal_grid(samples_laplace, dimensions)

# Plot results -------------------------------------------------------------------------
# plt.figure(figsize=(20, 5))
# plt.subplot(141)
# plt.title("True model")
# xrt.displayModel(
#     true_slowness,
#     paths=paths[::50],
#     clim=(-200, 200),
#     extent=extent,
#     cmap=plt.cm.seismic,
# )
# plt.subplot(142)
# plt.title("LSMR recovery")
# xrt.displayModel(
#     model_recovered.reshape([nx, ny]),
#     # paths=paths,
#     clim=(-200, 200),
#     extent=extent,
#     cmap=plt.cm.seismic,
# )

# plt.subplot(143)
# plt.title("Bayesian recovery\n(Laplace prior)")
# xrt.displayModel(
#     mean_laplace.reshape([nx, ny]),
#     # paths=paths,
#     clim=(-200, 200),
#     extent=extent,
#     cmap=plt.cm.seismic,
# )
# plt.subplot(144)
# plt.title("Bayesian recovery\n(Gaussian prior)")
# xrt.displayModel(
#     mean_gaussian.reshape([nx, ny]),
#     # paths=paths,
#     clim=(-200, 200),
#     extent=extent,
#     cmap=plt.cm.seismic,
# )
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 5))

# max_std = numpy.max([std_laplace.max(), std_gaussian.max()])

# plt.subplot(121)
# plt.title("Bayesian recovery\n(Laplace prior)")
# xrt.displayModel(
#     std_laplace.reshape([nx, ny]), clim=(0, max_std), extent=extent, cmap=plt.cm.seismic
# )
# plt.subplot(122)
# plt.title("Bayesian recovery\n(Gaussian prior)")
# xrt.displayModel(
#     std_gaussian.reshape([nx, ny]),
#     clim=(0, max_std),
#     extent=extent,
#     cmap=plt.cm.seismic,
# )
# plt.tight_layout()
# plt.show()

# Make a movie
# for i in range(500):
#     plt.figure(figsize=(10, 5))
#     plt.title("Bayesian recovery\n(Laplacian prior)")
#     xrt.displayModel(
#         samples_laplace[int(190*i)][:-1, None].reshape([nx, ny]),
#         clim=(-200, 200),
#         extent=extent,
#         cmap=plt.cm.seismic,
#     )
#     plt.savefig(f"laplace_movie/samples_{i:04d}.png")
#     plt.close()

# # Make another movie
# for i in range(500):
#     plt.figure(figsize=(10, 5))
#     plt.title("Bayesian recovery\n(Gaussian prior)")
#     xrt.displayModel(
#         samples_gaussian[int(190*i)][:-1, None].reshape([nx, ny]),
#         clim=(-200, 200),
#         extent=extent,
#         cmap=plt.cm.seismic,
#     )
#     plt.savefig(f"gaussian_movie/samples_{i:04d}.png")
#     plt.close()
