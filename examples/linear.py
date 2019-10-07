import hmc_tomography
import numpy

G = numpy.eye(200)
G[1, 0] = 0.95
G[0, 1] = 0.9

G *= 5

m_true = numpy.zeros((200,1))

d = G @ m_true

print(G.T @ G)

# true_m = numpy.linalg.inv(G) @ d

target = hmc_tomography.Targets.LinearMatrix(200, G=G, d=d)
prior = hmc_tomography.Priors.Normal(
    target.dimensions, means=-numpy.zeros((200, 1)), covariance=numpy.ones((200, 1))
)

prior = hmc_tomography.Priors.Sparse(200)

mass_matrix = hmc_tomography.MassMatrices.Unit(target.dimensions)


sampler = hmc_tomography.Samplers.HMC(target, mass_matrix, prior)

filename = "samples_linear_2.h5"

sampler.sample(
    filename,
    proposals=10000,
    online_thinning=1,
    sample_ram_buffer_size=20000,
    time_step=0.01,
    initial_model=-numpy.zeros((200, 1)),
)

samples = hmc_tomography.Post.Samples(filename, burn_in=250)

hmc_tomography.Post.Visualization.visualize_2_dimensions(samples, bins=200)
