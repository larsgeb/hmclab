from SalvusWrap.wrapper import SimpleWrap
from hmc_tomography import MassMatrices, Priors, SalvusTarget, Samplers
import numpy as np
import salvus_mesh
import os

target = SalvusTarget.Salvus(
    "wrap_sample_test.toml", annealing=0.1 ** 0.5
)

prior = Priors.Normal(
    target.dimensions,
    target.get_model_vector(),
    np.ones_like(target.get_model_vector()) * (10 ** 2),
    target.get_model_vector() - 1000,
    target.get_model_vector() + 1000,
)

mass = MassMatrices.Unit(target.dimensions)

sampler = Samplers.HMC(target, mass, prior)

samples_filename = "samples.h5"
proposals = 100
online_thinning = 1
sample_ram_buffer_size = 4
integration_steps = 10
time_step = 0.05
randomize_integration_steps = True
randomize_time_step = True

sampler.sample(
    samples_filename,
    proposals,
    online_thinning,
    sample_ram_buffer_size,
    integration_steps,
    time_step,
    randomize_integration_steps,
    randomize_time_step,
    initial_model=target.get_model_vector(),
)
