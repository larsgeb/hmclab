"""
Main executable for now
"""
from hmc_tomography import sampler

sampler = sampler.sampler('example_configuration.yml')

sampler.momentum[0] = 4.0
sampler.momentum[1] = 3.0

K = sampler.kinetic_energy(sampler.momentum)

