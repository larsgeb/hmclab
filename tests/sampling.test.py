"""
Test which encompasses all aspects of sampling.
"""
import sys
sys.path.append('..')
from hmc_tomography import Samplers

print("\r\nStarting sampling test ...\r\n")
sampler = Samplers.HMC('../tests/sampling.test.yml')

sampler.momentum[0] = 4.0
sampler.momentum[1] = 3.0

K = sampler.kinetic_energy(sampler.momentum)

print("Test successful.\r\n")
