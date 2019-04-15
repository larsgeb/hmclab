import sys

sys.path.append('.')
from hmc_tomography import sampler


def main(sampler):
    print("Unit test on kinetic energy.")
    sampler = sampler.sampler('tests/kinetic_energy.test.yml', quiet=True)
    sampler.momentum[0] = 4.0
    sampler.momentum[1] = 3.0
    if (12.5 != sampler.kinetic_energy(sampler.momentum)):
        raise Exception("Failed unit test.")
    print("Test successful.")


main(sampler)
