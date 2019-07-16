import sys
sys.path.append('..')
from hmc_tomography import Samplers


def main(sampler):
    """
    Unit test for kinetic energy
    Parameters
    ----------
    sampler

    Returns
    -------

    """
    print("\r\nStarting kinetic energy test ...\r\n")
    sampler = sampler('../tests/kinetic_energy.test.yml', quiet=True)
    sampler.momentum[0] = 4.0
    sampler.momentum[1] = 3.0
    if 12.5 != sampler.kinetic_energy(sampler.momentum):
        raise Exception("Failed unit test.")
    print("Test successful.\r\n")


main(Samplers.HMC)
