import sys
import numpy

sys.path.append("..")
from hmc_tomography import Samplers, MassMatrices


def main():
    """
    Unit test for kinetic energy

    Returns
    -------

    """
    print("\r\nStarting mass matrix test ...\r\n")

    for mass_matrix_object in MassMatrices.MassMatrix.__subclasses__():

        if mass_matrix_object is MassMatrices.Diagonal:
            mass_matrix: MassMatrices = mass_matrix_object(
                10, diagonal=numpy.ones((10, 1))*30
            )
            print(numpy.var(mass_matrix.generate_momentum()))
        else:
            mass_matrix: MassMatrices = mass_matrix_object(10)

        print(f"Mass matrix name: {mass_matrix.full_name()}")


    print("\r\nTest successful.\r\n")

    return 0


exit(main())
