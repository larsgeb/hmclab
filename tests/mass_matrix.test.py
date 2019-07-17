import sys
import numpy
import matplotlib.pyplot as pyplot

sys.path.append("..")
from hmc_tomography import MassMatrices


def main() -> int:
    """
    Unit test for kinetic energy

    Returns
    -------

    """
    print("\r\nStarting mass matrix test ...\r\n")

    dimensions = 10

    for mass_matrix_object in MassMatrices.MassMatrix.__subclasses__():

        if mass_matrix_object is MassMatrices.Diagonal:
            mass_matrix: MassMatrices = mass_matrix_object(
                dimensions, diagonal=10 ** numpy.random.randn(dimensions, 1)
            )
        elif mass_matrix_object is MassMatrices.LBFGS:
            break
        else:
            mass_matrix = mass_matrix_object(dimensions)

        print(f"Mass matrix name: {mass_matrix.full_name()}")

        k = []
        for i in range(100000):
            k.append(
                mass_matrix.kinetic_energy(mass_matrix.generate_momentum())
            )

        figure_diagonal = pyplot.figure(figsize=(8, 4))

        axis_matrix = figure_diagonal.add_axes([0.1, 0.2, 0.3, 0.6])
        axis_matrix_colourbar = figure_diagonal.add_axes(
            [0.41, 0.2, 0.025, 0.6]
        )
        axis_histogram_kinetic_energy = figure_diagonal.add_axes(
            [0.55, 0.2, 0.35, 0.6]
        )
        matrix_image = axis_matrix.imshow(mass_matrix.matrix)
        figure_diagonal.colorbar(
            matrix_image, cax=axis_matrix_colourbar, orientation="vertical"
        )
        axis_histogram_kinetic_energy.hist(k, 100)
        pyplot.show()

    print("\r\nTest successful.\r\n")

    return 0


exit(main())
