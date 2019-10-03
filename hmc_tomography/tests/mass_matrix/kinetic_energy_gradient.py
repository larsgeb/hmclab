import sys
import numpy
import traceback
from termcolor import cprint

sys.path.append("..")
from hmc_tomography import MassMatrices


def main(dimensions=50, indent=0):
    """

    Parameters
    ----------
    indent
    dimensions
    """
    kinetic_energy_gradient_errors = 0
    prefix = indent * "\t"
    cprint(
        prefix
        + f"Starting kinetic_energy_gradient test for all mass matrices with\r\n"
        + prefix
        + f"{dimensions} dimensions...\r\n",
        "blue",
        attrs=["bold"],
    )
    momentum = numpy.ones((dimensions, 1))
    for mass_matrix_class in MassMatrices._AbstractMassMatrix.__subclasses__():
        try:
            print(prefix + f"Mass matrix name: {mass_matrix_class.__name__}")

            if mass_matrix_class == MassMatrices.LBFGS:
                mass_matrix: MassMatrices._AbstractMassMatrix = MassMatrices.LBFGS(
                    dimensions,
                    10,
                    numpy.zeros((dimensions, 1)),
                    numpy.ones((dimensions, 1)),
                    1e-2,
                )
            else:
                mass_matrix: MassMatrices._AbstractMassMatrix = mass_matrix_class(
                    dimensions
                )

            # Actual test ------------------------------------------------------
            mass_matrix.kinetic_energy_gradient(momentum)
            # ------------------------------------------------------------------

            cprint(prefix + f"Test successful.\r\n", "green")
        except NotImplementedError:
            cprint(
                prefix
                + f"Mass matrix {mass_matrix_class.__name__} not implemented"
                + "\r\n"
                + prefix
                + f"yet, won't fail test.\r\n",
                "yellow",
            )
        except Exception as e:
            kinetic_energy_gradient_errors += 1
            cprint(prefix + f"Test unsuccessful. Traceback with exception:", "red")
            tb1 = traceback.TracebackException.from_exception(e)
            print("".join(tb1.format()), "\r\n")

    if kinetic_energy_gradient_errors == 0:
        cprint(
            prefix + "All kinetic_energy_gradient tests successful.\r\n",
            "green",
            attrs=["bold"],
        )
    else:
        cprint(
            prefix + "Not all kinetic_energy_gradient tests successful.\r\n",
            "red",
            attrs=["bold"],
        )

    return kinetic_energy_gradient_errors


if __name__ == "__main__":
    main()
