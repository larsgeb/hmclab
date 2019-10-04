import traceback
import numpy
from termcolor import cprint

from hmc_tomography import MassMatrices


def main(dimensions=50, indent=0):
    """

    Parameters
    ----------
    indent
    dimensions
    """
    generate_momentum_errors = 0
    prefix = indent * "\t"
    cprint(
        prefix
        + f"Starting generate_momentum test for all mass matrices with\r\n"
        + prefix
        + f"{dimensions} dimensions...\r\n",
        "blue",
        attrs=["bold"],
    )
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
            mass_matrix.generate_momentum()
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
            generate_momentum_errors += 1
            cprint(prefix + f"Test unsuccessful. Traceback with exception:", "red")
            tb1 = traceback.TracebackException.from_exception(e)
            print("".join(tb1.format()), "\r\n")

    if generate_momentum_errors == 0:
        cprint(
            prefix + "All generate_momentum tests successful.\r\n",
            "green",
            attrs=["bold"],
        )
    else:
        cprint(
            prefix + "Not all generate_momentum tests successful.\r\n",
            "red",
            attrs=["bold"],
        )

    return generate_momentum_errors


if __name__ == "__main__":
    main()
