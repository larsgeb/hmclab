import sys
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
    exit_code = 0
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
            mass_matrix: MassMatrices._AbstractMassMatrix = mass_matrix_class(dimensions)

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
            exit_code = 1
            cprint(
                prefix + f"Test unsuccessful. Traceback with exception:", "red"
            )
            tb1 = traceback.TracebackException.from_exception(e)
            print("".join(tb1.format()), "\r\n")

    if exit_code == 0:
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

    return exit_code


if __name__ == "__main__":
    main()
