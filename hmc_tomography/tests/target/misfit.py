import sys
import traceback
import numpy
from termcolor import cprint

sys.path.append("..")
from hmc_tomography import Targets


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
        + f"Starting misfit test for all targets using\r\n"
        + prefix
        + f"{dimensions} dimensions...\r\n",
        "blue",
        attrs=["bold"],
    )
    for target_class in Targets._AbstractTarget.__subclasses__():
        try:
            print(prefix + f"Target name: {target_class.__name__}")
            target: Targets._AbstractTarget = target_class(dimensions)

            # Actual test ------------------------------------------------------
            coordinates = numpy.ones((target.dimensions, 1))
            target.misfit(coordinates)
            # ------------------------------------------------------------------

            cprint(prefix + f"Test successful.\r\n", "green")
        except NotImplementedError:
            cprint(
                prefix
                + f"Target {target_class.__name__} not implemented"
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
            prefix + "All misfit tests successful.\r\n",
            "green",
            attrs=["bold"],
        )
    else:
        cprint(
            prefix + "Not all misfit tests successful.\r\n",
            "red",
            attrs=["bold"],
        )

    return exit_code


if __name__ == "__main__":
    main()
