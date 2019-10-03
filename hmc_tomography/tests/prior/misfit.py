import sys
import traceback
import numpy
from termcolor import cprint

sys.path.append("..")
from hmc_tomography import Priors


def main(dimensions=50, indent=0):
    """

        Parameters
        ----------
        indent
        dimensions
        """
    misfit_errors = 0
    prefix = indent * "\t"
    cprint(
        prefix
        + f"Starting misfit test for all priors using\r\n"
        + prefix
        + f"{dimensions} dimensions...\r\n",
        "blue",
        attrs=["bold"],
    )
    for prior_class in Priors._AbstractPrior.__subclasses__():
        try:
            print(prefix + f"Prior name: {prior_class.__name__}")

            prior: Priors._AbstractPrior = prior_class(dimensions)

            # Actual test ------------------------------------------------------
            coordinates = numpy.ones((dimensions, 1))
            result = prior.misfit(coordinates)
            assert type(result) == float
            # ------------------------------------------------------------------

            cprint(prefix + f"Test successful.\r\n", "green")
        except NotImplementedError:
            cprint(
                prefix
                + f"Prior {prior_class.__name__} not implemented"
                + "\r\n"
                + prefix
                + f"yet, won't fail test.\r\n",
                "yellow",
            )
        except Exception as e:
            misfit_errors += 1
            cprint(
                prefix + f"Test unsuccessful for {prior.name}. Traceback with "
                "exception:",
                "red",
            )
            tb1 = traceback.TracebackException.from_exception(e)
            print("".join(tb1.format()), "\r\n")

    if misfit_errors == 0:
        cprint(
            prefix + "All prior misfit tests successful.\r\n", "green", attrs=["bold"]
        )
    else:
        cprint(
            prefix + "Not all prior misfit tests successful.\r\n", "red", attrs=["bold"]
        )

    return misfit_errors


if __name__ == "__main__":
    main()
