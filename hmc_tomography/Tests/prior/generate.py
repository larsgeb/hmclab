import traceback
import numpy
from termcolor import cprint

from hmc_tomography import Priors


def main(dimensions=50, indent=0):
    """

        Parameters
        ----------
        indent
        dimensions
        """
    generate_errors = 0
    prefix = indent * "\t"
    cprint(
        prefix
        + f"Starting generate test for all priors using\r\n"
        + prefix
        + f"{dimensions} dimensions...\r\n",
        "blue",
        attrs=["bold"],
    )
    for prior_class in Priors._AbstractPrior.__subclasses__():
        try:
            print(prefix + f"Prior name: {prior_class.__name__}")

            # Take care of special constructors which do not have the base pattern
            if prior_class == Priors.CompositePrior:
                # If constructing a composite prior

                # Use only one prior if dimension is 1
                if dimensions < 2:
                    list_of_priors = [Priors.Normal(dimensions)]
                else:  # Construct two priors if dimension is larger than 1
                    list_of_priors = [Priors.Normal(dimensions - 1), Priors.Uniform(1)]

                prior: Priors._AbstractPrior = Priors.CompositePrior(
                    dimensions, list_of_priors
                )

            else:
                prior: Priors._AbstractPrior = prior_class(dimensions)

            # Actual test ------------------------------------------------------
            result = prior.generate()
            assert type(result) == numpy.ndarray
            assert result.shape == (prior.dimensions, 1)
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
        except TypeError:
            cprint(
                prefix
                + f"Prior {prior_class.__name__} doesn't allow generation of "
                + "samples,\r\n"
                + prefix
                + f"won't fail test.\r\n",
                "yellow",
            )
        except Exception as e:
            generate_errors += 1
            cprint(
                prefix + f"Test unsuccessful for {prior.name}. Traceback with "
                "exception:",
                "red",
            )
            tb1 = traceback.TracebackException.from_exception(e)
            print("".join(tb1.format()), "\r\n")

    if generate_errors == 0:
        cprint(
            prefix + "All prior generate tests successful.\r\n", "green", attrs=["bold"]
        )
    else:
        cprint(
            prefix + "Not all prior generate tests successful.\r\n",
            "red",
            attrs=["bold"],
        )

    return generate_errors


if __name__ == "__main__":
    main()
