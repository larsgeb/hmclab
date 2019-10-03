from termcolor import cprint
from hmc_tomography.tests.prior import misfit, gradient, generate


def test_all(dimensions=50, indent=0):
    prefix = indent * "\t"

    # Prior tests --------------------------------------------------------------
    prior_errors = 0
    cprint(prefix + "Running all prior tests.", "blue", attrs=["bold"])

    # Tests
    prior_errors += misfit.main(dimensions=dimensions, indent=indent + 1)
    prior_errors += gradient.main(dimensions=dimensions, indent=indent + 1)
    prior_errors += generate.main(dimensions=dimensions, indent=indent + 1)

    if prior_errors == 0:
        cprint(prefix + "All prior successful.", "green", attrs=["bold"])
    else:
        cprint(prefix + "Not all prior successful.", "red", attrs=["bold"])

    return prior_errors


if __name__ == "__main__":
    exit(test_all())
