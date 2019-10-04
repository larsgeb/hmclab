from termcolor import cprint
from hmc_tomography.Tests.target import misfit, gradient


def test_all(dimensions=50, indent=0):
    prefix = indent * "\t"

    # Target tests -------------------------------------------------------------
    target_errors = 0
    cprint(prefix + "Running all target tests.", "blue", attrs=["bold"])

    # Tests
    target_errors += misfit.main(dimensions=dimensions, indent=indent + 1)
    target_errors += gradient.main(dimensions=dimensions, indent=indent + 1)

    if target_errors == 0:
        cprint(prefix + "All target tests successful.", "green", attrs=["bold"])
    else:
        cprint(prefix + "Not all target tests successful.", "red", attrs=["bold"])

    return target_errors


if __name__ == "__main__":
    exit(test_all())
