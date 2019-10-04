from termcolor import cprint
from hmc_tomography.Tests import mass_matrix
from hmc_tomography.Tests import target
from hmc_tomography.Tests import prior


def test_all():
    total_errors = 0
    total_errors += mass_matrix.test_all()
    total_errors += target.test_all()
    total_errors += prior.test_all()
    cprint(
        "Total errors: %i" % total_errors,
        "green" if total_errors == 0 else "red",
        attrs=["bold"],
    )
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    exit(test_all())
