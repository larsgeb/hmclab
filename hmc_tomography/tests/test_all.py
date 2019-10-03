from termcolor import cprint
import hmc_tomography.tests.mass_matrix.test_all as mass_matrix_test
import hmc_tomography.tests.target.test_all as target_test
import hmc_tomography.tests.prior.test_all as prior_test


def test_all():
    total_errors = 0
    total_errors += mass_matrix_test.test_all()
    total_errors += target_test.test_all()
    total_errors += prior_test.test_all()
    cprint(
        "Total errors: %i" % total_errors,
        "green" if total_errors == 0 else "red",
        attrs=["bold"],
    )
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    exit(test_all())
