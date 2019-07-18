import sys

sys.path.append("..")

import tests.mass_matrix.test_all as mass_matrix_test
import tests.target.test_all as target_test
import tests.prior.test_all as prior_test


def test_all():
    mass_matrix_test.test_all()
    target_test.test_all()
    prior_test.test_all()


if __name__ == "__main__":
    exit(test_all())
