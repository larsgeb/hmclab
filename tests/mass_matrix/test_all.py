import sys
from termcolor import cprint

sys.path.append("..")
from tests.mass_matrix import (
    kinetic_energy,
    kinetic_energy_gradient,
    generate_momentum,
)


def test_all(dimensions=50, indent=0):
    prefix = indent * "\t"

    # Mass matrix tests --------------------------------------------------------
    mass_matrix_errors = 0
    cprint(prefix + "Running all mass matrix tests.", "blue", attrs=["bold"])

    # Tests
    mass_matrix_errors += kinetic_energy.main(
        dimensions=dimensions, indent=indent + 1
    )
    mass_matrix_errors += kinetic_energy_gradient.main(
        dimensions=dimensions, indent=indent + 1
    )
    mass_matrix_errors += generate_momentum.main(
        dimensions=dimensions, indent=indent + 1
    )

    if mass_matrix_errors == 0:
        cprint(
            prefix + "All mass matrix tests successful.",
            "green",
            attrs=["bold"],
        )
    else:
        cprint(
            prefix + "Not all mass matrix tests successful.",
            "red",
            attrs=["bold"],
        )

    if mass_matrix_errors == 0:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(test_all())
