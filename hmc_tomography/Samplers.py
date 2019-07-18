"""
Sampler classes and associated methods.
"""
from abc import ABC, abstractmethod

import numpy
import yaml
from typing import Tuple

from hmc_tomography.Priors import Prior
from hmc_tomography.MassMatrices import MassMatrix
from hmc_tomography.Targets import Target


class Sampler(ABC):

    name: str = "Monte Carlo sampler abstract base class"
    dimensions: int = -1
    prior: Prior
    target: Target

    @abstractmethod
    def sample(self):
        """

        """
        pass


class HMC(Sampler):
    """Hamiltonian Monte Carlo class.

    """

    name = "Hamiltonian Monte Carlo sampler"
    mass_matrix: MassMatrix

    def __init__(
        self,
        config_file_path: str,
        target: Target,
        mass_matrix: MassMatrix,
        prior: Prior,
        quiet: bool = False,
    ):
        """

        Parameters
        ----------
        config_file_path
        target
        mass_matrix
        prior
        quiet
        """
        # Sanity check on the dimensions of passed objects ---------------------
        if not (
            target.dimensions == prior.dimensions == mass_matrix.dimensions
        ):
            raise ValueError(
                "Incompatible target/prior/mass matrix.\r\n"
                f"Target dimensions:\t\t{target.dimensions}.\r\n"
                f"Prior dimensions:\t\t{prior.dimensions}.\r\n"
                f"Mass matrix dimensions:\t{mass_matrix.dimensions}.\r\n"
            )

        # Setting the passed objects -------------------------------------------
        self.dimensions = target.dimensions
        self.prior = prior
        self.mass_matrix = mass_matrix
        self.target = target

        # Loading and parsing configuration ------------------------------------
        with open(config_file_path, "r") as config_file:
            cfg = yaml.load(config_file, Loader=yaml.FullLoader)
        if quiet is not True:
            print("Sections in configuration file:")
            for section in cfg:
                print("{:<20} {:>20} ".format(section, cfg[section]))

    def sample(
        self, proposals=100, iterations: int = 10, time_step: float = 0.1
    ) -> int:
        """

        Parameters
        ----------
        proposals
        iterations
        time_step

        Returns
        -------

        """
        accepted = 0
        coordinates = numpy.ones((self.dimensions, 1))
        self.samples = coordinates.copy()

        for proposal in range(proposals):
            # Compute initial Hamiltonian --------------------------------------
            potential: float = self.target.misfit(
                coordinates
            ) + self.prior.misfit(coordinates)
            momentum = self.mass_matrix.generate_momentum()
            kinetic: float = self.mass_matrix.kinetic_energy(momentum)
            hamiltonian: float = potential + kinetic

            # Propagate using a numerical integrator ---------------------------
            new_coordinates, new_momentum = self.propagate_leapfrog(
                coordinates, momentum, iterations, time_step
            )

            # Compute resulting Hamiltonian ------------------------------------
            new_potential: float = self.target.misfit(
                new_coordinates
            ) + self.prior.misfit(new_coordinates)
            new_kinetic: float = self.mass_matrix.kinetic_energy(new_momentum)
            new_hamiltonian: float = new_potential + new_kinetic

            # Print results ----------------------------------------------------
            # print(
            #     f"""
            #     Initial Hamiltonian: \t{hamiltonian:.3f}
            #     New Hamiltonian: \t\t{new_hamiltonian:.3f}
            #     """
            # )

            # Evaluate acceptance criterion ------------------------------------
            if numpy.exp(hamiltonian - new_hamiltonian) > numpy.random.uniform(
                0, 1
            ):
                accepted += 1
                coordinates = new_coordinates.copy()
                # print("Accepted")
            else:
                # print("Rejected")
                pass

            # Append new state -------------------------------------------------
            self.samples = numpy.append(self.samples, coordinates, axis=1)

        print(f"Accepted proposals: {accepted}")
        return 0

    def propagate_leapfrog(
        self,
        coordinates: numpy.ndarray,
        momentum: numpy.ndarray,
        iterations: int,
        time_step: float,
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """

        Parameters
        ----------
        coordinates
        momentum
        iterations
        time_step

        Returns
        -------

        """
        coordinates = coordinates.copy()
        momentum = momentum.copy()

        # Leapfrog integration -------------------------------------------------
        coordinates += (
            0.5 * time_step * self.mass_matrix.kinetic_energy_gradient(momentum)
        )
        for i in range(iterations - 1):
            potential_gradient = self.target.gradient(
                coordinates
            ) + self.prior.gradient(coordinates)
            momentum -= time_step * potential_gradient
            coordinates += time_step * self.mass_matrix.kinetic_energy_gradient(
                momentum
            )
        potential_gradient = self.target.gradient(
            coordinates
        ) + self.prior.gradient(coordinates)
        momentum -= time_step * potential_gradient
        coordinates += (
            0.5 * time_step * self.mass_matrix.kinetic_energy_gradient(momentum)
        )

        return coordinates, momentum
