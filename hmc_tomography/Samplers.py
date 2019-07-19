"""
Sampler classes and associated methods.
"""
import sys
import time
from abc import ABC, abstractmethod

import numpy
import tqdm as tqdm
import yaml
from typing import Tuple

from hmc_tomography.Priors import Prior
from hmc_tomography.MassMatrices import MassMatrix
from hmc_tomography.Targets import Target


class Sampler(ABC):

    name: str = "Monte Carlo sampler abstract base class"
    dimensions: int = -1
    online_thinning: int = 1
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
        online_thinning=1,
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
        self.online_thinning = online_thinning

        # Loading and parsing configuration ------------------------------------
        with open(config_file_path, "r") as config_file:
            cfg = yaml.load(config_file, Loader=yaml.FullLoader)
        if quiet is not True:
            print("Sections in configuration file:")
            for section in cfg:
                print("{:<20} {:>20} ".format(section, cfg[section]))

    def sample(
        self,
        proposals=100,
        iterations: int = 10,
        time_step: float = 0.1,
        online_thinning: int = None,
    ) -> int:
        """

        Parameters
        ----------
        online_thinning : object
        proposals
        iterations
        time_step

        Returns
        -------

        """

        # Prepare sampling -----------------------------------------------------
        accepted = 0
        coordinates = numpy.ones((self.dimensions, 1))
        self.samples = coordinates.copy()
        if online_thinning is None:
            online_thinning = self.online_thinning

        # Flush output (works best if followed by sleep() )
        sys.stdout.flush()
        time.sleep(0.001)

        # Create progress bar
        iterable = tqdm.trange(
            proposals, desc="Sampling. Acceptance rate:", leave=True
        )

        # Start sampling, but catch CTRL+C (SIGINT) and continue ---------------
        try:
            for proposal in iterable:
                # Compute initial Hamiltonian
                potential: float = self.target.misfit(
                    coordinates
                ) + self.prior.misfit(coordinates)
                momentum = self.mass_matrix.generate_momentum()
                kinetic: float = self.mass_matrix.kinetic_energy(momentum)
                hamiltonian: float = potential + kinetic

                # Propagate using a numerical integrator
                new_coordinates, new_momentum = self.propagate_leapfrog(
                    coordinates, momentum, iterations, time_step
                )

                # Compute resulting Hamiltonian
                new_potential: float = self.target.misfit(
                    new_coordinates
                ) + self.prior.misfit(new_coordinates)
                new_kinetic: float = self.mass_matrix.kinetic_energy(
                    new_momentum
                )
                new_hamiltonian: float = new_potential + new_kinetic

                # Evaluate acceptance criterion
                if numpy.exp(
                    hamiltonian - new_hamiltonian
                ) > numpy.random.uniform(0, 1):
                    accepted += 1
                    coordinates = new_coordinates.copy()
                else:
                    pass

                # On-line thinning
                if proposal % online_thinning == 0:
                    self.samples = numpy.append(
                        self.samples, coordinates, axis=1
                    )
                    iterable.set_description(
                        f"Average acceptance rate: {accepted/(proposal+1):.2f}."
                        "Progress"
                    )
        except KeyboardInterrupt:  # Catch SIGINT
            iterable.close()  # Close tqdm progressbar

        # Flush output
        sys.stdout.flush()
        time.sleep(0.001)

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
        # Make sure not to alter a view but a copy of the passed arrays.
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
