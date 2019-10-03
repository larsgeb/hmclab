"""
Sampler classes and associated methods.
"""
import sys as _sys
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod

import h5py as _h5py
import numpy as _numpy
import time as _time
import tqdm as _tqdm
from typing import Tuple as _Tuple

from hmc_tomography.Priors import _AbstractPrior
from hmc_tomography.MassMatrices import _AbstractMassMatrix
from hmc_tomography.Targets import _AbstractTarget


class _AbstractSampler(_ABC):
    """Monte Carlo Sampler base class

    """

    name: str = "Monte Carlo sampler abstract base class"
    dimensions: int = -1
    prior: _AbstractPrior
    target: _AbstractTarget
    sample_hdf5_file = None
    sample_hdf5_dataset = None
    sample_ram_buffer: _numpy.ndarray

    @_abstractmethod
    def sample(
        self,
        samples_filename: str,
        proposals: int = 100,
        online_thinning: int = 1,
        sample_ram_buffer_size: int = 1000,
    ) -> int:
        """
        Parameters
        ----------
        proposals
        online_thinning
        sample_ram_buffer_size
        samples_filename

        """
        pass


class HMC(_AbstractSampler):
    """Hamiltonian Monte Carlo class.

    """

    name = "Hamiltonian Monte Carlo sampler"
    mass_matrix: _AbstractMassMatrix

    def __init__(
        self,
        target: _AbstractTarget,
        mass_matrix: _AbstractMassMatrix,
        prior: _AbstractPrior,
    ):
        """

        Parameters
        ----------
        target
        mass_matrix
        prior
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

        self.mass_matrix_update_hook = getattr(self.mass_matrix, "update", None)

    def sample(
        self,
        samples_filename: str,
        proposals: int = 100,
        online_thinning: int = 1,
        sample_ram_buffer_size: int = 1000,
        integration_steps: int = 10,
        time_step: float = 0.1,
        randomize_integration_steps: bool = True,
        randomize_time_step: bool = True,
        initial_model: _numpy.ndarray = None,
        ignore_update_hook_mass: bool = False,
    ) -> int:
        """

        Parameters
        ----------
        ignore_update_hook_mass
        initial_model
        samples_filename
        proposals
        online_thinning
        sample_ram_buffer_size
        integration_steps
        time_step
        randomize_integration_steps
        randomize_time_step

        Returns
        -------

        """

        # Prepare sampling -----------------------------------------------------
        accepted = 0

        # Initial model
        if initial_model is None:
            coordinates = _numpy.ones((self.dimensions, 1))
        else:
            assert initial_model.shape == (self.dimensions, 1)
            coordinates = initial_model

        # Create RAM buffer for samples
        self.sample_ram_buffer = _numpy.empty(
            (self.dimensions + 1, sample_ram_buffer_size)
        )

        # Create or open HDF5 file
        total_samples_to_be_generated = int((1.0 / online_thinning) * proposals)
        self.open_hdf5(samples_filename, total_samples_to_be_generated)

        # Set attributes of the dataset to correspond to sampling settings
        self.sample_hdf5_dataset.attrs["proposals"] = proposals
        self.sample_hdf5_dataset.attrs["online_thinning"] = online_thinning
        self.sample_hdf5_dataset.attrs["time_step"] = time_step
        self.sample_hdf5_dataset.attrs["integration_steps"] = integration_steps
        self.sample_hdf5_dataset.attrs[
            "randomizetime_step"
        ] = randomize_time_step
        self.sample_hdf5_dataset.attrs[
            "randomize_iterations"
        ] = randomize_integration_steps

        # Flush output (works best if followed by sleep() )
        _sys.stdout.flush()
        _time.sleep(0.001)

        # Create progress bar
        proposals_total = _tqdm.trange(
            proposals, desc="Sampling. Acceptance rate:", leave=True
        )

        # Selection of integrator ----------------------------------------------
        propagate = self.propagate_leapfrog

        # Optional randomization -----------------------------------------------
        if randomize_integration_steps:

            def _iterations():
                return int(integration_steps * (0.5 + _numpy.random.rand()))

        else:

            def _iterations():
                return integration_steps

        if randomize_time_step:

            def _time_step():
                return float(time_step * (0.5 + _numpy.random.rand()))

        else:

            def _time_step():
                return time_step

        # Start sampling, but catch CTRL+C (SIGINT) ----------------------------
        try:
            for proposal in proposals_total:

                # Compute initial Hamiltonian
                potential: float = self.target.misfit(
                    coordinates
                ) + self.prior.misfit(coordinates)
                momentum = self.mass_matrix.generate_momentum()
                kinetic: float = self.mass_matrix.kinetic_energy(momentum)
                hamiltonian: float = potential + kinetic

                # Propagate using the numerical integrator
                new_coordinates, new_momentum, update_m, update_g = propagate(
                    coordinates, momentum, _iterations(), _time_step()
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
                if _numpy.exp(
                    hamiltonian - new_hamiltonian
                ) > _numpy.random.uniform(0, 1):
                    accepted += 1
                    coordinates = new_coordinates.copy()
                else:
                    pass

                if (
                    callable(self.mass_matrix_update_hook)
                    and not ignore_update_hook_mass
                ):
                    self.mass_matrix.update(update_m, update_g)

                # On-line thinning and  writing samples to disk ----------------
                if proposal % online_thinning == 0:
                    buffer_location: int = int(
                        (proposal / online_thinning) % sample_ram_buffer_size
                    )
                    self.sample_ram_buffer[:-1, buffer_location] = coordinates[
                        :, 0
                    ]
                    self.sample_ram_buffer[
                        -1, buffer_location
                    ] = self.target.misfit(coordinates) + self.prior.misfit(
                        coordinates
                    )
                    proposals_total.set_description(
                        f"Average acceptance rate: {accepted/(proposal+1):.2f}."
                        "Progress"
                    )
                    # Write out to disk when at the end of the buffer
                    if buffer_location == sample_ram_buffer_size - 1:
                        start = int(
                            (proposal / online_thinning)
                            - sample_ram_buffer_size
                            + 1
                        )
                        end = int((proposal / online_thinning))
                        self.flush_samples(start, end, self.sample_ram_buffer)

        except KeyboardInterrupt:  # Catch SIGINT ------------------------------
            # Close tqdm progressbar
            proposals_total.close()
            # Flush the last samples
        finally:  # Write out samples still in the buffer ----------------------
            buffer_location: int = int(
                (proposal / online_thinning) % sample_ram_buffer_size
            )
            start = (
                int((proposal / online_thinning) / sample_ram_buffer_size)
                * sample_ram_buffer_size
            )
            end = (
                int(proposal / online_thinning) - 1
            )  # Write out one less to be sure
            if (
                end - start + 1 > 0
                and buffer_location != sample_ram_buffer_size - 1
            ):
                self.flush_samples(
                    start, end, self.sample_ram_buffer[:, :buffer_location]
                )

        # Flush output
        _sys.stdout.flush()
        _time.sleep(0.001)

        return 0

    def propagate_leapfrog(
        self,
        coordinates: _numpy.ndarray,
        momentum: _numpy.ndarray,
        iterations: int,
        time_step: float,
    ) -> _Tuple[_numpy.ndarray, _numpy.ndarray]:
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
        # Coordinates half step before loop
        coordinates += (
            0.5 * time_step * self.mass_matrix.kinetic_energy_gradient(momentum)
        )
        if self.prior.bounded:  # Correct if the distribution is bounded
            self.prior.corrector(coordinates, momentum)

        # Integration loop
        for i in range(iterations - 1):
            potential_gradient = self.target.gradient(
                coordinates
            ) + self.prior.gradient(coordinates)
            momentum -= time_step * potential_gradient
            coordinates += time_step * self.mass_matrix.kinetic_energy_gradient(
                momentum
            )
            if self.prior.bounded:  # Correct if the distribution is bounded
                self.prior.corrector(coordinates, momentum)

        # Full momentum and half step coordinates after loop
        potential_gradient = self.target.gradient(
            coordinates
        ) + self.prior.gradient(coordinates)

        # For the update
        update_coordinates = _numpy.copy(coordinates)
        update_gradient = _numpy.copy(potential_gradient)

        momentum -= time_step * potential_gradient
        coordinates += (
            0.5 * time_step * self.mass_matrix.kinetic_energy_gradient(momentum)
        )

        if self.prior.bounded:  # Correct if the distribution is bounded
            self.prior.corrector(coordinates, momentum)

        return coordinates, momentum, update_coordinates, update_gradient

    def open_hdf5(self, name: str, length: int, dtype: str = "f8"):
        self.sample_hdf5_file = _h5py.File(name, "w")
        time = 0
        # time = _time.strftime("%Y-%m-%d %H:%M:%S", _time.gmtime())
        self.sample_hdf5_dataset = self.sample_hdf5_file.create_dataset(
            f"samples {time}", (self.dimensions + 1, length), dtype=dtype
        )
        self.sample_hdf5_dataset.attrs["sampler"] = "HMC"

    def flush_samples(self, start: int, end: int, data: _numpy.ndarray):
        self.sample_hdf5_dataset.attrs["end_of_samples"] = end + 1
        self.sample_hdf5_dataset[:, start : end + 1] = data
