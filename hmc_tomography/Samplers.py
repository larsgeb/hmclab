"""Sampler classes and associated methods.
"""
import sys as _sys
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod

import h5py as _h5py
import numpy as _numpy
import time as _time
import tqdm as _tqdm
import warnings as _warnings
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

    samples_filename: str = ""
    """Variable to store the samples filename. This is stored because during sampling
    the filename might be altered. This alteration is never done without user
    interaction."""

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
        if not (target.dimensions == prior.dimensions == mass_matrix.dimensions):
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
        suppress_warnings: bool = True,
        overwrite_samples: bool = False,
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

        # If suppress warnings ---------------------------------------------------------
        if suppress_warnings:
            _warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Prepare sampling -------------------------------------------------------------
        accepted = 0

        # Initial model
        if initial_model is None:
            coordinates = _numpy.zeros((self.dimensions, 1))
        else:
            assert initial_model.shape == (self.dimensions, 1)
            coordinates = initial_model

        # Create RAM buffer for samples
        self.sample_ram_buffer = _numpy.empty(
            (self.dimensions + 1, sample_ram_buffer_size)
        )

        # Create or open HDF5 file
        total_samples_to_be_generated = int((1.0 / online_thinning) * proposals)

        # Open file, with some intricate logic

        return_code = self.open_samples_hdf5(
            samples_filename, total_samples_to_be_generated, overwrite=overwrite_samples
        )
        if return_code == 1:
            return 1

        # Set attributes of the dataset to correspond to sampling settings
        self.sample_hdf5_dataset.attrs["sampler"] = "HMC"
        self.sample_hdf5_dataset.attrs["proposals"] = proposals
        self.sample_hdf5_dataset.attrs["online_thinning"] = online_thinning
        self.sample_hdf5_dataset.attrs["time_step"] = time_step
        self.sample_hdf5_dataset.attrs["integration_steps"] = integration_steps
        self.sample_hdf5_dataset.attrs["randomizetime_step"] = randomize_time_step
        self.sample_hdf5_dataset.attrs[
            "randomize_iterations"
        ] = randomize_integration_steps
        self.sample_hdf5_dataset.attrs[
            "randomize_iterations"
        ] = randomize_integration_steps
        self.sample_hdf5_dataset.attrs["start_time"] = _time.strftime(
            "%Y-%m-%d %H:%M:%S", _time.gmtime()
        )

        # Flush output (works best if followed by sleep() )
        _sys.stdout.flush()
        _time.sleep(0.001)

        # Create progress bar
        proposals_total = _tqdm.trange(
            proposals, desc="Sampling. Acceptance rate:", leave=True
        )

        # Sampling acceptance history
        acceptance_history = _numpy.zeros((100,))

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

        # Start sampling, but catch CTRL+C / COMMAND + . (SIGINT) ----------------------
        try:
            current_proposal = 0
            for proposal in proposals_total:

                # Compute initial Hamiltonian
                potential: float = self.target.misfit(coordinates) + self.prior.misfit(
                    coordinates
                )
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
                new_kinetic: float = self.mass_matrix.kinetic_energy(new_momentum)
                new_hamiltonian: float = new_potential + new_kinetic

                # Evaluate acceptance criterion
                if _numpy.exp(hamiltonian - new_hamiltonian) > _numpy.random.uniform(
                    0, 1
                ):
                    accepted += 1
                    coordinates = new_coordinates.copy()
                    acceptance_history = _numpy.append([1], acceptance_history[0:-1])
                else:
                    acceptance_history = _numpy.append([0], acceptance_history[0:-1])

                if (
                    callable(self.mass_matrix_update_hook)
                    and not ignore_update_hook_mass
                ):
                    self.mass_matrix.update(update_m, update_g)

                # On-line thinning, progress bar and writing samples to disk -----------
                if proposal % online_thinning == 0:
                    buffer_location: int = int(
                        (proposal / online_thinning) % sample_ram_buffer_size
                    )
                    self.sample_ram_buffer[:-1, buffer_location] = coordinates[:, 0]
                    self.sample_ram_buffer[-1, buffer_location] = self.target.misfit(
                        coordinates
                    ) + self.prior.misfit(coordinates)

                    # Total acceptance rate
                    tot_acc = accepted / (proposal + 1)

                    # Last 100 acceptance rate
                    loc_acc = _numpy.sum(acceptance_history) / min(
                        100.0, current_proposal
                    )

                    proposals_total.set_description(
                        f"Tot. acc rate: {tot_acc:.2f}. "
                        f"Last 100 acc rate: {loc_acc:.2f}. "
                        "Progress"
                    )
                    # Write out to disk when at the end of the buffer
                    if buffer_location == sample_ram_buffer_size - 1:
                        start = int(
                            (proposal / online_thinning) - sample_ram_buffer_size + 1
                        )
                        end = int((proposal / online_thinning))
                        self.flush_samples(start, end, self.sample_ram_buffer)
                current_proposal += 1

        except KeyboardInterrupt:  # Catch SIGINT --------------------------------------
            # Close tqdm progressbar
            proposals_total.close()
            # optional TODO delete all non-written entries in hdf5 file. This is also
            # taken care of in plotting
        finally:  # Write out samples still in the buffer ------------------------------
            buffer_location: int = int(
                (proposal / online_thinning) % sample_ram_buffer_size
            )
            start = (
                int((proposal / online_thinning) / sample_ram_buffer_size)
                * sample_ram_buffer_size
            )

            # Write out one less to be sure
            end = int(proposal / online_thinning) - 1
            if end - start + 1 > 0 and buffer_location != sample_ram_buffer_size - 1:
                self.flush_samples(
                    start, end, self.sample_ram_buffer[:, :buffer_location]
                )

            # Trim the dataset to have the shape of end_of_samples
            self.sample_hdf5_dataset.resize((self.dimensions + 1, end + 1))

            # Close the HDF file
            self.sample_hdf5_file.close()

        # Re-enable warnings
        if suppress_warnings:
            _warnings.filterwarnings("default", category=RuntimeWarning)

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

            # Correct bounds
            self.prior.corrector(coordinates, momentum)

        # Full momentum and half step coordinates after loop
        potential_gradient = self.target.gradient(coordinates) + self.prior.gradient(
            coordinates
        )

        # For the update
        update_coordinates = _numpy.copy(coordinates)
        update_gradient = _numpy.copy(potential_gradient)

        momentum -= time_step * potential_gradient
        coordinates += (
            0.5 * time_step * self.mass_matrix.kinetic_energy_gradient(momentum)
        )

        self.prior.corrector(coordinates, momentum)

        return coordinates, momentum, update_coordinates, update_gradient

    def open_samples_hdf5(
        self,
        name: str,
        length: int,
        dtype: str = "f8",
        nested=False,
        overwrite: bool = False,
    ) -> int:

        if not name.endswith(".h5"):
            name += ".h5"

        try:
            if overwrite:
                flag = "w"
                _warnings.warn(
                    f"\r\nSilently overwriting samples file ({name}) if it exists.",
                    Warning,
                    stacklevel=100,
                )
            else:
                flag = "w-"

            # Create file, fail if exists and flag == w-
            self.sample_hdf5_file = _h5py.File(name, flag)

            # Create dataset
            self.sample_hdf5_dataset = self.sample_hdf5_file.create_dataset(
                "samples_0", (self.dimensions + 1, length), dtype=dtype, chunks=True
            )

            # Update the filename in the sampler object for later retrieval
            self.samples_filename = name

            # Return the exit code upwards through the recursive functions
            return 0
        except OSError:
            # If it exists, three options, abort, overwrite, or new filename
            _warnings.warn(
                f"\r\nIt seems that the samples file ({name}) already exists.",
                Warning,
                stacklevel=100,
            )

            # Keep track of user input
            choice_made = False

            # Keep trying until the user makes a valid choice
            while not choice_made:

                # Prompt the user
                if nested:
                    # If this is not the first time that this is called, also print the warning again
                    input_choice = input(
                        f"{name} also exists. (n)ew file name, (o)verwrite or (a)bort? >> "
                    )
                else:
                    input_choice = input("(n)ew file name, (o)verwrite or (a)bort? >> ")

                if input_choice == "n":
                    # User wants a new file
                    choice_made = True

                    # Ask user for the new filename
                    name = input("new file name? (adds missing .h5) >> ")

                    # Call the current method again, with the new filename
                    return_code = self.open_samples_hdf5(
                        name, length, dtype=dtype, nested=True
                    )

                    # If the user chose abort in the recursive call, pass that on
                    return return_code

                elif input_choice == "o":
                    # User wants to overwrite the file
                    choice_made = True

                    # Create file, truncate if exists. This should never give an
                    # error on file exists, but could fail for other reasons. Therefore,
                    # no try-catch block.
                    self.sample_hdf5_file = _h5py.File(name, "w")

                    # Create dataset
                    self.sample_hdf5_dataset = self.sample_hdf5_file.create_dataset(
                        "samples_0",
                        (self.dimensions + 1, length),
                        dtype=dtype,
                        chunks=True,
                    )

                    # Update the filename in the sampler object for later retrieval
                    self.samples_filename = name
                    return 0

                elif input_choice == "a":
                    # User wants to abort sampling
                    choice_made = True
                    return 1

    def flush_samples(self, start: int, end: int, data: _numpy.ndarray):
        self.sample_hdf5_dataset.attrs["end_of_samples"] = end + 1
        self.sample_hdf5_dataset[:, start : end + 1] = data

