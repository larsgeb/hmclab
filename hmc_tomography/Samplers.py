"""Sampler classes and associated methods.

The classes in this module provide different sampling algorithms to appraise
distributions. All of them are designed to work in a minimal way; you can run the
sampling method with only a target distribution and filename to write your samples to.
However, the true power of any algorithm only shows when the user injects his expertise
through tuning parameters.

Sampling can be initialised from both an instance of a sampler or directly as a static
method:

.. code-block:: python

    from hmc_tomography import HMC

    HMC_instance = HMC()

    # Sampling using the static method
    HMC.sample(distribution, "samples.h5")

    # Sampling using the instance method
    HMC_instance.sample(distribution, "samples.h5")

All of the classes inherit from :class:`._AbstractSampler`; a base class outlining
required methods and their signatures (required in- and outputs).


"""
import warnings as _warnings
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod
from time import time as _time
from typing import Union as _Union

import h5py as _h5py
import numpy as _numpy
import tqdm.auto as _tqdm_au

from hmc_tomography.Distributions import _AbstractDistribution
from hmc_tomography.MassMatrices import Unit as _Unit
from hmc_tomography.MassMatrices import _AbstractMassMatrix


class _AbstractSampler(_ABC):
    """Abstract base class for Markov chain Monte Carlo samplers.

    """

    name: str = "Monte Carlo sampler abstract base class"
    """The name of the sampler"""

    dimensions: int = None
    """An integer representing the dimensionality in which the MCMC sampler works."""

    distribution: _AbstractDistribution = None
    """The _AbstractDistribution object on which the sampler works."""

    samples_hdf5_filename: str = None
    """A string containing the path+filename of the hdf5 file to which samples will be
    stored."""

    samples_hdf5_filehandle = None
    """A HDF5 file handle of the hdf5 file to which samples will be stored."""

    samples_hdf5_dataset = None
    """A string containing the HDF5 group of the hdf5 file to which samples will be
    stored. """

    ram_buffer: _numpy.ndarray = None
    """A NumPy ndarray containing the samples that are as of yet not written to disk."""

    current_model: _numpy.ndarray = None
    """A NumPy array containing the model at the current state of the Markov chain. """

    proposed_model: _numpy.ndarray = None
    """A NumPy array containing the model at the proposed state of the Markov chain. """

    current_x: float = None
    """A NumPy array containing :math:`\chi = -\log\left( p\\right)` (i.e. the misfit,
    negative log probability) of the distribution at the current state of the
    Markov chain. """

    proposed_x: float = None
    """A NumPy array containing :math:`\chi = -\log\left( p\\right)` (i.e. the misfit,
    negative log probability) of the distribution at the proposed state of the
    Markov chain. """

    accepted_proposals: int = None
    """An integer representing the amount of accepted proposals."""

    amount_of_writes: int = None
    """An integer representing the amount of times the sampler has written to disk."""

    progressbar_refresh_rate: int = 1000
    """An integer representing how many samples lie between an update of the progress
    bar statistics (acceptance rate etc.)."""

    max_time: float = None
    """A float representing the maximum time in seconds that sampling is allowed to take
    before it is automatically terminated. The value None is used for unlimited time."""

    def __init__(self):
        self.sample = self._sample

    def _init_sampler(
        self,
        samples_hdf5_filename,
        distribution,
        initial_model,
        proposals,
        online_thinning,
        ram_buffer_size,
        overwrite_existing_file,
        max_time,
        **kwargs,
    ):

        # Parse the distribution -------------------------------------------------------

        # Store the distribution
        assert issubclass(type(distribution), _AbstractDistribution)
        self.distribution = distribution

        # Extract dimensionality from the distribution
        assert distribution.dimensions > 0
        assert type(distribution.dimensions) == int
        self.dimensions = distribution.dimensions

        # Set up proposals -------------------------------------------------------------

        # Assert that proposals is a positive integer
        assert proposals > 0
        assert type(proposals) == int
        self.proposals = proposals

        # Assert that online_thinning is a positive integer
        assert online_thinning > 0
        assert type(online_thinning) == int
        self.online_thinning = online_thinning

        # Assert that we would write out the last sample, preventing wasteful
        # computations
        assert self.proposals % self.online_thinning == 0
        self.proposals_after_thinning = int(self.proposals / self.online_thinning)

        # Set up the sample RAM buffer -------------------------------------------------

        if ram_buffer_size is not None:
            # Assert that ram_buffer_size is a positive integer
            assert ram_buffer_size > 0
            assert type(ram_buffer_size) == int

            # Set the ram buffer size
            self.ram_buffer_size = ram_buffer_size

        else:
            # This is all automated stuff. You can force any size by setting it
            # manually.

            # Detailed explanation: we strive for approximately 1 gigabyte in
            # memory before we write to disk, by default. The amount of floats that are
            # in memory is calculated as follows: (dimensions + 1) *
            # ram_buffer_size. The plus ones comes from requiring to store the
            # misfit. 1 gigabyte is approximately 1e8 64 bits floats (actually 1.25e8).
            # Additionally, there is a cap at 10000 samples.
            ram_buffer_size = min(int(_numpy.floor(1e8 / self.dimensions)), 10000)
            # Reduce this number until it fits in the amount of proposals
            while self.proposals_after_thinning % ram_buffer_size != 0:
                ram_buffer_size = ram_buffer_size - 1

            # Now, this number might be larger than the actual amount of samples, so we
            # take the minimum of this and the amount of proposals to write as the
            # actual ram size.
            self.ram_buffer_size = min(ram_buffer_size, self.proposals_after_thinning)

            assert type(self.ram_buffer_size) == int

        shape = (self.dimensions + 1, self.ram_buffer_size)

        self.ram_buffer = _numpy.empty(shape, dtype=_numpy.float64)

        # Set up the samples file ------------------------------------------------------

        # Parse the filename
        assert type(samples_hdf5_filename) == str
        if samples_hdf5_filename[-3:] != ".h5":
            samples_hdf5_filename += ".h5"
        self.samples_hdf5_filename = samples_hdf5_filename

        # Open the HDF5 file
        self._open_samples_hdf5(
            self.samples_hdf5_filename,
            self.proposals_after_thinning,
            overwrite=overwrite_existing_file,
        )

        # Set up the initial model and preallocate other necessary arrays --------------

        if initial_model is None:
            initial_model = _numpy.zeros((self.dimensions, 1))

        assert initial_model.shape == (self.dimensions, 1)

        self.current_model = initial_model.astype(_numpy.float64)
        self.current_x = distribution.misfit(self.current_model)

        self.proposed_model = _numpy.empty_like(self.current_model)
        self.proposed_x = _numpy.nan

        # Set up accepted_proposals for acceptance rate --------------------------------
        self.accepted_proposals = 0
        self.amount_of_writes = 0

        # Set up time limit ------------------------------------------------------------

        if max_time is not None:
            max_time = float(max_time)
            assert max_time > 0.0
            self.max_time = max_time

        # Do specific stuff ------------------------------------------------------------

        # Set up specifics for each algorithm
        self._init_sampler_specific(**kwargs)

        # Write out the tuning settings
        self._write_tuning_settings()

        # Create attributes before sampling, such that SWMR works
        self.samples_hdf5_dataset.attrs["write_index"] = -1
        self.samples_hdf5_dataset.attrs["last_written_sample"] = -1

    def _close_sampler(self):
        self.samples_hdf5_filehandle.close()

    def _sample_loop(self):
        """The actual sampling code."""

        # As soon as all attributes and datasets are created, enable SWMR mode.
        self.samples_hdf5_filehandle.swmr_mode = True

        # Create progressbar -----------------------------------------------------------
        try:
            self.proposals_iterator = _tqdm_au.trange(
                self.proposals,
                desc="Sampling. Acceptance rate:",
                leave=True,
                dynamic_ncols=True,
            )
        except Exception:
            self.proposals_iterator = _tqdm_au.trange(
                self.proposals, desc="Sampling. Acceptance rate:", leave=True,
            )

        # Run the Markov process -------------------------------------------------------
        try:

            # If we are given a maximum time, start timer now
            if self.max_time is not None:
                t_end = _time() + self.max_time

            for self.current_proposal in self.proposals_iterator:

                # Propose a new sample
                self._propose()

                # Evaluate acceptance criterium
                self._evaluate_acceptance()

                # If we are on a thinning number (i.e. one of the non-discarded samples)
                if self.current_proposal % self.online_thinning == 0:

                    # Write sample to ram array
                    self._sample_to_ram()

                    # Calculate the index this sample has after thinning
                    after_thinning = int(self.current_proposal / self.online_thinning)

                    # Check if this number is at the end of the buffer
                    if (
                        after_thinning % self.ram_buffer_size
                    ) == self.ram_buffer_size - 1:

                        # If so, write samples to disk
                        self._samples_to_disk()

                # Update the progressbar
                self._update_progressbar()

                # Check elapsed time
                if self.max_time is not None and t_end < _time():
                    # Raise KeyboardInterrupt if we're over time
                    raise KeyboardInterrupt

        except KeyboardInterrupt:  # Catch SIGINT --------------------------------------
            # Close progressbar
            self.proposals_iterator.close()
        finally:  # Write out the last samples not on a full buffer --------------------
            self._samples_to_disk()

        self._close_sampler()

    def _open_samples_hdf5(
        self,
        name: str,
        length: int,
        dtype: str = "f8",
        nested=False,
        overwrite: bool = False,
    ) -> int:

        # Add file extension
        if not name.endswith(".h5"):
            name += ".h5"

        # Try to create file
        try:
            if overwrite:  # honor overwrite flag
                flag = "w"
                _warnings.warn(
                    f"\r\nSilently overwriting samples file ({name}) if it exists.",
                    Warning,
                    stacklevel=100,
                )
            else:
                flag = "w-"

            # Create file, fail if exists and flag == w-
            self.samples_hdf5_filehandle = _h5py.File(name, flag, libver="latest")

        except OSError:
            # Catch error on file creations, likely that the file already exists

            # If it exists, prompt the user with a warning
            _warnings.warn(
                f"\r\nIt seems that the samples file ({name}) already exists, or the"
                f"file could otherwise not be created.",
                Warning,
                stacklevel=100,
            )

            # Keep track of user input
            choice_made = False

            # Keep trying until the user makes a valid choice
            while not choice_made:

                # Prompt user with three options, abort, overwrite, or new filename
                if nested:
                    # If this is not the first time that this is called, also print the
                    # warning again
                    input_choice = input(
                        f"{name} also exists. (n)ew file name, (o)verwrite or (a)bort? "
                        ">> "
                    )
                else:
                    input_choice = input("(n)ew file name, (o)verwrite or (a)bort? >> ")

                # Act on choice
                if input_choice == "n":
                    # User wants a new file
                    choice_made = True

                    # Ask user for the new filename
                    new_name = input("new file name? (adds missing .h5) >> ")

                    # Call the current method again, with the new filename (recursion!)
                    self._open_samples_hdf5(new_name, length, dtype=dtype, nested=True)

                    # Exit from here
                    return

                elif input_choice == "o":
                    # User wants to overwrite the file
                    choice_made = True

                    # Create file, truncate if exists. This should never give an
                    # error on file exists, but could fail for other reasons. Therefore,
                    # no try-catch block.
                    self.samples_hdf5_filehandle = _h5py.File(
                        name, "w", libver="latest"
                    )

                elif input_choice == "a":
                    # User wants to abort sampling
                    choice_made = True
                    raise AttributeError(
                        "Wasn't able to create the samples file. This exception should "
                        "come paired with an OSError thrown by h5py."
                    )

        # Update the filename in the sampler object for later retrieval
        self.samples_hdf5_filename = name

        # Create dataset
        self.samples_hdf5_dataset = self.samples_hdf5_filehandle.create_dataset(
            "samples_0",
            (self.dimensions + 1, 1),
            maxshape=(self.dimensions + 1, length),  # one extra for misfit
            dtype=dtype,
            chunks=True,
        )

        # Set the current index of samples to the start of the file
        self.samples_hdf5_dataset.attrs["write_index"] = -1
        self.samples_hdf5_dataset.attrs["last_written_sample"] = -1

    def _update_progressbar(self):

        if self.current_proposal % self.progressbar_refresh_rate:
            # Calculate acceptance rate
            acceptance_rate = self.accepted_proposals / (self.current_proposal + 1)

            self.proposals_iterator.set_description(
                f"Tot. acc rate: {acceptance_rate:.2f}. Progress", refresh=False,
            )

    def _sample_to_ram(self):
        # Calculate proposal number after thinning
        current_proposal_after_thinning = int(
            self.current_proposal / self.online_thinning
        )
        # Assert that it's an integer (if we only write on end of buffer)
        assert self.current_proposal % self.online_thinning == 0

        # Calculate index for the RAM array
        index_in_ram = current_proposal_after_thinning % self.ram_buffer_size

        # Place samples in RAM
        self.ram_buffer[:-1, index_in_ram] = self.current_model[:, 0]

        # Place misfit in RAM
        self.ram_buffer[-1, index_in_ram] = self.current_x

    def _samples_to_disk(self):

        # Calculate proposal number after thinning
        current_proposal_after_thinning = int(
            _numpy.floor(self.current_proposal / self.online_thinning)
        )

        # This should always be the case if we write during or at the end of sampling,
        # but not after a KeyboardInterrupt. However, samples are definitely only
        # written to RAM on a whole number.
        # assert self.current_proposal % self.online_thinning == 0:

        # Calculate start/end indices
        robust_start = (
            current_proposal_after_thinning
            - current_proposal_after_thinning % self.ram_buffer_size
        )
        end = current_proposal_after_thinning + 1

        # If there is something to write
        if (
            self.samples_hdf5_dataset.attrs["write_index"] == robust_start
            or self.samples_hdf5_dataset.attrs["write_index"] == -1
        ):

            # Some sanity checks on the indices
            assert (
                robust_start >= 0 and robust_start <= self.proposals_after_thinning + 1
            )
            assert end >= 0 and end <= self.proposals_after_thinning + 1
            assert end >= robust_start

            # Update the amount of writes
            self.amount_of_writes += 1

            # Update the size in the h5 file
            self.samples_hdf5_dataset.resize(
                (self.dimensions + 1, current_proposal_after_thinning + 1)
            )

            # Write samples to disk
            self.samples_hdf5_dataset[:, robust_start:end] = self.ram_buffer[
                :, : end - robust_start
            ]

            # Reset the markers in the HDF5 file
            self.samples_hdf5_dataset.attrs["write_index"] = end
            self.samples_hdf5_dataset.attrs["last_written_sample"] = end - 1
            self.samples_hdf5_dataset.flush()

    @_abstractmethod
    def _propose(self):
        pass

    @_abstractmethod
    def _evaluate_acceptance(self):
        """This abstract method evaluates the acceptance criterion in the MCMC
        algorithm. Pass or fail, it updates the objects attributes accordingly;
        modifying current_model and current_x as needed."""
        pass

    @_abstractmethod
    def _write_tuning_settings(self):
        """An abstract method that writes all the relevant tuning settings of the
        algorithm to the HDF5 file."""
        pass

    @_abstractmethod
    def _init_sampler_specific(self):
        """An abstract method that sets up all required attributes and method for the
        algorithm."""
        pass


class RWMH(_AbstractSampler):
    step_length: _Union[float, _numpy.ndarray] = 1.0
    """A parameter describing the standard deviation of a multivariate normal (MVN) used
    as the proposal distribution for Random Walk Metropolis-Hastings. Using a
    _numpy.ndarray column vector (shape dimensions × 1) will give every dimensions a
    unique step length. Correlations in the MVN are not yet implemented. Has a strong
    influence on acceptance rate. **An essential tuning parameter.**"""

    def _sample(
        self,
        samples_hdf5_filename: str,
        distribution: _AbstractDistribution,
        step_length: _Union[float, _numpy.ndarray] = 1.0,
        initial_model: _numpy.ndarray = None,
        proposals: int = 100,
        online_thinning: int = 1,
        ram_buffer_size: int = None,
        overwrite_existing_file: bool = False,
        max_time: float = None,
    ):
        """Sampling using the Metropolis-Hastings algorithm.

        Parameters
        ----------
        samples_hdf5_filename: str
            String containing the (path+)filename of the HDF5 file to contain the
            samples. **A required parameter.**
        distribution: _AbstractDistribution
            The distribution to sample. Should be an instance of a subclass of
            _AbstractDistribution. **A required parameter.**
        step_length: _Union[float, _numpy.ndarray]
            A parameter describing the standard deviation of a multivariate normal (MVN)
            used as the proposal distribution for Random Walk Metropolis-Hastings. Using
            a _numpy.ndarray column vector (shape dimensions × 1) will give every
            dimensions a unique step length. Correlations in the MVN are not yet
            implemented. Has a strong influence on acceptance rate. **An essential
            tuning parameter.**
        initial_model: _numpy
            A NumPy column vector (shape dimensions × 1) containing the starting model
            of the Markov chain. This model will not be written out as a sample.
        proposals: int
            An integer representing the amount of proposals the algorithm should make.
        online_thinning: int
            An integer representing the degree of online thinning, i.e. the interval
            between storing samples.
        ram_buffer_size: int
            An integer representing how many samples should be kept in RAM before
            writing to storage.
        overwrite_existing_file: bool
            A boolean describing whether or not to silently overwrite existing files.
            Use with caution.
        max_time: float
            A float representing the maximum time in seconds that sampling is allowed to
            take before it is automatically terminated. The value None is used for
            unlimited time.
        **kwargs
            Arbitrary keyword arguments.

        Raises
        ------
        AssertionError
            For any unspecified invalid entry.
        ValueError
            For any invalid value of algorithm settings.
        TypeError
            For any invalid value of algorithm settings.


        """
        self._init_sampler(
            samples_hdf5_filename=samples_hdf5_filename,
            distribution=distribution,
            step_length=step_length,
            initial_model=initial_model,
            proposals=proposals,
            online_thinning=online_thinning,
            ram_buffer_size=ram_buffer_size,
            overwrite_existing_file=overwrite_existing_file,
            max_time=max_time,
        )

        self._sample_loop()

    @classmethod
    def sample(
        cls,
        samples_hdf5_filename: str,
        distribution: _AbstractDistribution,
        step_length: float = 1.0,
        initial_model: _numpy.ndarray = None,
        proposals: int = 100,
        online_thinning: int = 1,
        ram_buffer_size: int = None,
        overwrite_existing_file: bool = False,
        max_time: float = None,
    ):
        """Simply a forward to the instance method of the sampler; check out
        _sample()."""

        instance = cls()

        instance._sample(
            samples_hdf5_filename=samples_hdf5_filename,
            distribution=distribution,
            step_length=step_length,
            initial_model=initial_model,
            proposals=proposals,
            online_thinning=online_thinning,
            ram_buffer_size=ram_buffer_size,
            overwrite_existing_file=overwrite_existing_file,
            max_time=max_time,
        )

        return instance

    def _init_sampler_specific(self, **kwargs):

        # Parse all possible kwargs
        for key in ("step_length", "example_extra_option"):
            if key in kwargs:
                setattr(self, key, kwargs[key])

        # Assert that step length is either a float and bigger than zero, or a full
        # matrix / diagonal
        try:
            self.step_length = float(self.step_length)
            assert self.step_length > 0.0, (
                "RW-MH step length should be a positive float or a numpy.ndarray. The "
                "passed argument is a float equal to or smaller than zero."
            )
        except TypeError:
            assert type(self.step_length) == _numpy.ndarray, (
                "RW-MH step length should be a numpy.ndarray of shape (dimensions, 1) "
                "or a positive float. The passed argument is neither."
            )
            assert self.step_length.shape == (self.dimensions, 1), (
                "RW-MH step length should be a numpy.ndarray of shape (dimensions, 1) "
                "or a positive float. The passed argument is an ndarray of the wrong "
                "shape."
            )

    def _write_tuning_settings(self):
        # TODO Write this function
        pass

    def _propose(self):

        # Propose a new model according to the MH Random Walk algorithm with a Gaussian
        # proposal distribution
        self.proposed_model = (
            self.current_model
            + self.step_length * _numpy.random.randn(self.dimensions, 1)
        )
        assert self.proposed_model.shape == (self.dimensions, 1)

    def _evaluate_acceptance(self):

        # Compute new misfit
        self.proposed_x = self.distribution.misfit(self.proposed_model)

        # Evaluate acceptance rate
        if _numpy.exp(self.current_x - self.proposed_x) > _numpy.random.uniform(0, 1):
            self.current_model = _numpy.copy(self.proposed_model)
            self.current_x = self.proposed_x
            self.accepted_proposals += 1


class HMC(_AbstractSampler):
    time_step: float = 0.1
    """A positive float representing the time step to be used in solving Hamiltons
    equations. Has a strong influence on acceptance rate. **An essential tuning
    parameter.**"""

    amount_of_steps: int = 10
    """A positive integer representing the amount of integration steps to be used in
    solving Hamiltons equations. Has a medium influence on acceptance rate. **An
    essential tuning parameter.**"""

    mass_matrix: _AbstractMassMatrix = None
    """An object representing the artificial masses assigned to each parameters. Needs
    to be a subtype of _AbstractMassMatrix. Has a strong influence on convergence rate.
    **An essential tuning parameter.**"""

    current_momentum: _numpy.ndarray = None
    """A NumPy ndarray (shape dimensions × 1) containing the momentum at the current
    state of the Markov chain. Indicates direction in which the model will initially
    move along its numerical trajectory. Will be resampled from the mass matrix for each
    proposal."""

    current_k: float = _numpy.nan
    """A float representing the kinetic energy associated with the current state.
    Typically follows the ChiSquared[dimensions] distribution."""

    current_h: float = _numpy.nan
    """A float representing the total energy associated with the current state."""

    proposed_momentum: _numpy.ndarray = None
    """A NumPy ndarray (shape dimensions × 1) containing the momentum at the proposed
    state of the Markov chain. Indicates direction in which the model was moving at the
    end of its numerical trajectory. Will be computed deterministically from the
    current_momentum, i.e. the state the Markov chain started in."""

    proposed_k: float = _numpy.nan
    """A float representing the kinetic energy associated with the proposed state."""

    proposed_h: float = _numpy.nan
    """A float representing the total energy associated with the proposed state."""

    def _sample(
        self,
        samples_hdf5_filename: str,
        distribution: _AbstractDistribution,
        time_step: float = 0.1,
        amount_of_steps: int = 10,
        mass_matrix: _AbstractMassMatrix = None,
        integrator: str = "lf",
        initial_model: _numpy.ndarray = None,
        proposals: int = 100,
        online_thinning: int = 1,
        ram_buffer_size: int = None,
        overwrite_existing_file: bool = False,
        max_time: float = None,
    ):
        """Sampling using the Metropolis-Hastings algorithm.

        Parameters
        ----------
        samples_hdf5_filename: str
            String containing the (path+)filename of the HDF5 file to contain the
            samples. **A required parameter.**
        distribution: _AbstractDistribution
            The distribution to sample. Should be an instance of a subclass of
            _AbstractDistribution. **A required parameter.**
        time_step: float
            A positive float representing the time step to be used in solving Hamiltons
            equations. Has a strong influence on acceptance rate. **An essential tuning
            parameter.**
        amount_of_steps: int
            A positive integer representing the amount of integration steps to be used
            in solving Hamiltons equations. Has a medium influence on acceptance rate.
            **An essential tuning parameter.**
        mass_matrix: _AbstractMassMatrix
            An object representing the artificial masses assigned to each parameters.
            Needs to be a subtype of _AbstractMassMatrix. Has a strong influence on
            convergence rate. One passing None, defaults to the Unit mass matrix. **An
            essential tuning parameter.**
        initial_model: _numpy
            A NumPy column vector (shape dimensions × 1) containing the starting model
            of the Markov chain. This model will not be written out as a sample.
        proposals: int
            An integer representing the amount of proposals the algorithm should make.
        online_thinning: int
            An integer representing the degree of online thinning, i.e. the interval
            between storing samples.
        ram_buffer_size: int
            An integer representing how many samples should be kept in RAM before
            writing to storage.
        overwrite_existing_file: bool
            A boolean describing whether or not to silently overwrite existing files.
            Use with caution.
        max_time: float
            A float representing the maximum time in seconds that sampling is allowed to
            take before it is automatically terminated. The value None is used for
            unlimited time.

        Raises
        ------
        AssertionError
            For any unspecified invalid entry.
        ValueError
            For any invalid value of algorithm settings.
        TypeError
            For any invalid value of algorithm settings.


        """
        self._init_sampler(
            samples_hdf5_filename=samples_hdf5_filename,
            distribution=distribution,
            time_step=time_step,
            amount_of_steps=amount_of_steps,
            mass_matrix=mass_matrix,
            integrator=integrator,
            initial_model=initial_model,
            proposals=proposals,
            online_thinning=online_thinning,
            ram_buffer_size=ram_buffer_size,
            overwrite_existing_file=overwrite_existing_file,
            max_time=max_time,
        )

        self._sample_loop()

    @classmethod
    def sample(
        cls,
        samples_hdf5_filename: str,
        distribution: _AbstractDistribution,
        time_step: float = 0.1,
        amount_of_steps: int = 10,
        mass_matrix: _AbstractMassMatrix = None,
        integrator: str = "lf",
        initial_model: _numpy.ndarray = None,
        proposals: int = 100,
        online_thinning: int = 1,
        ram_buffer_size: int = None,
        overwrite_existing_file: bool = False,
        max_time: float = None,
    ):
        """Simply a forward to the instance method of the sampler; check out
        _sample()."""

        instance = cls()

        instance._sample(
            samples_hdf5_filename=samples_hdf5_filename,
            distribution=distribution,
            time_step=time_step,
            amount_of_steps=amount_of_steps,
            mass_matrix=mass_matrix,
            integrator=integrator,
            initial_model=initial_model,
            proposals=proposals,
            online_thinning=online_thinning,
            ram_buffer_size=ram_buffer_size,
            overwrite_existing_file=overwrite_existing_file,
            max_time=max_time,
        )

        return instance

    def _init_sampler_specific(self, **kwargs):
        # Parse all possible kwargs
        for key in ("time_step", "amount_of_steps", "mass_matrix", "integrator"):
            setattr(self, key, kwargs[key])
            kwargs.pop(key)

        if len(kwargs) != 0:
            raise TypeError(
                f"Unidentified argument(s) not applicable to sampler: {kwargs}"
            )

        # Step length ------------------------------------------------------------------
        # Assert that step length for Hamiltons equations is a float and bigger than
        # zero
        self.time_step = float(self.time_step)
        assert self.time_step > 0.0

        # Step amount ------------------------------------------------------------------
        # Assert that number of steps for Hamiltons equations is a positive integer
        assert type(self.amount_of_steps) == int
        assert self.amount_of_steps > 0

        # Mass matrix ------------------------------------------------------------------
        # Set the mass matrix if it is not yet set using the default: a unit mass
        if self.mass_matrix is None:
            self.mass_matrix = _Unit(self.dimensions)

        # Assert that the mass matrix is the right type and dimension
        assert isinstance(self.mass_matrix, _AbstractMassMatrix)
        assert self.mass_matrix.dimensions == self.dimensions

        # Integrator -------------------------------------------------------------------
        self.integrator = str(self.integrator)

        if self.integrator not in self.available_integrators:
            raise ValueError(
                f"Unknown integrator used. Choices are: {self.available_integrators}"
            )

    def _write_tuning_settings(self):
        # TODO Write this function
        pass

    def _propose(self):

        # Generate a momentum sample
        self.current_momentum = self.mass_matrix.generate_momentum()

        # Propagate the sample
        self.integrators[self.integrator](self)

    def _evaluate_acceptance(self):

        self.current_x = self.distribution.misfit(self.current_model)
        self.current_k = self.mass_matrix.kinetic_energy(self.current_momentum)
        self.current_h = self.current_x + self.current_k

        self.proposed_x = self.distribution.misfit(self.proposed_model)
        self.proposed_k = self.mass_matrix.kinetic_energy(self.proposed_momentum)
        self.proposed_h = self.proposed_x + self.proposed_k

        # Evaluate acceptence rate
        if _numpy.exp(self.current_h - self.proposed_h) > _numpy.random.uniform(0, 1):
            self.current_model = _numpy.copy(self.proposed_model)
            self.current_x = self.proposed_x
            self.accepted_proposals += 1

    def _propagate_leapfrog(self,):

        # Make sure not to alter a view but a copy of arrays ---------------------------
        position = self.current_model.copy()
        momentum = self.current_momentum.copy()

        # Leapfrog integration ---------------------------------------------------------
        position += (
            0.5 * self.time_step * self.mass_matrix.kinetic_energy_gradient(momentum)
        )

        self.distribution.corrector(position, momentum)

        # Integration loop
        for i in range(self.amount_of_steps - 1):

            # Calculate gradient
            potential_gradient = self.distribution.gradient(position)

            momentum -= self.time_step * potential_gradient
            position += self.time_step * self.mass_matrix.kinetic_energy_gradient(
                momentum
            )

            # Correct bounds
            self.distribution.corrector(position, momentum)

        # Full momentum and half step position after loop ------------------------------

        # Calculate gradient
        potential_gradient = self.distribution.gradient(position)

        momentum -= self.time_step * potential_gradient
        position += (
            0.5 * self.time_step * self.mass_matrix.kinetic_energy_gradient(momentum)
        )

        self.distribution.corrector(position, momentum)

        self.proposed_model = position.copy()
        self.proposed_momentum = momentum.copy()

    def _propagate_4_stage_simplified(self):
        # Schema: (a1,b1,a2,b2,a3,b2,a2,b1,a1)
        a1 = 0.071353913450279725904
        a2 = 0.268548791161230105820
        a3 = 1.0 - 2.0 * a1 - 2.0 * a2
        b1 = 0.191667800000000000000
        b2 = 1.0 / 2.0 - b1

        a1 *= self.time_step
        a2 *= self.time_step
        a3 *= self.time_step
        b1 *= self.time_step
        b2 *= self.time_step

        # Make sure not to alter a view but a copy of the passed arrays.
        position = self.current_model.copy()
        momentum = self.current_momentum.copy()

        # Leapfrog integration -------------------------------------------------
        for i in range(self.amount_of_steps):

            # A1
            position += a1 * self.mass_matrix.kinetic_energy_gradient(momentum)
            self.distribution.corrector(position, momentum)

            # B1
            potential_gradient = self.distribution.gradient(position)
            momentum -= b1 * potential_gradient

            # A2
            position += a2 * self.mass_matrix.kinetic_energy_gradient(momentum)
            self.distribution.corrector(position, momentum)

            # B2
            potential_gradient = self.distribution.gradient(position)
            momentum -= b2 * potential_gradient

            # A3
            position += a3 * self.mass_matrix.kinetic_energy_gradient(momentum)
            self.distribution.corrector(position, momentum)

            # B2
            potential_gradient = self.distribution.gradient(position)
            momentum -= b2 * potential_gradient

            # A2
            position += a2 * self.mass_matrix.kinetic_energy_gradient(momentum)
            self.distribution.corrector(position, momentum)

            # B1
            potential_gradient = self.distribution.gradient(position)
            momentum -= b1 * potential_gradient

            # A1
            position += a1 * self.mass_matrix.kinetic_energy_gradient(momentum)
            self.distribution.corrector(position, momentum)

        # For the update
        self.proposed_model = _numpy.copy(position)
        self.proposed_momentum = _numpy.copy(momentum)

    def _propagate_3_stage_simplified(self):

        # Schema: (a1,b1,a2,b2,a2,b1,a1)
        a1 = 0.11888010966548
        a2 = 1.0 / 2.0 - a1
        b1 = 0.29619504261126
        b2 = 1.0 - 2.0 * b1

        a1 *= self.time_step
        a2 *= self.time_step
        b1 *= self.time_step
        b2 *= self.time_step

        # Make sure not to alter a view but a copy of the passed arrays.
        position = self.current_model.copy()
        momentum = self.current_momentum.copy()

        # Leapfrog integration -------------------------------------------------
        for i in range(self.amount_of_steps):

            # A1
            position += a1 * self.mass_matrix.kinetic_energy_gradient(momentum)
            self.distribution.corrector(position, momentum)

            # B1
            potential_gradient = self.distribution.gradient(position)
            momentum -= b1 * potential_gradient

            # A2
            position += a2 * self.mass_matrix.kinetic_energy_gradient(momentum)
            self.distribution.corrector(position, momentum)

            # B2
            potential_gradient = self.distribution.gradient(position)
            momentum -= b2 * potential_gradient

            # A2
            position += a2 * self.mass_matrix.kinetic_energy_gradient(momentum)
            self.distribution.corrector(position, momentum)

            # B1
            potential_gradient = self.distribution.gradient(position)
            momentum -= b1 * potential_gradient

            # A1
            position += a1 * self.mass_matrix.kinetic_energy_gradient(momentum)
            self.distribution.corrector(position, momentum)

        self.proposed_model = position.copy()
        self.proposed_momentum = momentum.copy()

    integrators = {
        "lf": _propagate_leapfrog,
        "3s": _propagate_3_stage_simplified,
        "4s": _propagate_4_stage_simplified,
    }
    available_integrators = integrators.keys()
