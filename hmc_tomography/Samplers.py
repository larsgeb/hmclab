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
from datetime import datetime as _datetime
from typing import Union as _Union

import h5py as _h5py
import numpy as _numpy
import tqdm.auto as _tqdm_au

from hmc_tomography.Distributions import _AbstractDistribution
from hmc_tomography.MassMatrices import Unit as _Unit
from hmc_tomography.MassMatrices import _AbstractMassMatrix
from hmc_tomography.Helpers.Timers import AccumulatingTimer as _AccumulatingTimer
from hmc_tomography.Helpers.CustomExceptions import InvalidCaseError

dev_assertion_message = (
    "Something went wrong internally, please report this to the developers."
)


class H5FileOpenedError(FileExistsError):
    pass


class _AbstractSampler(_ABC):
    """Abstract base class for Markov chain Monte Carlo samplers."""

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

    ram_buffer_size: int = None
    """A positive integer indicating the size of the RAM buffer in amount of samples."""

    ram_buffer: _numpy.ndarray = None
    """A NumPy ndarray containing the samples that are as of yet not written to disk."""

    current_model: _numpy.ndarray = None
    """A NumPy array containing the model at the current state of the Markov chain. """

    proposed_model: _numpy.ndarray = None
    """A NumPy array containing the model at the proposed state of the Markov chain. """

    current_x: float = None
    """A NumPy array containing :math:`\chi = -\log\left( p\\right)` (i.e. the misfit,
    negative log probability) of the distribution at the current state of the
    Markov chain."""

    proposed_x: float = None
    """A NumPy array containing :math:`\chi = -\log\left( p\\right)` (i.e. the misfit,
    negative log probability) of the distribution at the proposed state of the
    Markov chain. """

    accepted_proposals: int = None
    """An integer representing the amount of accepted proposals."""

    amount_of_writes: int = None
    """An integer representing the amount of times the sampler has written to disk."""

    progressbar_refresh_rate: float = 0.25
    """A float representing how long lies between an update of the progress bar
    statistics (acceptance rate etc.)."""

    max_time: float = None
    """A float representing the maximum time in seconds that sampling is allowed to take
    before it is automatically terminated. The value None is used for unlimited time."""

    diagnostic_mode: bool = True
    functions_to_diagnose = []
    sampler_specific_functions_to_diagnose = []

    def __init__(self):
        pass

    def _init_sampler(
        self,
        samples_hdf5_filename: str,
        distribution: _AbstractDistribution,
        initial_model: _numpy.ndarray,
        proposals: int,
        online_thinning: int,
        ram_buffer_size,
        overwrite_existing_file,
        max_time,
        diagnostic_mode: bool = False,
        **kwargs,
    ):
        """A method that is called everytime any markov chain sampler object is
        constructed.

        Args:
            samples_hdf5_filename ([type]): [description]
            distribution ([type]): [description]
            initial_model ([type]): [description]
            proposals ([type]): [description]
            online_thinning ([type]): [description]
            ram_buffer_size ([type]): [description]
            overwrite_existing_file ([type]): [description]
            max_time ([type]): [description]
        """

        assert type(samples_hdf5_filename) == str, (
            f"First argument should be a string containing the path of the file to "
            f"which to write samples. It was an object of type "
            f"{type(samples_hdf5_filename)}."
        )

        # Parse the distribution -------------------------------------------------------

        # Store the distribution
        assert issubclass(type(distribution), _AbstractDistribution), (
            "The passed target distribution should be a derived class of "
            "_AbstractDistribution."
        )
        self.distribution = distribution

        # Extract dimensionality from the distribution
        assert type(distribution.dimensions) == int and distribution.dimensions > 0, (
            "The passed target distribution should have an integer dimension larger "
            "than zero."
        )
        self.dimensions = distribution.dimensions

        # Set up proposals -------------------------------------------------------------

        # Assert that proposals is a positive integer
        assert type(proposals) == int and proposals > 0, (
            "The amount of proposal requested (`proposals`) should be an integer "
            "number larger than zero."
        )
        self.proposals = proposals

        # Assert that online_thinning is a positive integer
        assert type(online_thinning) == int and online_thinning > 0, (
            "The amount of online thinning (`online_thinning`) should be an integer "
            "number larger than zero."
        )
        self.online_thinning = online_thinning

        # Assert that we would write out the last sample, preventing wasteful
        # computations
        assert self.proposals % self.online_thinning == 0, (
            "The amount of proposals (`proposals`) needs to be a multiple of the "
            "online thinning (`online_thinning`) number, to prevent sample wastage."
        )
        self.proposals_after_thinning = int(self.proposals / self.online_thinning)

        # Set up the sample RAM buffer -------------------------------------------------

        if ram_buffer_size is not None:
            # Assert that ram_buffer_size is a positive integer
            assert type(ram_buffer_size) == int and ram_buffer_size > 0, (
                "The ram buffer size (`ram_buffer_size`) needs to be an integer larger "
                "than zero."
            )

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

            assert type(self.ram_buffer_size) == int, dev_assertion_message

        shape = (self.dimensions + 1, self.ram_buffer_size)

        self.ram_buffer = _numpy.empty(shape, dtype=_numpy.float64)

        # Set up the samples file ------------------------------------------------------

        # Parse the filename
        assert (
            type(samples_hdf5_filename) == str
        ), "The samples filename needs to be a string."
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

        assert initial_model.shape == (self.dimensions, 1), (
            f"The initial model (`initial_model`) dimension is incompatible with the "
            f"target distribution. Supplied model shape: {initial_model.shape}."
            f"Required shape: {(self.dimensions, 1)}"
        )

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
            assert (
                max_time > 0.0
            ), "The maximal runtime (`max_time`) should be a float larger than zero."
            self.max_time = max_time

        # Prepare diagnostic mode if needed --------------------------------------------
        self.diagnostic_mode = diagnostic_mode

        if self.diagnostic_mode:
            self._propose = _AccumulatingTimer(self._propose)
            self._evaluate_acceptance = _AccumulatingTimer(self._evaluate_acceptance)
            self._sample_to_ram = _AccumulatingTimer(self._sample_to_ram)
            self._samples_to_disk = _AccumulatingTimer(self._samples_to_disk)
            self._update_progressbar = _AccumulatingTimer(self._update_progressbar)

            self.functions_to_diagnose = [
                self._propose,
                self._evaluate_acceptance,
                self._sample_to_ram,
                self._samples_to_disk,
                self._update_progressbar,
            ]

        # Do sampler specific operations -----------------------------------------------

        # Set up specifics for each algorithm
        self._init_sampler_specific(**kwargs)

        # Write out the tuning settings
        self._write_tuning_settings()

        # Create attributes before sampling, such that SWMR works
        self.samples_hdf5_dataset.attrs["write_index"] = -1
        self.samples_hdf5_dataset.attrs["last_written_sample"] = -1
        self.samples_hdf5_dataset.attrs["proposals"] = self.proposals
        self.samples_hdf5_dataset.attrs["acceptance_rate"] = 0
        self.samples_hdf5_dataset.attrs["online_thinning"] = self.online_thinning
        self.samples_hdf5_dataset.attrs["start_time"] = _datetime.now().strftime(
            "%d-%b-%Y (%H:%M:%S.%f)"
        )
        self.samples_hdf5_dataset.attrs["sampler"] = self.name

    def _close_sampler(self):

        self.samples_hdf5_dataset.attrs["acceptance_rate"] = self.accepted_proposals / (
            self.current_proposal + 1
        )
        self.samples_hdf5_dataset.attrs["end_time"] = _datetime.now().strftime(
            "%d-%b-%Y (%H:%M:%S.%f)"
        )
        self.samples_hdf5_dataset.attrs["runtime"] = str(
            self.end_time - self.start_time
        )
        self.samples_hdf5_dataset.attrs["runtime_seconds"] = (
            self.end_time - self.start_time
        ).total_seconds()

        self.samples_hdf5_filehandle.close()

        self._close_sampler_specific()

        if self.diagnostic_mode:
            # This block shows the percentage of time spent in each part of the sampler.
            percentage_time_spent = {}
            print("Detailed statistics:")

            total_time = (self.end_time - self.start_time).total_seconds()

            print(f"Total runtime: {total_time:.2f} seconds")

            for fn in self.functions_to_diagnose:
                percentage_time_spent[fn.function.__name__] = (
                    100 * fn.time_spent / total_time
                )
            print()
            print("General sampler components:")
            print("{:<30} {:<30}".format("Function", "percentage of time"))
            for name, percentage in percentage_time_spent.items():
                print("{:<30} {:<30.2f}".format(name, percentage))

            percentage_time_spent = {}
            for fn in self.sampler_specific_functions_to_diagnose:
                percentage_time_spent[fn.function.__name__] = (
                    100 * fn.time_spent / total_time
                )
            print()
            print(f"{self.name} specific components:")
            print("{:<30} {:<30}".format("Function", "percentage of time"))
            for name, percentage in percentage_time_spent.items():
                print("{:<30} {:<30.2f}".format(name, percentage))

    def _sample_loop(self):
        """The actual sampling code."""

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
                self.proposals,
                desc="Sampling. Acceptance rate:",
                leave=True,
            )

        # Start time for updating progressbar
        self.last_update_time = _time()

        # Capture NumPy exceptions -----------------------------------------------------
        class Log(object):
            messages = []

            def write(self, msg):
                self.messages.append(msg)

        self.log = Log()
        _numpy.seterrcall(self.log)
        _numpy.seterr(all="log")  # seterr to known value

        # Run the Markov process -------------------------------------------------------
        self.start_time = _datetime.now()
        try:
            # If the sampler is given a maximum time, start timer now
            scheduled_termination_time = 0
            if self.max_time is not None:
                scheduled_termination_time = _time() + self.max_time

            # Iterate through amount of proposals
            for self.current_proposal in self.proposals_iterator:

                # Propose a new sample.
                # The underlying method changes when different algorithms are selected.
                self._propose()

                # Evaluate acceptance criterium.
                # The underlying method changes when different algorithms are selected.
                self._evaluate_acceptance()

                # If we are on a thinning number ... (i.e. one of the non-discarded
                # samples)
                if self.current_proposal % self.online_thinning == 0:

                    # Write sample to ram array
                    self._sample_to_ram()

                    # Calculate the index this sample has after thinning
                    after_thinning = int(self.current_proposal / self.online_thinning)

                    # Check if this number is at the end of the buffer ...
                    if (
                        after_thinning % self.ram_buffer_size
                    ) == self.ram_buffer_size - 1:

                        # If so, write samples to disk
                        self._samples_to_disk()

                # Update the progressbar
                self._update_progressbar()

                # Check elapsed time
                if self.max_time is not None and scheduled_termination_time < _time():
                    # Raise KeyboardInterrupt if we're over time
                    raise TimeoutError

        except KeyboardInterrupt:  # Catch SIGINT --------------------------------------
            # Assume current proposal couldn't be finished, so ignore it.
            self.current_proposal -= 1
            # Close progressbar
            self.proposals_iterator.close()
        except TimeoutError:  # Catch SIGINT --------------------------------------
            # Close progressbar
            self.proposals_iterator.close()
        finally:  # Write out the last samples not on a full buffer --------------------
            self._samples_to_disk()
            self.end_time = _datetime.now()

        self._close_sampler()

    def _open_samples_hdf5(
        self,
        name: str,
        length: int,
        dtype: str = "f8",
        nested=False,
        overwrite: bool = False,
    ) -> int:

        choice_made = False

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

        except OSError as e:
            # Catch error on file creations, likely that the file already exists
            if (
                not str(e)
                == f"Unable to create file (unable to open file: name = '{name}', "
                f"errno = 17, error message = 'File exists', flags = 15, o_flags = c2)"
            ):
                raise H5FileOpenedError(
                    "This file is already opened as HDF5 file. If you "
                    "want to write to it, close the filehandle."
                )

            # If it exists, prompt the user with a warning
            _warnings.warn(
                f"\r\nIt seems that the samples file ({name}) already exists, or the "
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
                        f"{name} also exists. (n)ew file name, (o)verwrite, (s)kip "
                        "sampling or (a)bort code? >> "
                    )
                else:
                    input_choice = input(
                        "(n)ew file name, (o)verwrite, (s)kip sampling or (a)bort "
                        "code? >> "
                    )

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
                    # User wants to abort code
                    choice_made = True
                    raise FileExistsError(
                        "Aborting code execution due to samples file existing."
                    )

                elif input_choice == "s":
                    # User wants to abort sampling, but continue code
                    choice_made = True
                    raise FileExistsError(
                        "Skipping sampling due to samples file existing. Code "
                        "execution continues."
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

        if (
            self.current_proposal == 0
            or (_time() - self.last_update_time) > self.progressbar_refresh_rate
            or self.current_proposal == self.proposals - 1
        ):
            self.last_update_time = _time()

            # Calculate acceptance rate
            acceptance_rate = self.accepted_proposals / (self.current_proposal + 1)

            self.proposals_iterator.set_description(
                f"Tot. acc rate: {acceptance_rate:.2f}. Progress",
                refresh=False,
            )

    def _sample_to_ram(self):
        # Calculate proposal number after thinning
        current_proposal_after_thinning = int(
            self.current_proposal / self.online_thinning
        )
        # Assert that it's an integer (if we only write on end of buffer)
        assert self.current_proposal % self.online_thinning == 0, dev_assertion_message

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
            ), dev_assertion_message
            assert (
                end >= 0 and end <= self.proposals_after_thinning + 1
            ), dev_assertion_message
            assert end >= robust_start, dev_assertion_message

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
            self.ram_buffer.fill(_numpy.nan)

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
        """An abstract method that sets up all required attributes and methods for the
        algorithm."""
        pass

    @_abstractmethod
    def _close_sampler_specific(self):
        """An abstract method that does any post-sampling operations for the
        algorithm."""
        pass

    def get_diagnostics(self):
        if not self.diagnostic_mode:
            raise InvalidCaseError(
                "Can't return diagnostics if sampler is not in diagnostic mode"
            )
        return self.functions_to_diagnose + self.sampler_specific_functions_to_diagnose


class RWMH(_AbstractSampler):
    stepsize: _Union[float, _numpy.ndarray] = 1.0
    """A parameter describing the standard deviation of a multivariate normal (MVN) used
    as the proposal distribution for Random Walk Metropolis-Hastings. Using a
    _numpy.ndarray column vector (shape dimensions × 1) will give every dimensions a
    unique step length. Correlations in the MVN are not yet implemented. Has a strong
    influence on acceptance rate. **An essential tuning parameter.**"""

    autotuning: bool = False
    """A boolean indicating if the stepsize is tuned towards a specific acceptance rate
    using diminishing adaptive parameters."""

    learning_rate: float = 0.75
    """A float tuning the rate of decrease at which a new step size is adapted,
    according to rate = (iteration_number) ** (-learning_rate). Needs to be larger than
    0.5 and equal to or smaller than 1.0."""

    target_acceptance_rate: float = 0.65
    """A float representing the optimal acceptance rate used for autotuning, if
    autotuning is True."""

    acceptance_rates: _numpy.ndarray = None
    """A NumPy ndarray containing all past acceptance rates. Collected if autotuning is
    True."""

    stepsizes: _numpy.ndarray = None
    """A NumPy ndarray containing all past step lengths for the RWMH algorithm.
    Collected if autotuning is True."""

    minimal_stepsize: float = 1e-18
    """Minimal step length which is chosen if timestep becomes zero or negative during
    autotuning."""

    name = "Random Walk Metropolis Hastings"

    def sample(
        self,
        samples_hdf5_filename: str,
        distribution: _AbstractDistribution,
        stepsize: _Union[float, _numpy.ndarray] = 1.0,
        initial_model: _numpy.ndarray = None,
        proposals: int = 100,
        online_thinning: int = 1,
        diagnostic_mode: bool = False,
        ram_buffer_size: int = None,
        overwrite_existing_file: bool = False,
        max_time: float = None,
        autotuning: bool = False,
        target_acceptance_rate: float = 0.65,
        learning_rate: float = 0.75,
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
        stepsize: _Union[float, _numpy.ndarray]
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
        diagnostic_mode: bool
            A boolean describing if subroutines of sampling should be timed. Useful for
            finding slow parts of the algorithm. Will add overhead to each function.
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
        autotuning: bool
            A boolean describing whether or not stepsize is automatically adjusted. Uses
            a diminishing adapting scheme to satisfy detailed balance.
        target_acceptance_rate: float
            A float between 0.0 and 1.0 that is the target acceptance rate of
            autotuning. The algorithm will try to achieve this acceptance rate.
        learning_rate: float
            A float larger than 0.5 but smaller than or equal to 1.0, describing how
            aggressively the stepsize is updated. Lower is more aggresive.
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
        # We put the creation of the sampler entirely in a try/catch block, so we can
        # actually close the hdf5 file if something goes wrong.
        try:
            self._init_sampler(
                samples_hdf5_filename=samples_hdf5_filename,
                distribution=distribution,
                stepsize=stepsize,
                initial_model=initial_model,
                proposals=proposals,
                diagnostic_mode=diagnostic_mode,
                online_thinning=online_thinning,
                ram_buffer_size=ram_buffer_size,
                overwrite_existing_file=overwrite_existing_file,
                max_time=max_time,
                autotuning=autotuning,
                target_acceptance_rate=target_acceptance_rate,
                learning_rate=learning_rate,
            )
        except Exception as e:
            if self.samples_hdf5_filehandle is not None:
                self.samples_hdf5_filehandle.close()
            raise e

        self._sample_loop()

    def _init_sampler_specific(self, **kwargs):

        # Parse all possible kwargs
        for key in [
            "stepsize",
            "autotuning",
            "target_acceptance_rate",
            "learning_rate",
        ]:
            setattr(self, key, kwargs[key])
            kwargs.pop(key)

        # Autotuning -------------------------------------------------------------------
        if self.autotuning:
            assert self.learning_rate > 0.5 and self.learning_rate <= 1.0, (
                f"The learning rate should be larger than 0.5 and smaller than or "
                f"equal to 1.0, otherwise the Markov chain does not converge. Chosen: "
                f"{self.learning_rate}"
            )
            assert type(self.stepsize) == float, (
                "Autotuning RWMH is only implemented for scalar stepsizes. If you need"
                "it for non-scalar steps, write us an email."
            )
            self.acceptance_rates = _numpy.empty((self.proposals, 1))
            self.stepsizes = _numpy.empty((self.proposals, 1))

        if len(kwargs) != 0:
            raise TypeError(
                f"Unidentified argument(s) not applicable to sampler: {kwargs}"
            )

        # Assert that step length is either a float and bigger than zero, or a full
        # matrix / diagonal
        try:
            self.stepsize = float(self.stepsize)
            assert self.stepsize > 0.0, (
                "RW-MH step length should be a positive float or a numpy.ndarray. The "
                "passed argument is a float equal to or smaller than zero."
            )
        except TypeError:
            self.stepsize = _numpy.asarray(self.stepsize)
            assert type(self.stepsize) == _numpy.ndarray, (
                "RW-MH step length should be a numpy.ndarray of shape (dimensions, 1) "
                "or a positive float. The passed argument is neither."
            )
            assert self.stepsize.shape == (self.dimensions, 1), (
                "RW-MH step length should be a numpy.ndarray of shape (dimensions, 1) "
                "or a positive float. The passed argument is an ndarray of the wrong "
                "shape."
            )

        if self.diagnostic_mode:
            self.distribution.misfit = _AccumulatingTimer(self.distribution.misfit)
            self.distribution.gradient = _AccumulatingTimer(self.distribution.gradient)
            self.distribution.corrector = _AccumulatingTimer(
                self.distribution.corrector
            )
            self.sampler_specific_functions_to_diagnose = [
                self.distribution.misfit,
                self.distribution.gradient,
                self.distribution.corrector,
            ]

            if self.autotuning:
                self.autotune = _AccumulatingTimer(self.autotune)
                self.sampler_specific_functions_to_diagnose.append(self.autotune)

    def _close_sampler_specific(self):
        if self.autotuning:
            self.acceptance_rates = self.acceptance_rates[: self.current_proposal]
            self.stepsizes = self.stepsizes[: self.current_proposal]

    def _write_tuning_settings(self):
        if type(self.stepsize) == float:
            self.samples_hdf5_dataset.attrs["stepsize"] = self.stepsize
        else:
            self.samples_hdf5_dataset.attrs[
                "stepsize"
            ] = self.stepsize.__class__.__name__

    def autotune(self, acceptance_rate):
        # Write out parameters
        self.acceptance_rates[self.current_proposal] = acceptance_rate
        self.stepsizes[self.current_proposal] = self.stepsize

        # Compute weight according to diminishing scheme, see also:
        # The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian
        # Monte Carlo, Hoffman & Gelman, 2014, equation (5).
        schedule_weight = (self.current_proposal + 1) ** (-self.learning_rate)

        if _numpy.isnan(acceptance_rate):
            acceptance_rate = 0

        # Update stepsize
        self.stepsize -= schedule_weight * (
            self.target_acceptance_rate - min(acceptance_rate, 1)
        )

        if self.stepsize <= 0:
            _warnings.warn(
                "The timestep of the algorithm went below zero. You possibly "
                "started the algorithm in a region with extremely strong "
                "gradients. The sampler will now default to a minimum timestep of "
                f"{self.minimal_stepsize}. If this doesn't work, and if choosing "
                "a different initial model does not make this warning go away, try"
                "setting a smaller minimal time step and initial time step value."
            )
            self.stepsize = max(self.stepsize, self.minimal_stepsize)

    def _propose(self):

        # Propose a new model according to the MH Random Walk algorithm with a Gaussian
        # proposal distribution
        self.proposed_model = self.current_model + self.stepsize * _numpy.random.randn(
            self.dimensions, 1
        )
        assert self.proposed_model.shape == (self.dimensions, 1), dev_assertion_message

    def _evaluate_acceptance(self):

        # Compute new misfit
        self.proposed_x = self.distribution.misfit(self.proposed_model)

        acceptance_rate = _numpy.exp(self.current_x - self.proposed_x)

        if self.autotuning:
            self.autotune(acceptance_rate)

        # Evaluate MH acceptance rate
        if acceptance_rate > _numpy.random.uniform(0, 1):
            self.current_model = _numpy.copy(self.proposed_model)
            self.current_x = self.proposed_x
            self.accepted_proposals += 1

    def plot_stepsizes(self):
        import matplotlib.pyplot as _plt

        _plt.semilogy(self.stepsizes)
        _plt.xlabel("iteration")
        _plt.ylabel("stepsize")
        return _plt.gca()

    def plot_acceptance_rate(self):
        import matplotlib.pyplot as _plt

        _plt.semilogy(self.acceptance_rates)
        _plt.xlabel("iteration")
        _plt.ylabel("stepsize")
        return _plt.gca()


class HMC(_AbstractSampler):
    stepsize: float = 0.1
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

    name = "Hamiltonian Monte Carlo"
    """Sampler name."""

    autotuning: bool = False
    """A boolean indicating if the stepsize is tuned towards a specific acceptance rate
    using diminishing adaptive parameters."""

    learning_rate: float = 0.75
    """A float tuning the rate of decrease at which a new step size is adapted,
    according to rate = (iteration_number) ** (-learning_rate). Needs to be larger than
    0.5 and equal to or smaller than 1.0."""

    target_acceptance_rate: float = 0.65
    """A float representing the optimal acceptance rate used for autotuning, if
    autotuning is True."""

    acceptance_rates: _numpy.ndarray = None
    """A NumPy ndarray containing all past acceptance rates. Collected if autotuning is
    True."""

    stepsizes: _numpy.ndarray = None
    """A NumPy ndarray containing all past stepsizes for the HMC algorithm. Collected if
    autotuning is True."""

    minimal_stepsize: float = 1e-18
    """Minimal stepsize which is chosen if stepsize becomes zero or negative during
    autotuning."""

    def sample(
        self,
        samples_hdf5_filename: str,
        distribution: _AbstractDistribution,
        stepsize: float = 0.1,
        amount_of_steps: int = 10,
        mass_matrix: _AbstractMassMatrix = None,
        integrator: str = "lf",
        initial_model: _numpy.ndarray = None,
        proposals: int = 100,
        online_thinning: int = 1,
        diagnostic_mode: bool = False,
        ram_buffer_size: int = None,
        overwrite_existing_file: bool = False,
        max_time: float = None,
        autotuning: bool = False,
        target_acceptance_rate: float = 0.65,
        learning_rate: float = 0.75,
    ):
        """Sampling using the Hamiltonian Monte Carlo algorithm.

        Parameters
        ----------
        samples_hdf5_filename: str
            String containing the (path+)filename of the HDF5 file to contain the
            samples. **A required parameter.**
        distribution: _AbstractDistribution
            The distribution to sample. Should be an instance of a subclass of
            _AbstractDistribution. **A required parameter.**
        stepsize: float
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
        integrator: str
            String containing "lf", "3s" or "4s" for a leapfrog, 3-stage or 4-stage
            symplectic integrator respectively, to be used during the trajectory
            calculation.
        initial_model: _numpy
            A NumPy column vector (shape dimensions × 1) containing the starting model
            of the Markov chain. This model will not be written out as a sample.
        proposals: int
            An integer representing the amount of proposals the algorithm should make.
        online_thinning: int
            An integer representing the degree of online thinning, i.e. the interval
            between storing samples.
        diagnostic_mode: bool
            A boolean describing if subroutines of sampling should be timed. Useful for
            finding slow parts of the algorithm. Will add overhead to each function.
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
        autotuning: bool
            A boolean describing whether or not stepsize is automatically adjusted. Uses
            a diminishing adapting scheme to satisfy detailed balance.
        target_acceptance_rate: float
            A float between 0.0 and 1.0 that is the target acceptance rate of
            autotuning. The algorithm will try to achieve this acceptance rate.
        learning_rate: float
            A float larger than 0.5 but smaller than or equal to 1.0, describing how
            aggressively the stepsize is updated. Lower is more aggresive.

        Raises
        ------
        AssertionError
            For any unspecified invalid entry.
        ValueError
            For any invalid value of algorithm settings.
        TypeError
            For any invalid value of algorithm settings.


        """

        # We put the creation of the sampler entirely in a try/catch block, so we can
        # actually close the hdf5 file if something goes wrong.
        try:
            self._init_sampler(
                samples_hdf5_filename=samples_hdf5_filename,
                distribution=distribution,
                stepsize=stepsize,
                amount_of_steps=amount_of_steps,
                mass_matrix=mass_matrix,
                integrator=integrator,
                initial_model=initial_model,
                autotuning=autotuning,
                target_acceptance_rate=target_acceptance_rate,
                learning_rate=learning_rate,
                proposals=proposals,
                diagnostic_mode=diagnostic_mode,
                online_thinning=online_thinning,
                ram_buffer_size=ram_buffer_size,
                overwrite_existing_file=overwrite_existing_file,
                max_time=max_time,
            )
        except Exception as e:
            if self.samples_hdf5_filehandle is not None:
                self.samples_hdf5_filehandle.close()

            if type(e) is FileExistsError:
                if (
                    str(e) == "Skipping sampling due to samples file existing. Code "
                    "execution continues."
                ):
                    return
            raise e

        self._sample_loop()

    def _init_sampler_specific(self, **kwargs):
        # Parse all possible kwargs
        for key in (
            "stepsize",
            "amount_of_steps",
            "mass_matrix",
            "integrator",
            "autotuning",
            "target_acceptance_rate",
            "learning_rate",
        ):
            setattr(self, key, kwargs[key])
            kwargs.pop(key)

        if len(kwargs) != 0:
            raise TypeError(
                f"Unidentified argument(s) not applicable to sampler: {kwargs}"
            )

        # Autotuning -------------------------------------------------------------------
        if self.autotuning:
            assert self.learning_rate > 0.5 and self.learning_rate <= 1.0, (
                f"The learning rate should be larger than 0.5 and smaller than or "
                f"equal to 1.0, otherwise the Markov chain does not converge. Chosen: "
                f"{self.learning_rate}"
            )
            self.acceptance_rates = _numpy.empty((self.proposals, 1))
            self.stepsizes = _numpy.empty((self.proposals, 1))

        # Step length ------------------------------------------------------------------
        # Assert that step length for Hamiltons equations is a float and bigger than
        # zero
        self.stepsize = float(self.stepsize)
        assert self.stepsize > 0.0, "Stepsize should be a float larger than zero."

        # Step amount ------------------------------------------------------------------
        # Assert that number of steps for Hamiltons equations is a positive integer
        assert type(self.amount_of_steps) == int, (
            "The amount of steps (amount_of_steps) the HMC integrator should make "
            "should be an integer."
        )
        assert self.amount_of_steps > 0, (
            "The amount of steps (amount_of_steps) the HMC integrator should make "
            "should be larger than zero."
        )

        # Mass matrix ------------------------------------------------------------------
        # Set the mass matrix if it is not yet set using the default: a unit mass
        if self.mass_matrix is None:
            self.mass_matrix = _Unit(self.dimensions)

        # Assert that the mass matrix is the right type and dimension
        assert isinstance(self.mass_matrix, _AbstractMassMatrix), (
            "The passed mass matrix (mass_matrix) should be a class derived from "
            "_AbstractMassMatrix."
        )
        assert self.mass_matrix.dimensions == self.dimensions, (
            f"The passed mass matrix (mass_matrix) should have dimensions equal to the "
            f"target distribution. Passed: {self.mass_matrix.dimensions}, "
            f"required: {self.dimensions}."
        )

        # Integrator -------------------------------------------------------------------
        self.integrator = str(self.integrator)

        if self.integrator not in self.available_integrators:
            raise ValueError(
                f"Unknown integrator used. Choices are: {self.available_integrators}"
            )

        if self.diagnostic_mode:
            self.distribution.misfit = _AccumulatingTimer(self.distribution.misfit)
            self.distribution.gradient = _AccumulatingTimer(self.distribution.gradient)
            self.mass_matrix.generate_momentum = _AccumulatingTimer(
                self.mass_matrix.generate_momentum
            )
            self.mass_matrix.kinetic_energy = _AccumulatingTimer(
                self.mass_matrix.kinetic_energy
            )
            self.mass_matrix.kinetic_energy_gradient = _AccumulatingTimer(
                self.mass_matrix.kinetic_energy_gradient
            )
            self.distribution.corrector = _AccumulatingTimer(
                self.distribution.corrector
            )
            self.integrators[self.integrator] = _AccumulatingTimer(
                self.integrators[self.integrator]
            )

            self.sampler_specific_functions_to_diagnose = [
                self.distribution.misfit,
                self.distribution.gradient,
                self.integrators[self.integrator],
                self.mass_matrix.generate_momentum,
                self.mass_matrix.kinetic_energy,
                self.mass_matrix.kinetic_energy_gradient,
                self.distribution.corrector,
            ]

            if self.autotuning:
                self.autotune = _AccumulatingTimer(self.autotune)
                self.sampler_specific_functions_to_diagnose.append(self.autotune)

    def _close_sampler_specific(self):
        if self.autotuning:
            self.acceptance_rates = self.acceptance_rates[: self.current_proposal]
            self.stepsizes = self.stepsizes[: self.current_proposal]

    def _write_tuning_settings(self):
        self.samples_hdf5_dataset.attrs["stepsize"] = self.stepsize
        self.samples_hdf5_dataset.attrs["amount_of_steps"] = self.amount_of_steps
        self.samples_hdf5_dataset.attrs["mass_matrix"] = self.mass_matrix.name
        self.samples_hdf5_dataset.attrs["integrator"] = self.integrators_full_names[
            self.integrator
        ]

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

        acceptance_rate = _numpy.exp(self.current_h - self.proposed_h)

        if self.autotuning:
            self.autotune(acceptance_rate)

        if acceptance_rate > _numpy.random.uniform(0, 1):
            self.current_model = _numpy.copy(self.proposed_model)
            self.current_x = self.proposed_x
            self.accepted_proposals += 1

    def autotune(self, acceptance_rate):
        # Write out parameters
        self.acceptance_rates[self.current_proposal] = acceptance_rate
        self.stepsizes[self.current_proposal] = self.stepsize

        # Compute weight according to diminishing scheme, see also:
        # The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian
        # Monte Carlo, Hoffman & Gelman, 2014, equation (5).
        schedule_weight = (self.current_proposal + 1) ** (-self.learning_rate)

        if _numpy.isnan(acceptance_rate):
            acceptance_rate = 0

        # Update stepsize
        proposed_stepsize = self.stepsize - schedule_weight * (
            self.target_acceptance_rate - min(acceptance_rate, 1)
        )

        if proposed_stepsize <= 0:
            _warnings.warn(
                "The stepsize of the algorithm went below zero. You possibly "
                "started the algorithm in a region with extremely strong "
                "gradients. The sampler will now default to a minimum stepsize of "
                f"{self.minimal_stepsize}. If this doesn't work, and if choosing "
                "a different initial model does not make this warning go away, try"
                "setting a smaller minimal stepsize and initial stepsize value."
            )
            proposed_stepsize = max(proposed_stepsize, self.minimal_stepsize)

        if (
            _numpy.abs(_numpy.log10(proposed_stepsize) - _numpy.log10(self.stepsize))
            > 1
        ):
            if proposed_stepsize > self.stepsize:
                self.stepsize *= 10
            else:
                self.stepsize *= 0.1
        else:
            self.stepsize = proposed_stepsize

    def _propagate_leapfrog(
        self,
    ):

        # Make sure not to alter a view but a copy of arrays ---------------------------
        position = self.current_model.copy()
        momentum = self.current_momentum.copy()

        # Leapfrog integration ---------------------------------------------------------
        position += (
            0.5 * self.stepsize * self.mass_matrix.kinetic_energy_gradient(momentum)
        )

        self.distribution.corrector(position, momentum)

        # verbose_integration
        verbose_integration = False
        if verbose_integration:
            integration_iterator = _tqdm_au.trange(
                self.amount_of_steps - 1,
                position=2,
                leave=False,
                desc="Leapfrog integration",
            )
        else:
            integration_iterator = range(self.amount_of_steps - 1)

        # Integration loop
        for i in integration_iterator:
            # Momentum step
            momentum -= self.stepsize * self.distribution.gradient(position)
            # Position step
            position += self.stepsize * self.mass_matrix.kinetic_energy_gradient(
                momentum
            )
            # Correct bounds
            self.distribution.corrector(position, momentum)

        # Full momentum and half step position after loop ------------------------------
        # Momentum step
        momentum -= self.stepsize * self.distribution.gradient(position)
        # Position step
        position += (
            0.5 * self.stepsize * self.mass_matrix.kinetic_energy_gradient(momentum)
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

        a1 *= self.stepsize
        a2 *= self.stepsize
        a3 *= self.stepsize
        b1 *= self.stepsize
        b2 *= self.stepsize

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

        a1 *= self.stepsize
        a2 *= self.stepsize
        b1 *= self.stepsize
        b2 *= self.stepsize

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
    integrators_full_names = {
        "lf": "leapfrog integrator",
        "3s": "three stage integrator",
        "4s": "four stage integrator",
    }
    available_integrators = integrators.keys()

    def plot_stepsizes(self):
        import matplotlib.pyplot as _plt

        _plt.semilogy(self.stepsizes)
        _plt.xlabel("iteration")
        _plt.ylabel("stepsize")
        return _plt.gca()

    def plot_acceptance_rate(self):
        import matplotlib.pyplot as _plt

        _plt.semilogy(self.acceptance_rates)
        _plt.xlabel("iteration")
        _plt.ylabel("acceptance rate")
        return _plt.gca()
