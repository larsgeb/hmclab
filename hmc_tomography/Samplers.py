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
import sys as _sys
from abc import ABC as _ABC
from abc import abstractmethod as _abstractmethod

import h5py as _h5py
import numpy as _numpy
import time as _time
import tqdm.auto as _tqdm_au
import warnings as _warnings
from typing import Tuple as _Tuple
from typing import Union as _Union

from hmc_tomography.Distributions import _AbstractDistribution
from hmc_tomography.MassMatrices import _AbstractMassMatrix
from hmc_tomography.MassMatrices import Unit as _Unit

from hmc_tomography.Helpers.CustomExceptions import (
    AbstractMethodError as _AbstractMethodError,
)
from hmc_tomography.Helpers.CustomExceptions import (
    InvalidCaseError as _InvalidCaseError,
)


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

    samples_ram_buffer: _numpy.ndarray = None
    """A NumPy ndarray containing the samples that are as of yet not written to disk."""

    current_model: _numpy.ndarray = None
    """A NumPy array containing the model at the current state of the Markov chain."""

    proposed_model: _numpy.ndarray = None
    """A NumPy array containing the model at the proposed state of the Markov chain."""

    current_x: float = None
    """A NumPy array containing :math:`\chi = -\log\left( p\\right)` (i.e. the misfit, 
    negative log probability) of the distribution at the current state of the Markov
    chain."""

    proposed_x: float = None
    """A NumPy array containing :math:`\chi = -\log\left( p\\right)` (i.e. the misfit, 
    negative log probability) of the distribution at the proposed state of the Markov
    chain."""

    accepted_proposals: int = None
    """An integer represeting the amount of accepted proposals."""

    amount_of_writes: int = None
    """An integer represeting the amount of times the sampler has written to disk."""

    def __init__(self):
        self.sample = self._sample

    def _init_sampler(
        self,
        samples_hdf5_filename: str,
        distribution,
        initial_model,
        proposals,
        online_thinning,
        samples_ram_buffer_size,
        overwrite_existing_file,
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

        if samples_ram_buffer_size is not None:
            # Assert that samples_ram_buffer_size is a positive integer
            assert samples_ram_buffer_size > 0
            assert type(samples_ram_buffer_size) == int

            # Assert that the total generated proposals are a multiple of this number
            try:
                assert self.proposals_after_thinning % samples_ram_buffer_size == 0
                self.samples_ram_buffer_size = samples_ram_buffer_size
            except AssertionError as e:
                # That doesn't fit nicely in the amount of proposals, let's make the
                # block bigger until it fits
                while self.proposals_after_thinning % samples_ram_buffer_size != 0:
                    samples_ram_buffer_size = samples_ram_buffer_size - 1
                self.samples_ram_buffer_size = samples_ram_buffer_size

                if self.samples_ram_buffer_size > 1e4:
                    _warnings.warn(
                        f"\r\nSample RAM buffer is incorrectly sized. Resizing such "
                        f"that it is a multiple of the amount of proposals that are "
                        f"written to disk (proposals divided by online thinning): "
                        f"{self.samples_ram_buffer_size}. \r\n\r\nResizing could not "
                        f"be done any smaller. This is a very large number of samples "
                        f"to keep in RAM. Consider altering the number of proposals to "
                        f"a multiple of thousands.\r\n\r\n",
                        Warning,
                        stacklevel=100,
                    )
                else:
                    _warnings.warn(
                        f"\r\nSample RAM buffer is incorrectly sized. Resizing such "
                        f"that it is a multiple of the amount of proposals that are "
                        f"written to disk (proposals divided by online thinning): "
                        f"{self.samples_ram_buffer_size}.",
                        Warning,
                        stacklevel=100,
                    )

        else:
            # This is all automated stuff. You can force any size by setting it
            # manually.

            # Detailed explanation: we strive for approximately 1 gigabyte in
            # memory before we write to disk, by default. The amount of floats that are
            # in memory is calculated as follows: (dimensions + 1) *
            # samples_ram_buffer_size. The plus ones comes from requiring to store the
            # misfit. 1 gigabyte is approximately 1e8 64 bits floats (actually 1.25e8).
            # Additionally, there is a cap at 10000 samples.
            samples_ram_buffer_size = min(
                int(_numpy.floor(1e8 / self.dimensions)), 10000
            )
            # Reduce this number until it fits in the amount of proposals
            while self.proposals_after_thinning % samples_ram_buffer_size != 0:
                samples_ram_buffer_size = samples_ram_buffer_size - 1

            # Now, this number might be larger than the actual amount of samples, so we
            # take the minimum of this and the amount of proposals to write as the
            # actual ram size.
            self.samples_ram_buffer_size = min(
                samples_ram_buffer_size, self.proposals_after_thinning
            )

            assert type(self.samples_ram_buffer_size) == int

        shape = (self.dimensions + 1, self.samples_ram_buffer_size)

        self.samples_ram_buffer = _numpy.empty(shape, dtype=_numpy.float64)

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

        # Do specific stuff

        self._init_sampler_specific(**kwargs)

        self._write_tuning_settings()

    @_abstractmethod
    def _init_sampler_specific(self):
        """An abstract method that sets up all required attributes and method for the 
        algorithm."""
        pass

    def _close_sampler(self):
        self.samples_hdf5_filehandle.close()

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
        except:
            self.proposals_iterator = _tqdm_au.trange(
                self.proposals, desc="Sampling. Acceptance rate:", leave=True,
            )

        # Run the Markov process -------------------------------------------------------
        for self.current_proposal in self.proposals_iterator:

            # Propose a new sample
            self._propose()

            # Evaluate acceptance criterion
            self._evaluate_acceptance()

            # Write sample to RAM (and disk if needed)
            self._flush_sample()

            # Update the progressbar
            self._update_progressbar()

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
            self.samples_hdf5_filehandle = _h5py.File(name, flag)

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
                        f"{name} also exists. (n)ew file name, (o)verwrite or (a)bort? >> "
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
                    self.samples_hdf5_filehandle = _h5py.File(name, "w")

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
            (self.dimensions + 1, length),  # one extra for misfit
            dtype=dtype,
            chunks=True,
        )

    def _update_progressbar(self):

        # Calculate acceptance rate
        acceptance_rate = self.accepted_proposals / (self.current_proposal + 1)

        if self.current_proposal % 1000:
            self.proposals_iterator.set_description(
                f"Tot. acc rate: {acceptance_rate:.2f}. Progress", refresh=False,
            )

    def _flush_sample(self):
        # Calculate proposal number after thinning
        current_proposal_after_thinning = int(
            self.current_proposal / self.online_thinning
        )

        # Assert that it's an integer
        if self.current_proposal % self.online_thinning == 0:

            # Calculate index for the RAM array
            index_in_ram = (
                current_proposal_after_thinning % self.samples_ram_buffer_size
            )

            # Place samples in RAM
            self.samples_ram_buffer[:-1, index_in_ram] = self.current_model[:, 0]

            # Place misfit in RAM
            self.samples_ram_buffer[-1, index_in_ram] = self.current_x

            # Check if the buffer is full
            if index_in_ram == self.samples_ram_buffer_size - 1:
                # Write samples to disk
                self._samples_to_disk()

    def _samples_to_disk(self):

        # Calculate proposal number after thinning
        current_proposal_after_thinning = int(
            self.current_proposal / self.online_thinning
        )

        # Calculate start/end indices
        start = current_proposal_after_thinning - self.samples_ram_buffer_size + 1
        end = current_proposal_after_thinning + 1

        # Some sanity checks on the indices
        assert start >= 0 and start <= self.proposals_after_thinning + 1
        assert end >= 0 and end <= self.proposals_after_thinning + 1
        assert end >= start

        self.amount_of_writes += 1

        # Write samples to disk
        self.samples_hdf5_dataset[:, start:end] = self.samples_ram_buffer

        # Reset the marker in the HDF5 file
        self.samples_hdf5_dataset.attrs["end_of_samples"] = self.current_proposal

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
        samples_ram_buffer_size: int = None,
        overwrite_existing_file: bool = False,
        **kwargs,
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
        samples_ram_buffer_size: int
            An integer representing how many samples should be kept in RAM before 
            writing to storage.
        overwrite_existing_file: bool
            A boolean describing whether or not to silently overwrite existing files. 
            Use with caution.
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
            samples_hdf5_filename,
            distribution,
            initial_model,
            proposals,
            online_thinning,
            samples_ram_buffer_size,
            overwrite_existing_file,
            **kwargs,
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
        samples_ram_buffer_size: int = None,
        overwrite_existing_file: bool = False,
        **kwargs,
    ):
        """Simply a forward to the instance method of the sampler; check out
        _sample()."""

        return cls()._sample(
            samples_hdf5_filename,
            distribution,
            initial_model,
            proposals,
            online_thinning,
            samples_ram_buffer_size,
            overwrite_existing_file,
            step_length,
            **kwargs,
        )

    def _init_sampler_specific(self, **kwargs):

        # Parse all possible kwargs
        for key in ("step_length", "example_extra_option"):
            if key in kwargs:
                setattr(self, key, kwargs[key])

        # Assert that step length is either a float and bigger than zero, or a full
        # matrix / diagonal
        try:
            self.step_length = float(self.step_length)
            assert self.step_length > 0.0
        except TypeError as e:
            assert type(self.step_length) == _numpy.ndarray
            assert self.step_length.shape == (self.dimensions, 1)

    def _write_tuning_settings(self):
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

        # Evaluate acceptence rate
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
        initial_model: _numpy.ndarray = None,
        proposals: int = 100,
        online_thinning: int = 1,
        samples_ram_buffer_size: int = None,
        overwrite_existing_file: bool = False,
        **kwargs,
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
        samples_ram_buffer_size: int
            An integer representing how many samples should be kept in RAM before 
            writing to storage.
        overwrite_existing_file: bool
            A boolean describing whether or not to silently overwrite existing files. 
            Use with caution.
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
            samples_hdf5_filename,
            distribution,
            initial_model,
            proposals,
            online_thinning,
            samples_ram_buffer_size,
            overwrite_existing_file,
            **kwargs,
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
        initial_model: _numpy.ndarray = None,
        proposals: int = 100,
        online_thinning: int = 1,
        samples_ram_buffer_size: int = None,
        overwrite_existing_file: bool = False,
        **kwargs,
    ):
        """Simply a forward to the instance method of the sampler; check out
        _sample()."""

        return cls()._sample(
            samples_hdf5_filename,
            distribution,
            initial_model,
            proposals,
            online_thinning,
            samples_ram_buffer_size,
            overwrite_existing_file,
            **kwargs,
        )

    def _init_sampler_specific(self, **kwargs):
        # Parse all possible kwargs
        for key in (
            "time_step",
            "amount_of_steps",
            "mass_matrix",
        ):
            if key in kwargs:
                setattr(self, key, kwargs[key])

        # Assert that step length for Hamiltons equations is a float and bigger than
        # zero
        self.time_step = float(self.time_step)
        assert self.time_step > 0.0

        # Assert that number of steps for Hamiltons equations is a positive integer
        assert type(self.amount_of_steps) == int
        assert self.amount_of_steps > 0

        # Set the mass matrix if it is not yet set using the default: a unit mass
        if self.mass_matrix is None:
            self.mass_matrix = _Unit(self.dimensions)

        # Assert that the mass matrix is the right type and dimension
        assert isinstance(self.mass_matrix, _AbstractMassMatrix)
        assert self.mass_matrix.dimensions == self.dimensions

    def _write_tuning_settings(self):
        pass

    def _propose(self):
        pass

    def _evaluate_acceptance(self):
        pass
