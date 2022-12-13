"""Sampler classes and associated methods.

The classes in this module provide different sampling algorithms to appraise
distributions. All of them are designed to work in a minimal way; you can run the
sampling method with only a target distribution and filename to write your samples to.
However, the true power of any algorithm only shows when the user injects his expertise
through tuning parameters.

Sampling can be initialised from an instance of a sampler:

.. code-block:: python

    from hmclab import HMC

    HMC_instance = HMC()

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
from typing import List as _List
from typing import Dict as _Dict
from multiprocess import (
    Process as _Process,
    Pipe as _Pipe,
    Value as _Value,
    Queue as _Queue,
)
import copy as _copy
import h5py as _h5py
from matplotlib import pyplot as _plt
import numpy as _numpy
import tqdm.auto as _tqdm_au

from hmclab.Distributions import _AbstractDistribution
from hmclab.MassMatrices import Unit as _Unit
from hmclab.MassMatrices import _AbstractMassMatrix
from hmclab.Helpers.Timers import AccumulatingTimer as _AccumulatingTimer
from hmclab.Helpers.CustomExceptions import InvalidCaseError as _InvalidCaseError
from hmclab.Samples import Samples as _Samples

import ipywidgets as _widgets
from IPython.display import display as _display

from hmclab.Helpers.CustomExceptions import AbstractMethodError as _AbstractMethodError

dev_assertion_message = (
    "Something went wrong internally, please report this to the developers."
)


class H5FileOpenedError(FileExistsError):
    """An internal exception that helps us keep track of H5 files."""

    pass


class _AbstractSampler(_ABC):
    """Abstract base class for Markov chain Monte Carlo samplers."""

    # Essential attributes -------------------------------------------------------------

    name: str = "Monte Carlo sampler abstract base class"
    """The name of the sampler"""

    rng = None

    dimensions: int = None
    """An integer representing the dimensionality in which the MCMC sampler works."""

    distribution: _AbstractDistribution = None
    """The _AbstractDistribution-derived object on which the sampler works."""

    samples_filename: str = None
    """A string containing the path+filename of the hdf5 file to which samples will be
    stored."""

    current_model: _numpy.ndarray = None
    """A NumPy array containing the model at the current state of the Markov chain. """

    proposed_model: _numpy.ndarray = None
    """A NumPy array containing the model at the proposed state of the Markov chain. """

    current_x: float = None
    r"""A float containing :math:`\chi = -\log\left( p\\right)` (i.e. the misfit,
    negative log probability) of the distribution at the current state of the
    Markov chain."""

    proposed_x: float = None
    r"""A float containing :math:`\chi = -\log\left( p\\right)` (i.e. the misfit,
    negative log probability) of the distribution at the proposed state of the
    Markov chain. """

    # Statistics attributes ------------------------------------------------------------

    accepted_proposals: int = None
    """A positive integer representing the amount of accepted proposals."""

    amount_of_writes: int = None
    """A positive integer representing the amount of times the sampler has written to
    disk."""

    progressbar_refresh_rate: float = 0.25
    """A float representing how many seconds lies between an update of the progress bar
    statistics (acceptance rate etc.)."""

    max_time: float = None
    """A float representing the maximum time in seconds that sampling is allowed to take
    before it is automatically terminated. The value None is used for unlimited time."""

    times_started: int = 0
    """An integer representing how often this sampler object has started sampling."""

    proposals: int = None
    """An integer representing the amount of requested proposals for a run."""

    online_thinning: int = None
    """An integer representing the interval between stored samples. Defaults to 1, i.e. 
    all samples are stored."""

    current_proposal: int = 0
    """An integer representing the current proposal index (before thinning) of the 
    Markov chain."""

    current_proposal_after_thinning: int = 0
    """An integer representing the current proposal index after thinning of the 
    Markov chain."""

    accepted_proposals = 0
    """An integer representing the amount of accepted proposals."""

    disable_progressbar: bool = False
    """A bool describing whether or not to disable the TQDM progressbar"""

    samples: _Samples = None

    # distribution = type("NoDistribution", (object,), {"name": None})
    # """The distribution """
    # TODO: assert why this was in the class

    # Diagnostics attributes -----------------------------------------------------------

    end_time: _datetime = None
    """Time (and date) at which sampling was terminated / finished."""

    start_time: _datetime = None
    """Time (and date) at which sampling was started."""

    diagnostic_mode: bool = True
    """Boolean describing whether functions need to be profiled for performance."""

    # TODO: figure out how to do _List[fn]
    functions_to_diagnose: _List = []
    """Array of functions to be profiled, typically contains functions only from the
    abstract base class."""

    sampler_specific_functions_to_diagnose: _List = []
    """Array of sampler specific functions to be profiled."""

    main_thread_keyboard_interrupt = None
    # TODO: figure out what this does

    # Parallel sampling attributes -----------------------------------------------------
    parallel: bool = False
    """A boolean describing if this sampler works in an array of multiple samplers."""

    sampler_index: int = 0
    """An integer describing the index of the parallel sampler in the array of samplers.
    Essential for two-way communication."""

    # TODO: add type hint
    exchange_schedule = None
    """A pre-communicated exchange schedule determining when and which samplers try to
    exchange states."""

    # TODO: add type hint
    pipe_matrix = None
    """A matrix of two-way communicators between samplers. Indexed using
    `sampler_index.`"""

    exchange_interval: int = None
    """Integer describing how many samples lie between the attempted swap of states
    between samplers."""

    def __str__(self) -> str:
        """Method for converting a sampler object to string, handy in outputs. Works for
        derived classes."""
        return f"An instance of the {self.name} sampler object."

    def _widget_data(self) -> _Dict:
        """Method for returning post-sampling summary data to be displayed in e.g.
        Jupyter widgets."""

        # Run details (panel 1) --------------------------------------------------------
        run_details = {}
        proposed_samples = self.current_proposal if self.current_proposal > 0 else None
        acceptance_rate = self.accepted_proposals / (self.current_proposal + 1)
        if not (self.start_time is None or self.end_time is None):
            runtime = (self.end_time - self.start_time).total_seconds()
            run_details["local start time (not timezone aware)"] = self.start_time
            run_details["runtime (seconds)"] = runtime
            run_details["proposals per seconds"] = proposed_samples / runtime
        written_samples = (
            self.current_proposal_after_thinning + 1
            if self.current_proposal_after_thinning > 0
            else None
        )

        run_details["acceptance rate"] = acceptance_rate
        run_details["output file"] = self.samples_filename
        run_details["proposals made (excluding first position)"] = proposed_samples
        run_details["samples written (after online thinning)"] = written_samples
        run_details["amount of writes"] = self.amount_of_writes
        run_details["dimensions"] = self.dimensions
        run_details["distribution"] = str(
            (
                self.distribution.name
                if (self.distribution.name is not None)
                else self.distribution
            )
            if self.distribution is not None
            else None
        )

        # Tuning settings (panel 2) ----------------------------------------------------
        settings = {}
        settings["proposals"] = self.proposals
        settings["online thinning (store every ...-th sample)"] = self.online_thinning

        # Combine with algorithm specific settings
        settings = {**settings, **self._tuning_settings()}

        # Algorithm (panel 3) ----------------------------------------------------------
        algorithm = {}
        algorithm["algorithm used"] = self.name
        algorithm["diagnostic mode"] = self.diagnostic_mode
        algorithm["this object has started sampling ... times"] = self.times_started

        return {
            "Details of last run": run_details,
            "Tuning settings": settings,
            "Algorithm": algorithm,
        }

    def print_results(self) -> None:
        """Print Jupyter widget from `_repr_html_()` to stdout."""
        print(self._repr_html_())

    def _repr_html_(
        self, nested: bool = False, widget_data: _Dict = None
    ) -> _Union[None, _widgets.Tab]:
        """Create a Jupyter widget with the sampling results and statistics."""

        default_layout = _widgets.Layout(padding="10px")

        # Helper function to make a Tab from a Python Dictionary
        def dictionary_to_widget(dictionary):
            left_column = _widgets.VBox(
                [_widgets.Label(str(key)) for key in dictionary.keys()],
                layout=default_layout,
            )
            right_column = _widgets.VBox(
                [_widgets.Label(str(key)) for key in dictionary.values()],
                layout=default_layout,
            )
            return _widgets.HBox([left_column, right_column], layout=default_layout)

        # Obtain sampling data
        if widget_data is None:
            widget_data = self._widget_data()

        # Create tab object
        tab = _widgets.Tab()

        # Populate children with sampling data
        tab.children = [
            dictionary_to_widget(panel_data) for panel_data in widget_data.values()
        ]
        panel_headings = [key for key in widget_data.keys()]
        for i in range(len(tab.children)):
            tab.set_title(i, panel_headings[i])

        # Return results, or print and return nothing.
        if nested:
            # This is used when multiple samplers ran in parallel, and the tabs of each
            # individual sampler still need to be combined.
            return tab
        else:
            _display(tab)
            return ""

    def __init__(
        self,
        seed: int = None,
    ) -> None:
        """Constructor that sets the random number generator for the sampler. Pass a
        seed to make a Markov chain reproducible (given that all settings are re-used.)

        Parameters
        ----------
        seed : int
            Description of parameter `seed`.


        """
        if seed is None:
            self.rng = _numpy.random.default_rng()
        else:
            self.rng = _numpy.random.default_rng(seed)

    def _init_sampler(
        self,
        samples_filename: str,
        distribution: _AbstractDistribution,
        initial_model: _numpy.ndarray,
        proposals: int,
        online_thinning: int,
        overwrite_existing_file: bool,
        max_time: int,
        disable_progressbar: bool = False,
        diagnostic_mode: bool = False,
        **kwargs,
    ):
        """A method that is called everytime any markov chain sampler object is
        constructed.

        Args:
            samples_filename ([type]): [description]
            distribution ([type]): [description]
            initial_model ([type]): [description]
            proposals ([type]): [description]
            online_thinning ([type]): [description]
            overwrite_existing_file ([type]): [description]
            max_time ([type]): [description]
        """

        assert type(samples_filename) == str, (
            f"First argument should be a string containing the path of the file to "
            f"which to write samples. It was an object of type "
            f"{type(samples_filename)}."
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

        # Set up the samples file ------------------------------------------------------

        assert (
            type(samples_filename) == str
        ), "The samples filename needs to be a string."
        self.samples_filename = samples_filename
        # Open the samples file
        self.samples = _Samples(
            samples_filename, mode="w", overwrite=overwrite_existing_file
        )

        # Set up the initial model and preallocate other necessary arrays --------------

        if initial_model is None:
            initial_model = _numpy.zeros((self.dimensions, 1))
        else:
            initial_model = _numpy.array(initial_model)
            initial_model.shape = (initial_model.size, 1)

        assert initial_model.shape == (self.dimensions, 1), (
            f"The initial model (`initial_model`) dimension is incompatible with the "
            f"target distribution. Supplied model shape: {initial_model.shape}."
            f"Required shape: {(self.dimensions, 1)}"
        )

        self.current_model = initial_model.astype(_numpy.float64)
        self.current_x = distribution.misfit(self.current_model)

        assert not _numpy.isnan(
            self.current_x
        ), "Initial position in model space gives NaN probability"
        assert not _numpy.isinf(
            self.current_x
        ), "Initial position in model space gives inf/-inf probability"

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

        self.disable_progressbar = disable_progressbar

        # Prepare diagnostic mode if needed --------------------------------------------
        self.diagnostic_mode = diagnostic_mode

        if self.diagnostic_mode:
            self._propose = _AccumulatingTimer(self._propose)
            self._evaluate_acceptance = _AccumulatingTimer(self._evaluate_acceptance)
            self.samples.append = _AccumulatingTimer(self.samples.append)
            self._update_progressbar = _AccumulatingTimer(self._update_progressbar)

            self.functions_to_diagnose = [
                self._propose,
                self._evaluate_acceptance,
                self.samples.append,
                self._update_progressbar,
            ]

        # Do sampler specific operations -----------------------------------------------

        # Set up specifics for each algorithm
        self._init_sampler_specific(**kwargs)

        # Write out the tuning settings
        self._write_tuning_settings()

        # Create attributes before sampling, such that SWMR works
        self.samples.write_attribute("write_index", -1)
        self.samples.write_attribute("last_written_sample", -1)
        self.samples.write_attribute("proposals", self.proposals)
        self.samples.write_attribute("acceptance_rate", 0)
        self.samples.write_attribute("online_thinning", self.online_thinning)
        self.samples.write_attribute(
            "start_time", _datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
        )
        self.samples.write_attribute("sampler", self.name)

    def _close_sampler(self):

        self.samples.write_attribute(
            "acceptance_rate", self.accepted_proposals / (self.current_proposal + 1)
        )

        self.samples.write_attribute(
            "end_time", _datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
        )
        self.samples.write_attribute("runtime", str(self.end_time - self.start_time))
        self.samples.write_attribute(
            "runtime_seconds", (self.end_time - self.start_time).total_seconds()
        )

        self._close_sampler_specific()

        self.samples.close()

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

        # This avoids some weird stdout bugs on OSX. Doesn't do anythign than
        # fiddle with the stdout.
        print(" ", end="", flush=True)

        # Create progressbar -----------------------------------------------------------
        try:
            self.proposals_iterator = _tqdm_au.trange(
                self.proposals,
                desc=f"Tot. acc rate: {0:.2f}. Progress",
                leave=True,
                dynamic_ncols=True,
                position=self.sampler_index,  # only relevant for parallel sampling
                disable=self.disable_progressbar,
            )
        except Exception:
            self.proposals_iterator = _tqdm_au.trange(
                self.proposals,
                desc=f"Tot. acc rate: {0:.2f}. Progress",
                leave=True,
                position=self.sampler_index,  # only relevant for parallel sampling
                disable=self.disable_progressbar,
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
        self.times_started += 1
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

                # Parallel communication section -------------
                if self.parallel and (
                    self.exchange_interval is not None
                ):  # pragma: no cover
                    # Check if chain is in the schedule for current iteration
                    if self.current_proposal % self.exchange_interval == 0 and (
                        self.sampler_index
                        in self.exchange_schedule[
                            int(self.current_proposal / self.exchange_interval), :
                        ]
                    ):

                        # Find where in schedule
                        position_in_schedule = _numpy.where(
                            self.sampler_index
                            == self.exchange_schedule[
                                int(self.current_proposal / self.exchange_interval), :
                            ]
                        )[0][0]

                        # Determine if master (master evaluates swap)
                        if position_in_schedule % 2 == 0:
                            master = False
                        else:
                            master = True

                        # Find counterpart
                        exchange_chain = self.exchange_schedule[
                            int(self.current_proposal / self.exchange_interval),
                            position_in_schedule + (-1 if master else +1),
                        ]

                        # Find correct pipes
                        left_pipe, right_pipe = self.pipe_matrix.retrieve_pipes(
                            self.sampler_index, exchange_chain
                        )

                        if self.sampler_index > exchange_chain:
                            pipe = left_pipe
                        else:
                            pipe = right_pipe

                        # Master sends model, then waits for misfit and other chains'
                        # model + misfit
                        if master:
                            (exchange_model,) = pipe.recv()
                            pipe.send([self.current_model])

                            # Compute misfit of received model
                            exchange_x = self.distribution.misfit(exchange_model)

                            # Compute misfit improvement
                            misfit_improvement = self.current_x - exchange_x

                            # Receive improvement of misfit of the counterpart
                            (counterpart_improvement,) = pipe.recv()

                            # Evaluate acceptance rate of exchange, based on both
                            # improvements.
                            if _numpy.exp(
                                misfit_improvement + counterpart_improvement
                            ) > self.rng.uniform(0, 1):
                                # If accepted, switch models
                                pipe.send([self.current_model])
                                self.current_model = exchange_model.copy()
                            else:
                                # If not accepted, send the exchanged model back to
                                # other chain, effectively not switching.
                                pipe.send([exchange_model])

                        else:
                            pipe.send([self.current_model])
                            (exchange_model,) = pipe.recv()

                            exchange_x = self.distribution.misfit(exchange_model)

                            misfit_improvement = self.current_x - exchange_x

                            pipe.send([misfit_improvement])

                            (self.current_model,) = pipe.recv()

                # --------------------------------------------

                # If we are on a thinning number ... (i.e. one of the non-discarded
                # samples)
                if self.current_proposal % self.online_thinning == 0:
                    self.samples.append(
                        _numpy.vstack([self.current_model, self.current_x])
                    )

                # Update the progressbar
                self._update_progressbar()

                # Check elapsed time
                if self.max_time is not None and scheduled_termination_time < _time():
                    # Raise TimeoutError if we're over time
                    raise TimeoutError

        except KeyboardInterrupt:  # Catch SIGINT --------------------------------------
            # Assume current proposal couldn't be finished, so ignore it.
            self.current_proposal -= 1
        except TimeoutError:  # Catch SIGINT -------------------------------------------
            pass
        except Exception as e:
            # Any other exception, we don't know how to handle
            self.proposals_iterator.close()
            self.proposals_iterator = None
            self.end_time = _datetime.now()
            self._close_sampler()
            raise e
        finally:
            self.proposals_iterator.close()
            self.proposals_iterator = None
            self.end_time = _datetime.now()
            self._close_sampler()

    def _update_progressbar(self):

        if (
            self.current_proposal == 0
            or (_time() - self.last_update_time) > self.progressbar_refresh_rate
            or self.current_proposal == self.proposals - 1
        ):
            self.last_update_time = _time()

            # Calculate acceptance rate
            acceptance_rate = self.accepted_proposals / (self.current_proposal + 1)

            if self.parallel:
                self.proposals_iterator.set_description(
                    f"Sampler {self.sampler_index}, tot. acc rate: "
                    f"{acceptance_rate:.2f}. Progress",
                    refresh=False,
                )
            else:
                self.proposals_iterator.set_description(
                    f"Tot. acc rate: {acceptance_rate:.2f}. Progress",
                    refresh=False,
                )

    @_abstractmethod
    def _propose(self):
        raise _AbstractMethodError()

    @_abstractmethod
    def _evaluate_acceptance(self):
        """This abstract method evaluates the acceptance criterion in the MCMC
        algorithm. Pass or fail, it updates the objects attributes accordingly;
        modifying current_model and current_x as needed."""
        raise _AbstractMethodError()

    @_abstractmethod
    def _write_tuning_settings(self):
        """An abstract method that writes all the relevant tuning settings of the
        algorithm to the HDF5 file."""
        raise _AbstractMethodError()

    @_abstractmethod
    def _init_sampler_specific(self):
        """An abstract method that sets up all required attributes and methods for the
        algorithm."""
        raise _AbstractMethodError()

    @_abstractmethod
    def _close_sampler_specific(self):
        """An abstract method that does any post-sampling operations for the
        algorithm."""
        raise _AbstractMethodError()

    def get_diagnostics(self):
        if not self.diagnostic_mode:
            raise _InvalidCaseError(
                "Can't return diagnostics if sampler is not in diagnostic mode"
            )
        return self.functions_to_diagnose + self.sampler_specific_functions_to_diagnose

    def load_results(self, burn_in: int = 0) -> _numpy.array:
        assert burn_in >= 0

        from hmclab.Samples import Samples as _Samples

        with _Samples(self.samples_filename, burn_in=burn_in) as samples:
            samples_numpy = samples.numpy

        return samples_numpy


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

    _stepsize_non_scalar_part = 1.0

    name = "Random Walk Metropolis Hastings"

    def sample(
        self,
        samples_filename: str,
        distribution: _AbstractDistribution,
        stepsize: _Union[float, _numpy.ndarray] = 1.0,
        initial_model: _numpy.ndarray = None,
        proposals: int = 100,
        online_thinning: int = 1,
        diagnostic_mode: bool = False,
        overwrite_existing_file: bool = False,
        max_time: float = None,
        autotuning: bool = False,
        target_acceptance_rate: float = 0.65,
        learning_rate: float = 0.75,
        queue=None,
        disable_progressbar=False,
    ):
        """Sampling using the Metropolis-Hastings algorithm.

        Parameters
        ----------
        samples_filename: str
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
                samples_filename=samples_filename,
                distribution=distribution,
                stepsize=stepsize,
                initial_model=initial_model,
                proposals=proposals,
                diagnostic_mode=diagnostic_mode,
                online_thinning=online_thinning,
                overwrite_existing_file=overwrite_existing_file,
                max_time=max_time,
                autotuning=autotuning,
                target_acceptance_rate=target_acceptance_rate,
                learning_rate=learning_rate,
                disable_progressbar=disable_progressbar,
            )
        except Exception as e:
            if self.samples is not None:
                self.samples.close()
            raise e

        self._sample_loop()

        if self.parallel:
            queue.put({f"{self.sampler_index}": self._widget_data()})
            queue.close()

        return self

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

            if type(self.stepsize) == _numpy.ndarray:
                self._stepsize_non_scalar_part = self.stepsize
                self.stepsize = 1.0

            assert type(self.stepsize) == float, (
                "Autotuning RWMH is only implemented for scalar stepsizes. If you need "
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

            # Also write these to the hdf5 dataset
            self.samples.write_attribute("acceptance_rates", self.acceptance_rates)
            self.samples.write_attribute("stepsizes", self.stepsizes)

    def _write_tuning_settings(self):

        if type(self.stepsize) == float:
            stepsize_write = self.stepsize
        else:
            stepsize_write = self.stepsize.__class__.__name__

        self.samples.write_attribute("stepsize", stepsize_write)

    def _tuning_settings(self):

        settings = {
            "step size"
            if not self.autotuning
            else "final step size after tuning": str(self.stepsize)
        }

        if self.autotuning:
            settings = {
                **settings,
                **{
                    "autotuning": self.autotuning,
                    "learning_rate": self.learning_rate,
                    "target_acceptance_rate": self.target_acceptance_rate,
                    "minimal_stepsize": self.minimal_stepsize,
                },
            }

        return settings

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
            if self.diagnostic_mode:
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
        self.proposed_model = (
            self.current_model
            + self.stepsize
            * self._stepsize_non_scalar_part
            * self.rng.normal(size=(self.dimensions, 1))
        )
        assert self.proposed_model.shape == (self.dimensions, 1), dev_assertion_message

    def _evaluate_acceptance(self):

        # Compute new misfit
        self.proposed_x = self.distribution.misfit(self.proposed_model)

        acceptance_rate = _numpy.exp(self.current_x - self.proposed_x)

        if self.autotuning:
            self.autotune(acceptance_rate)

        # Evaluate MH acceptance rate
        if acceptance_rate > self.rng.uniform(0, 1):
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

    randomize_stepsize: bool = True
    """Boolean describing whether or not to randomize the stepsize slightly for every trajectory,
    by a Uniform~[0.5-1.5] * stepsize."""

    stepsizes: _numpy.ndarray = None
    """A NumPy ndarray containing all past stepsizes for the HMC algorithm. Collected if
    autotuning is True."""

    minimal_stepsize: float = 1e-18
    """Minimal stepsize which is chosen if stepsize becomes zero or negative during
    autotuning."""

    integrator = None

    def sample(
        self,
        samples_filename: str,
        distribution: _AbstractDistribution,
        stepsize: float = 0.1,
        randomize_stepsize: bool = True,
        amount_of_steps: int = 10,
        mass_matrix: _AbstractMassMatrix = None,
        integrator: str = "lf",
        initial_model: _numpy.ndarray = None,
        proposals: int = 100,
        online_thinning: int = 1,
        diagnostic_mode: bool = False,
        overwrite_existing_file: bool = False,
        max_time: float = None,
        autotuning: bool = False,
        target_acceptance_rate: float = 0.65,
        learning_rate: float = 0.75,
        queue=None,
        disable_progressbar=False,
    ):
        """Sampling using the Hamiltonian Monte Carlo algorithm.

        Parameters
        ----------
        samples_filename: str
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
        randomize_stepsize: bool
            A boolean enabling the randomization of the stepsize, which helps avoiding
            MCMC resonance.
        diagnostic_mode: bool
            A boolean describing if subroutines of sampling should be timed. Useful for
            finding slow parts of the algorithm. Will add overhead to each function.
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
                samples_filename=samples_filename,
                distribution=distribution,
                stepsize=stepsize,
                randomize_stepsize=randomize_stepsize,
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
                overwrite_existing_file=overwrite_existing_file,
                max_time=max_time,
                disable_progressbar=disable_progressbar,
            )

        except Exception as e:
            if self.samples is not None:
                self.samples.close()

            if type(e) is FileExistsError:
                if (
                    str(e) == "Skipping sampling due to samples file existing. Code "
                    "execution continues."
                ):
                    return
            raise e

        self._sample_loop()

        if self.parallel:
            queue.put({f"{self.sampler_index}": self._widget_data()})

        return self

    def _init_sampler_specific(self, **kwargs):
        # Parse all possible kwargs
        for key in (
            "stepsize",
            "randomize_stepsize",
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

        self.mass_matrix.rng = self.rng

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

            # Also write these to the hdf5 dataset
            self.samples.write_attribute("acceptance_rates", self.acceptance_rates)
            self.samples.write_attribute("stepsizes", self.stepsizes)

    def _write_tuning_settings(self):
        self.samples.write_attribute("stepsize", self.stepsize)
        self.samples.write_attribute("amount_of_steps", self.amount_of_steps)
        self.samples.write_attribute("mass_matrix", self.mass_matrix.name)
        self.samples.write_attribute(
            "integrator", self.integrators_full_names[self.integrator]
        )

    def _tuning_settings(self):
        settings = {
            "step size"
            if not self.autotuning
            else "final step size after tuning": str(self.stepsize),
            "amount of steps": self.amount_of_steps,
            "mass matrix type": self.mass_matrix.name
            if self.mass_matrix is not None
            else None,
            "integrator": self.integrators_full_names[self.integrator]
            if self.integrator is not None
            else None,
        }

        if self.autotuning:
            settings = {
                **settings,
                **{
                    "autotuning": self.autotuning,
                    "learning_rate": self.learning_rate,
                    "target_acceptance_rate": self.target_acceptance_rate,
                    "minimal_stepsize": self.minimal_stepsize,
                },
            }

        return settings

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

        if acceptance_rate > self.rng.uniform(0, 1):
            self.current_model = _numpy.copy(self.proposed_model)
            self.current_x = self.proposed_x
            self.accepted_proposals += 1
            self.mass_matrix.accept()
        else:
            self.mass_matrix.reject()

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
            if self.diagnostic_mode:
                _warnings.warn(
                    "The timestep of the algorithm went below zero. You possibly "
                    "started the algorithm in a region with extremely strong "
                    "gradients. The sampler will now default to a minimum timestep of "
                    f"{self.minimal_stepsize}. If this doesn't work, and if choosing "
                    "a different initial model does not make this warning go away, try"
                    "setting a smaller minimal time step and initial time step value."
                )
            self.stepsize = max(self.stepsize, self.minimal_stepsize)

    def _propagate_leapfrog(
        self,
    ):

        # Make sure not to alter a view but a copy of arrays ---------------------------
        position = self.current_model.copy()
        momentum = self.current_momentum.copy()
        dXdpos = None

        if self.randomize_stepsize:
            local_stepsize = self.rng.uniform(0.5, 1.5) * self.stepsize
        else:
            local_stepsize = self.stepsize

        # Leapfrog integration ---------------------------------------------------------
        position += (
            0.5
            * local_stepsize
            * self.mass_matrix.kinetic_energy_gradient(momentum, position, dXdpos)
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
            dXdpos = self.distribution.gradient(position)
            momentum -= local_stepsize * dXdpos
            # Position step
            position += local_stepsize * self.mass_matrix.kinetic_energy_gradient(
                momentum, position, dXdpos
            )
            # Correct bounds
            self.distribution.corrector(position, momentum)

        # Full momentum and half step position after loop ------------------------------
        # Momentum step
        dXdpos = self.distribution.gradient(position)
        momentum -= local_stepsize * dXdpos
        # Position step
        position += (
            0.5
            * local_stepsize
            * self.mass_matrix.kinetic_energy_gradient(momentum, position, dXdpos)
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

        if self.randomize_stepsize:
            local_stepsize = self.rng.uniform(0.5, 1.5) * self.stepsize
        else:
            local_stepsize = self.stepsize

        a1 *= local_stepsize
        a2 *= local_stepsize
        a3 *= local_stepsize
        b1 *= local_stepsize
        b2 *= local_stepsize

        # Make sure not to alter a view but a copy of the passed arrays.
        position = self.current_model.copy()
        momentum = self.current_momentum.copy()
        dXdpos = None

        # Leapfrog integration -------------------------------------------------
        for i in range(self.amount_of_steps):

            # A1
            position += a1 * self.mass_matrix.kinetic_energy_gradient(
                momentum, position, dXdpos
            )
            self.distribution.corrector(position, momentum)

            # B1
            dXdpos = self.distribution.gradient(position)
            momentum -= b1 * dXdpos

            # A2
            position += a2 * self.mass_matrix.kinetic_energy_gradient(
                momentum, position, dXdpos
            )
            self.distribution.corrector(position, momentum)

            # B2
            dXdpos = self.distribution.gradient(position)
            momentum -= b2 * dXdpos

            # A3
            position += a3 * self.mass_matrix.kinetic_energy_gradient(
                momentum, position, dXdpos
            )
            self.distribution.corrector(position, momentum)

            # B2
            dXdpos = self.distribution.gradient(position)
            momentum -= b2 * dXdpos

            # A2
            position += a2 * self.mass_matrix.kinetic_energy_gradient(
                momentum, position, dXdpos
            )
            self.distribution.corrector(position, momentum)

            # B1
            dXdpos = self.distribution.gradient(position)
            momentum -= b1 * dXdpos

            # A1
            position += a1 * self.mass_matrix.kinetic_energy_gradient(
                momentum, position, dXdpos
            )
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

        if self.randomize_stepsize:
            local_stepsize = self.rng.uniform(0.5, 1.5) * self.stepsize
        else:
            local_stepsize = self.stepsize

        a1 *= local_stepsize
        a2 *= local_stepsize
        b1 *= local_stepsize
        b2 *= local_stepsize

        # Make sure not to alter a view but a copy of the passed arrays.
        position = self.current_model.copy()
        momentum = self.current_momentum.copy()
        dXdpos = None

        # Leapfrog integration -------------------------------------------------
        for i in range(self.amount_of_steps):

            # A1
            position += a1 * self.mass_matrix.kinetic_energy_gradient(
                momentum, position, dXdpos
            )
            self.distribution.corrector(position, momentum)

            # B1
            dXdpos = self.distribution.gradient(position)
            momentum -= b1 * dXdpos

            # A2
            position += a2 * self.mass_matrix.kinetic_energy_gradient(
                momentum, position, dXdpos
            )
            self.distribution.corrector(position, momentum)

            # B2
            dXdpos = self.distribution.gradient(position)
            momentum -= b2 * dXdpos

            # A2
            position += a2 * self.mass_matrix.kinetic_energy_gradient(
                momentum, position, dXdpos
            )
            self.distribution.corrector(position, momentum)

            # B1
            dXdpos = self.distribution.gradient(position)
            momentum -= b1 * dXdpos

            # A1
            position += a1 * self.mass_matrix.kinetic_energy_gradient(
                momentum, position, dXdpos
            )
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


class PipeMatrix:
    def __init__(self, n_endpoints):
        self.n_endpoints = n_endpoints

        self.left_pipes = [
            [None for i in range(n_endpoints)] for i in range(n_endpoints)
        ]
        self.right_pipes = [
            [None for i in range(n_endpoints)] for i in range(n_endpoints)
        ]

        for point1 in range(self.n_endpoints):
            for point2 in range(point1):
                left, right = _Pipe(True)
                self.left_pipes[point1][point2] = left
                self.right_pipes[point1][point2] = right

            # The following pipe is to communicate with the main process. Left is used
            # to send from main, right to receive at fork. Is used to send interrupts.
            left, right = _Pipe(True)
            self.left_pipes[point1][point1] = left
            self.right_pipes[point1][point1] = right

    def pipes_to_subprocess(self):
        return [self.left_pipes[point][point] for point in range(self.n_endpoints)]

    def pipes_from_main(self):
        return [self.right_pipes[point][point] for point in range(self.n_endpoints)]

    def retrieve_pipes(self, point1, point2):
        assert point1 != point2
        left_point = max(point1, point2)
        right_point = min(point1, point2)

        return (
            self.left_pipes[left_point][right_point],
            self.right_pipes[left_point][right_point],
        )

    def close(self):
        for point1 in range(self.n_endpoints):
            for point2 in range(point1):
                self.left_pipes[point1][point2].close()
                self.right_pipes[point1][point2].close()


class MyProc(_Process):
    pass


class ParallelSampleSMP:
    # SMP stands for Shared Memory Parallelism, i.e. single machine parallel sampling
    # Initlization makes a deepcopy of the samplers, but NOT of the posteriors.
    def __init__(self, seed=None):
        if seed is None:
            self.rng = _numpy.random.default_rng()
        else:
            self.rng = _numpy.random.default_rng(seed)

    def sample(
        self,
        samplers: _AbstractSampler,
        filenames: str,
        posteriors: _AbstractDistribution,
        overwrite_existing_files=False,
        proposals: int = 100,
        exchange: bool = True,
        exchange_interval: int = 1,
        initial_model: _Union[_numpy.array, _List[_numpy.array]] = None,
        kwargs: _Union[_Dict, _List[_Dict]] = None,
    ):

        assert overwrite_existing_files, (
            "You have to manually enable overwriting samples. This is for safety. The "
            "existing file dialog doesn't work in the parallel case. "
            "Set `overwrite_existing_files=True`."
        )

        number_of_chains = len(samplers)

        # Save passed parameters and check their shapes. -------------------------------
        self.samplers = _copy.deepcopy(samplers)

        assert len(filenames) == number_of_chains, (
            f"The number of supplied initial models ({len(filenames)}) "
            f"is not equal to the amount of chains ({number_of_chains}). "
            f"Supply {number_of_chains} models."
        )

        assert len(posteriors) == number_of_chains, (
            f"The number of supplied initial models ({len(posteriors)}) "
            f"is not equal to the amount of chains ({number_of_chains}). "
            f"Supply {number_of_chains} posteriors."
        )

        if type(initial_model) == list:
            assert len(initial_model) == number_of_chains, (
                f"The number of supplied initial models ({len(initial_model)}) "
                f"is not equal to the amount of chains ({number_of_chains}). "
                f"Supply either 1 or {number_of_chains} models."
            )

        if type(kwargs) == list:
            assert len(kwargs) == number_of_chains, (
                f"The number of supplied kwargs dictionaries ({len(kwargs)}) "
                f"is not equal to the amount of chains ({number_of_chains}). "
                f"Supply either 1 or {number_of_chains} kwargs. Note that even if you "
                f"don't want to pass arguments to one of the chains, and empty "
                f"dictionary still has to be included."
            )

        # Exchange and parallel communication details ----------------------------------
        # All markov chain processes in an array
        ps = []
        # If we need to exchange samples, create a schedule and a communication matrix.
        if exchange:
            # Every separate chain can at most switch 1 time per proposal. The total is
            # rounded down, because every chain needs a partner chain.
            exchanges_per_proposal = int(_numpy.floor(number_of_chains / 2))
            exchange_schedule = _numpy.vstack(
                [
                    self.rng.choice(
                        number_of_chains, exchanges_per_proposal * 2, replace=False
                    )
                    for i in range(int(proposals / exchange_interval))
                ]
            )
            # Communication object
            pipe_matrix = PipeMatrix(number_of_chains)
        else:
            exchange_schedule = None
            pipe_matrix = None
            exchange_interval = None
        self.exchange_schedule = exchange_schedule

        # A queue to which the final sampling details are passed at the end of sampling
        self.queue = _Queue(maxsize=number_of_chains)
        main_thread_keyboard_interrupt = _Value("f", 0)

        # Settings within the samplers -------------------------------------------------
        # Do modifications to the samplers that we don't want to do in the
        # constructor or sampling function. The reasoning here is that it is better
        # to avoid obfuscating the non-parallel code.
        for i, sampler in enumerate(self.samplers):
            sampler.parallel = True
            sampler.exchange_interval = exchange_interval
            sampler.sampler_index = i
            sampler.exchange_schedule = exchange_schedule
            sampler.pipe_matrix = pipe_matrix
            sampler.main_thread_keyboard_interrupt = main_thread_keyboard_interrupt

        # Start parallel sampling
        try:

            # These arguments need to be passed to all chains
            fixed_kwargs = {
                "overwrite_existing_file": True,
                "proposals": proposals,
                "queue": self.queue,
            }

            for i_chain, (sampler, filename, posterior) in enumerate(
                zip(self.samplers, filenames, posteriors)
            ):

                if type(initial_model) == list:
                    chain_initial_model = initial_model[i_chain]
                else:
                    chain_initial_model = initial_model

                if type(kwargs) == list:
                    chain_kwargs = kwargs[i_chain]
                elif kwargs is None:
                    chain_kwargs = {}
                else:
                    chain_kwargs = kwargs

                total_kwargs = {
                    **{"initial_model": chain_initial_model},
                    **chain_kwargs,
                    **fixed_kwargs,
                }

                ps.append(
                    MyProc(
                        target=sampler.sample,
                        args=(filename, posterior),
                        kwargs=total_kwargs,
                    )
                )

            print(f"Starting {number_of_chains} markov chains...")
            for p in ps:
                p.start()

            for p in ps:
                p.join()

        except KeyboardInterrupt:
            # Keyboard interrupt is also send automatically to subprocesses, so the only
            # thing left to do is wait on them.
            for p in ps:
                p.join()
        except Exception as e:
            if exchange:
                pipe_matrix.close()
            raise e
        finally:
            self.sampler_widget_data = []
            while not self.queue.empty():
                self.sampler_widget_data.append(self.queue.get())
            # Beat it into the correct format
            data = {}
            for idata in self.sampler_widget_data:
                data = {**data, **idata}
            self.sampler_widget_data = data

        if exchange:
            pipe_matrix.close()

        return self

    def _repr_html_(self):

        tab = _widgets.Tab()
        if self.sampler_widget_data:
            tab.children = [
                sampler._repr_html_(
                    nested=True, widget_data=self.sampler_widget_data[f"{i}"]
                )
                for i, sampler in enumerate(self.samplers)
            ]

        for i in range(len(tab.children)):
            tab.set_title(i, f"Sampler {i}")

        _display(tab)

        return ""

    def print_results(self):
        print(self._repr_html_())


class _AbstractVisualSampler(_AbstractSampler):
    """This class acts as a superclass to all possible visual variations on the algorithms.

    We directly 'hijack' 3 methods; the constructor, _init_sampler and _close_sampler, to
    create and close the plots. The beauty is that this can still call the original methods
    from _AbstractSampler. We also need to define some general function (i.e. that are used
    for all visual samplers) that override subclass methods. Since this is not possible
    from the superclass, we provide that function (update plots) itself to be linked in the
    subclass.
    """

    misfits_to_plot: _numpy.array
    """Array of stored misfits that are plotted"""
    samples_to_plot: _numpy.array
    """Array of stored samples that are plotted"""
    plot_update_interval: int = 100
    """Update interval (in number of samples) of the plots. Might strongly influence
    algorithm performance."""
    animate_proposals: bool = False
    """Whether to animate the proposals themselves. Animation is controlled by subclass."""
    leave_proposal_animation: bool = False
    """Whether to leave the animation of the proposals."""
    animation_domain = None
    """Array describing  the extents of the animation domain for the samples, in
    [xmin, xmax, ymin, ymax]. If not supplied, the domain is dynamically extended."""
    dims_to_plot = [0, 1]
    """Which dimensions to animate samples for."""

    background_image = None
    """Array storing a background image to plot behind the samples."""

    def __init__(
        self,
        plot_update_interval=None,
        dims_to_plot=None,
        animate_proposals=None,
        leave_proposal_animation=None,
        animation_domain=None,
        background=None,
        seed=None,
    ):

        # Parse parameters
        if plot_update_interval is not None:
            self.plot_update_interval = plot_update_interval

        if dims_to_plot is not None:
            self.dims_to_plot = dims_to_plot

        if animate_proposals is not None:
            self.animate_proposals = animate_proposals

        # If we are animating proposals, we take a neglegible performance hit to draw,
        # every sample, hence we set the interval to 1.
        if self.animate_proposals:
            self.plot_update_interval = 1

        if leave_proposal_animation is not None:
            self.leave_proposal_animation = leave_proposal_animation

        if animation_domain is not None:
            self.animation_domain = animation_domain

        if background is not None:
            self.x1s, self.x2s, self.background_image = background

        # Call the original constructor
        super().__init__(seed=seed)

    def _init_sampler(
        self,
        samples_filename: str,
        distribution: _AbstractDistribution,
        initial_model: _numpy.ndarray,
        proposals: int,
        online_thinning: int,
        overwrite_existing_file: bool,
        max_time: int,
        disable_progressbar: bool = False,
        diagnostic_mode: bool = False,
        **kwargs,
    ):

        dimensions = distribution.dimensions
        for dim in self.dims_to_plot:
            assert (
                dim < dimensions
            ), "You requested to animate a dimension which is not in the distribution"
        assert self.plot_update_interval > 0
        if self.animation_domain is not None:
            assert self.animation_domain[1] > self.animation_domain[0]
            assert self.animation_domain[3] > self.animation_domain[2]

        # Create arrays to store animation parameters
        self.misfits_to_plot = _numpy.empty((proposals, 1))
        self.samples_to_plot = _numpy.empty((proposals, 2))

        # Create collection of plots
        self.plots = {}
        self.plots["global_misfit"] = {}
        self.plots["samples"] = {}

        # Create figure
        self.plots["figure"] = _plt.figure(figsize=(10, 5))
        _plt.show(block=False)

        # Subplot 1; misfits over time
        ar = 2
        if self.animation_domain is not None:
            ar = 1 + (self.animation_domain[1] - self.animation_domain[0]) / (
                self.animation_domain[3] - self.animation_domain[2]
            )

        a0, a1 = self.plots["figure"].subplots(
            1, 2, gridspec_kw={"width_ratios": [1, ar]}
        )

        self.plots["global_misfit"]["axis"] = a0  # _plt.subplot(121)
        self.plots["global_misfit"]["title"] = _plt.title("Misfit over time")
        self.plots["global_misfit"]["axis"].set_xlim([0, proposals])
        self.plots["global_misfit"]["axis"].set_xlabel("sample index")
        self.plots["global_misfit"]["axis"].set_ylabel("Unnormalized -log(p)")
        self.plots["global_misfit"]["scatterplot"] = None

        # Subplot 2; samples over time
        self.plots["samples"]["axis"] = a1  # _plt.subplot(122)
        self.plots["samples"]["axis"].set_aspect(1)
        self.plots["samples"]["title"] = _plt.title("2d samples over time")
        self.plots["samples"]["axis"].set_xlabel(
            f"Model dimension {self.dims_to_plot[0]}"
        )
        self.plots["samples"]["axis"].set_ylabel(
            f"Model dimension {self.dims_to_plot[1]}"
        )
        self.plots["samples"]["scatterplot"] = None
        if self.background_image is not None:

            self.plots["samples"]["axis"].contour(
                self.x1s,
                self.x2s,
                _numpy.exp(-self.background_image),
                levels=20,
                alpha=0.5,
                zorder=0,
            )
        if self.animation_domain is not None:
            self.plots["samples"]["axis"].set_xlim(
                [self.animation_domain[0], self.animation_domain[1]]
            )
            self.plots["samples"]["axis"].set_ylim(
                [self.animation_domain[2], self.animation_domain[3]]
            )

        self.plots["samples"]["legend"] = None

        # Run original function
        return super()._init_sampler(
            samples_filename,
            distribution,
            initial_model,
            proposals,
            online_thinning,
            overwrite_existing_file,
            max_time,
            disable_progressbar=disable_progressbar,
            diagnostic_mode=diagnostic_mode,
            **kwargs,
        )

    def _update_plots_after_acceptance(self, force=False):

        if len(_plt.get_fignums()) == 0:
            # If the figure is closed by the user, we skip the plotting
            return

        # Load current state
        self.misfits_to_plot[self.current_proposal] = self.current_x
        self.samples_to_plot[self.current_proposal, :] = self.current_model[
            self.dims_to_plot
        ].flatten()

        # Beat the data into a nice shape
        index, misfit = (
            _numpy.arange(self.misfits_to_plot[: self.current_proposal + 1].size)[
                :, None
            ],
            self.misfits_to_plot[: self.current_proposal + 1],
        )
        samples_x, samples_y = (
            self.samples_to_plot[: self.current_proposal + 1, 0, None],
            self.samples_to_plot[: self.current_proposal + 1, 1, None],
        )

        if self.plots["global_misfit"]["scatterplot"] is None:
            # Create plots if they are not there yet ...
            self.plots["global_misfit"]["scatterplot"] = self.plots["global_misfit"][
                "axis"
            ].scatter(index, misfit, s=10, c="r")

            self.plots["samples"]["scatterplot"] = self.plots["samples"][
                "axis"
            ].scatter(samples_x, samples_y, s=30, label="samples")

            if self.plots["samples"]["legend"] is None:
                self.plots["samples"]["legend"] = self.plots["samples"]["axis"].legend()

        else:
            # ... or update them
            if self.current_proposal % self.plot_update_interval == 0 or force:
                self.plots["global_misfit"]["scatterplot"].set_offsets(
                    _numpy.hstack((index, misfit))
                )

                ymin = misfit.min() - 0.1 * (misfit.max() - misfit.min())
                ymax = misfit.max() + 0.1 * (misfit.max() - misfit.min())

                if ymin == ymax:
                    ymin -= 0.5
                    ymax += 0.5

                self.plots["global_misfit"]["axis"].set_ylim(
                    [
                        ymin,
                        ymax,
                    ]
                )

                self.plots["samples"]["scatterplot"].set_offsets(
                    _numpy.hstack((samples_x, samples_y))
                )

                if self.animation_domain is None:
                    # Update the bounds as fit if they were not given by the user

                    xmin, xmax = samples_x.min(), samples_x.max()
                    ymin, ymax = samples_y.min(), samples_y.max()

                    if xmin == xmax:
                        xmin -= 0.5
                        xmax += 0.5
                    if ymin == ymax:
                        ymin -= 0.5
                        ymax += 0.5

                    self.plots["samples"]["axis"].set_xlim([xmin, xmax])
                    self.plots["samples"]["axis"].set_ylim([ymin, ymax])
                self.plots["figure"].canvas.draw()
                _plt.pause(0.00001)

    def _close_sampler(self):

        # Set the maximum sample on the misfit plot to the latest sample
        self._update_plots_after_acceptance(force=True)
        self.plots["global_misfit"]["axis"].set_xlim([0, self.current_proposal + 1])

        self.plots["figure"].canvas.draw()
        _plt.pause(0.00001)
        _plt.close()
        return super()._close_sampler()


class RWMH_visual(_AbstractVisualSampler, RWMH):
    """Visual version of Random Walk Metropolis Hastings"""

    def _evaluate_acceptance(self):
        """Animate every new sample after criterion evaluation"""
        return_value = super()._evaluate_acceptance()
        self._update_plots_after_acceptance()
        return return_value


class HMC_visual(_AbstractVisualSampler, HMC):
    """Visual version of Hamiltonian Monte Carlo"""

    def _evaluate_acceptance(self):
        """Animate every new sample after criterion evaluation"""
        return_value = super()._evaluate_acceptance()
        self._update_plots_after_acceptance()
        return return_value

    def _propagate_leapfrog_visual(self):
        """Animate the leapfrog integration."""

        if not self.animate_proposals or len(_plt.get_fignums()) == 0:
            # If the figure is closed by the user, we skip the plotting
            return super()._propagate_leapfrog()

        position = self.current_model.copy()
        momentum = self.current_momentum.copy()
        dXdpos = None

        # These are the positions stored for animating the trajectory
        positions_x = _numpy.array([])
        positions_y = _numpy.array([])
        positions_x = _numpy.append(positions_x, position[self.dims_to_plot[0]])
        positions_y = _numpy.append(positions_y, position[self.dims_to_plot[1]])

        self.plots["samples"]["scatterplot_proposal"] = self.plots["samples"][
            "axis"
        ].plot(positions_x, positions_y, "r", alpha=0.5, label="trajectories", zorder=0)
        line = self.plots["samples"]["scatterplot_proposal"].pop(0)

        self.plots["figure"].canvas.draw()
        _plt.pause(0.00001)

        if self.randomize_stepsize:
            local_stepsize = self.rng.uniform(0.5, 1.5) * self.stepsize
        else:
            local_stepsize = self.stepsize

        # Leapfrog integration ---------------------------------------------------------
        position += (
            0.5
            * local_stepsize
            * self.mass_matrix.kinetic_energy_gradient(momentum, position, dXdpos)
        )

        line.set_xdata(
            _numpy.append(line.get_xdata().flatten(), position[self.dims_to_plot[0]])
        )
        line.set_ydata(
            _numpy.append(line.get_ydata().flatten(), position[self.dims_to_plot[1]])
        )

        positions_x = _numpy.array([])
        positions_y = _numpy.array([])
        positions_x = _numpy.append(positions_x, position[self.dims_to_plot[0]])
        positions_y = _numpy.append(positions_y, position[self.dims_to_plot[1]])
        self.plots["samples"]["scatterplot_proposal_grads"] = self.plots["samples"][
            "axis"
        ].plot(
            positions_x,
            positions_y,
            "r",
            marker=".",
            ls="",
            alpha=1,
            markersize=3,
            label="computed gradients",
            zorder=0,
        )
        line_grads = self.plots["samples"]["scatterplot_proposal_grads"].pop(0)
        line_grads.set_xdata(
            _numpy.append(
                line_grads.get_xdata().flatten(), position[self.dims_to_plot[0]]
            )
        )
        line_grads.set_ydata(
            _numpy.append(
                line_grads.get_ydata().flatten(), position[self.dims_to_plot[1]]
            )
        )

        self.plots["figure"].canvas.draw()
        _plt.pause(0.00001)

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
            dXdpos = self.distribution.gradient(position)
            momentum -= local_stepsize * dXdpos
            # Position step
            position += local_stepsize * self.mass_matrix.kinetic_energy_gradient(
                momentum, position, dXdpos
            )

            # Correct bounds
            self.distribution.corrector(position, momentum)

            line.set_xdata(
                _numpy.append(
                    line.get_xdata().flatten(), position[self.dims_to_plot[0]]
                )
            )
            line.set_ydata(
                _numpy.append(
                    line.get_ydata().flatten(), position[self.dims_to_plot[1]]
                )
            )
            line_grads.set_xdata(
                _numpy.append(
                    line_grads.get_xdata().flatten(), position[self.dims_to_plot[0]]
                )
            )
            line_grads.set_ydata(
                _numpy.append(
                    line_grads.get_ydata().flatten(), position[self.dims_to_plot[1]]
                )
            )
            self.plots["figure"].canvas.draw()
            _plt.pause(0.00001)

        # Full momentum and half step position after loop ------------------------------
        # Momentum step
        dXdpos = self.distribution.gradient(position)
        momentum -= local_stepsize * dXdpos
        # Position step
        position += (
            0.5
            * local_stepsize
            * self.mass_matrix.kinetic_energy_gradient(momentum, position, dXdpos)
        )
        self.distribution.corrector(position, momentum)

        line.set_xdata(
            _numpy.append(line.get_xdata().flatten(), position[self.dims_to_plot[0]])
        )
        line.set_ydata(
            _numpy.append(line.get_ydata().flatten(), position[self.dims_to_plot[1]])
        )
        self.plots["figure"].canvas.draw()
        _plt.pause(0.00001)

        self.proposed_model = position.copy()
        self.proposed_momentum = momentum.copy()

        if not self.leave_proposal_animation:
            line.remove()
            line_grads.remove()

    integrators = {
        "lf": _propagate_leapfrog_visual,
    }
    integrators_full_names = {
        "lf": "leapfrog integrator",
    }
    available_integrators = integrators.keys()
