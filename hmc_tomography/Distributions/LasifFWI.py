# Common imports
import pathlib as _pathlib
from shutil import copyfile as _copyfile
from time import sleep as _sleep
from time import time as _time
from typing import Dict as _Dict, Union as _Union
import numpy as _numpy

import h5py as _h5py
import toml as _toml

import lasif as _lasif
import lasif.salvus_utils as _salvus_utils
from hmc_tomography.Distributions import _AbstractDistribution
from salvus.flow.sites.salvus_job import SalvusJob
from salvus.flow.sites.salvus_job_array import SalvusJobArray


class LasifFWI(_AbstractDistribution):

    lasif_root: _pathlib.Path = None
    """Path to the LASIF project root folder."""

    communicator: _lasif.components.communicator.Communicator = None
    """LASIF communicator object from LASIF project."""

    config_dict: _Dict = None
    """LASIF configuration dictionary."""

    _last_computed_misfit: _numpy.ndarray = None
    """A float that is set when a misfit is computed, but reset once a different model
    is loaded. By keeping track of this variable, we can avoid some repeated
    computations."""

    _last_computed_gradient: _numpy.ndarray = None
    """A numpy.ndarray of floats that is set when a gradient is computed, but reset once
    a different model is loaded. By keeping track of this variable, we can avoid some
    repeated computations."""

    running_job: _Union[SalvusJob, SalvusJobArray, None] = None
    """Attribute containing the job (array) object of the currently running simulations,
    or None if no simulation is running currently."""

    salvus_verbosity: int = 1
    """Verbosity level of Salvus. Use 1 for output, use 0 for quiet."""

    verbosity: int = 0
    """Verbosity level of an instance. Use 1 for output, use 0 for quiet."""

    parametrization = ["RHO", "VP", "VS"]
    """List of strings representing the parametrization and parameter order in model
    vector."""

    mesh_to_model_map = [None] * len(parametrization)
    """A list of integers representing the column of the mesh data array which contains
    the parameters corresponding to the parametrization array parameter at that index.
    """

    mesh_parametrization_fields = [None]
    """A list of strings describing the contents of the columns in the mesh data array:
    mesh["MODEL"]["data"].attrs.get("DIMENSION_LABELS")."""

    def __init__(self, path_to_project: str, verbosity: int = 0):

        # Set verbosity
        self.verbosity = verbosity

        # Set path
        self.lasif_root = _pathlib.Path(path_to_project)

        # Set communicator
        self.communicator = _lasif.api.find_project_comm(path_to_project)

        # Load configuration
        config_file_path = self.lasif_root / "lasif_config.toml"
        self.config_dict = _toml.load(config_file_path)

        # Moving meshes around =========================================================
        # In this section a duplicate of the original mesh file is created in which we
        # alter parameters. This way, we don't have to worry about losing our original
        # mesh. We update the location of this mesh in the communicator instance, but
        # don't dump the updated dictionary to the original TOML file, such that we
        # don't alter the LASIF project and don't need to do any clean up.

        # Get the original mesh file path + name
        self.original_mesh_file = self.config_dict["lasif_project"]["domain_settings"][
            "domain_file"
        ]

        # Set the working mesh file path + name
        self.working_mesh_file = (
            self.original_mesh_file.replace(".h5", "") + "_working_file.h5"
        )

        # Create the working mesh file from the copy path + name
        _copyfile(
            self.config_dict["lasif_project"]["domain_settings"]["domain_file"],
            self.working_mesh_file,
        )

        # Change the LASIF mesh file to the working file, but don't dump this back to
        # disk: when we reload the LASIF project it will have it's original location.
        self.communicator.project.lasif_config["domain_settings"][
            "domain_file"
        ] = self.working_mesh_file

        # Actually opening the mesh ====================================================

        # Open the mesh
        self.mesh = _h5py.File(self.working_mesh_file, "r+",)

        # Parsing the parametrization ==================================================
        self.mesh["MODEL"]["data"].attrs.get("DIMENSION_LABELS")

        # Get model parametrization as a list of strings
        self.mesh_parametrization_fields = [
            string.replace("b'[ ", "").replace(" ]'", "")
            for string in str(
                self.mesh["MODEL"]["data"].attrs.get("DIMENSION_LABELS")[1]
            ).split(" | ")
        ]

        for i_parameter, parameter in enumerate(self.parametrization):
            assert parameter in self.mesh_parametrization_fields

            self.mesh_to_model_map[
                i_parameter
            ] = self.mesh_parametrization_fields.index(parameter)

        # Calculate amount of free parameters
        self.dimensions = (
            self.mesh["MODEL"]["data"].shape[0]
            * len(self.parametrization)
            * self.mesh["MODEL"]["data"].shape[2]
        )

    @property
    def current_model(self):
        """The attribute setter for the current model.

        Gets the model directly from the mesh file."""

        return self.mesh["MODEL"]["data"][:, self.mesh_to_model_map, :].reshape(
            self.dimensions, 1
        )

    @current_model.setter
    def current_model(self, current_model):
        """The attribute setter for the current model.

        It also invalidates the current misfit and gradient, such that it requires new
        simulations."""

        assert current_model.shape == (self.dimensions, 1)
        if _numpy.all(
            current_model
            == self.mesh["MODEL"]["data"][:, self.mesh_to_model_map, :].flatten(
                order="C"
            )[:, None]
        ):
            # If the model we're trying to set is already the same as the one in the
            # mesh, we don't need to do anything.
            return
        else:
            # If the model we're trying to set is NOT the same as the one in the mesh,
            # we update the mesh and invalidate simulations.
            self.mesh["MODEL"]["data"][
                :, self.mesh_to_model_map, :
            ] = current_model.reshape(
                self.mesh["MODEL"]["data"].shape[0],
                len(self.parametrization),
                # One column less for z-index, hence the minus 1. the other values of
                # the second dimensions (shape[1]) are VP, VS, RHO (typically). We still
                # need some checks on that.
                self.mesh["MODEL"]["data"].shape[2],
            )

            self._last_computed_misfit = None
            self._last_computed_gradient = None

            return

    def misfit(self, model):

        self.current_model = model

        # The misfit is still stored if we didn't change our model from the last
        # computation, so we can return it without re-running any simulation
        if self._last_computed_misfit is not None:

            print("Model is the same, not re-starting simulation.")

            return self._last_computed_misfit

        # Set up new iteration
        self.current_iteration = f"{_time()}"

        _lasif.api.set_up_iteration(self.lasif_root, iteration=self.current_iteration)

        # Get all events to simulate
        events = _lasif.api.list_events(
            self.lasif_root,
            just_list=True,
            iteration=self.current_iteration,
            output=True,
        )

        # Create simulation objects
        simulations = []
        for event in events:
            simulation = _salvus_utils.create_salvus_forward_simulation(
                comm=self.communicator,
                event=event,
                iteration=self.current_iteration,
                side_set="r1",
            )
            simulations.append(simulation)

        # Submit the simulations
        self.running_job = _salvus_utils.submit_salvus_simulation(
            comm=self.communicator,
            simulations=simulations,
            events=events,
            iteration=self.current_iteration,
            sim_type="forward",
            verbosity=self.salvus_verbosity,
        )

        # Retrieve outputs
        try:
            _salvus_utils.retrieve_salvus_simulations_blocking(
                comm=self.communicator,
                events=events,
                iteration=self.current_iteration,
                sim_type="forward",
                verbosity=self.verbosity,
            )
        except Exception as e:
            # If anything goes bad, at least close the HDF5 file
            self.mesh.close()

            if self.running_job is not None:
                self.running_job.cancel()
                self.running_job = None

            if self.verbosity > 0:
                print("Keyboard interrupt during forward run, closing mesh file.")
            raise e

        self.running_job = None

        _lasif.api.calculate_adjoint_sources(
            self.lasif_root, iteration=self.current_iteration, window_set="B",
        )

        self._last_computed_misfit = _lasif.api.write_misfit(
            lasif_root=self.lasif_root, iteration=self.current_iteration
        )

        return self._last_computed_misfit

    def gradient(
        self, model, iteration=None, override_misfit=False, multiply_mass=False
    ):

        self.current_model = model

        # if self._last_computed_misfit is not None and not override_misfit:
        #     print("Model is the same, not re-starting forward simulation.")
        # else:
        #     self.misfit(model)

        if self._last_computed_gradient is not None:
            print("Model is the same, not re-starting gradient simulation.")
            return self._last_computed_gradient

        if iteration is None:
            iteration = self.current_iteration

        # Create adjoint simulations ===================================================
        events = _lasif.api.list_events(
            self.lasif_root, just_list=True, iteration=iteration, output=True
        )
        simulations = []
        for event in events:
            simulation = _salvus_utils.create_salvus_adjoint_simulation(
                comm=self.communicator, event=event, iteration=iteration,
            )
            simulations.append(simulation)

        # Submit adjoint simulation ====================================================
        self.running_job = _lasif.salvus_utils.submit_salvus_simulation(
            comm=self.communicator,
            simulations=simulations,
            events=events,
            iteration=iteration,
            sim_type="adjoint",
        )

        # Retrieve =====================================================================
        events = _lasif.api.list_events(
            self.lasif_root, just_list=True, iteration=iteration, output=True
        )

        try:
            _lasif.salvus_utils.retrieve_salvus_simulations_blocking(
                comm=self.communicator,
                events=events,
                iteration=iteration,
                sim_type="adjoint",
                verbosity=self.verbosity,
            )
        except Exception as e:
            # If anything goes bad, at least close the HDF5 file
            self.mesh.close()

            if self.running_job is not None:
                self.running_job.cancel()
                self.running_job = None

            if self.verbosity > 0:
                print("Keyboard interrupt during adjoint run, closing mesh file.")
            raise e

        self.running_job = None

        return self.construct_gradient(multiply_mass=multiply_mass)

    def generate(self):
        pass

    def __del__(self):
        self.mesh.close()

        if self.running_job is not None:
            self.running_job.cancel()

        if self.verbosity > 0:
            print("Destructor called, closing mesh file.")

    @staticmethod
    def create_default() -> "LasifFWI":
        raise NotImplementedError()

    def construct_gradient(self, iteration=None, multiply_mass=False) -> _numpy.ndarray:
        if iteration is None:
            iteration = self.current_iteration

        events = _lasif.api.list_events(
            self.lasif_root, just_list=True, iteration=iteration, output=True
        )

        gradient = _numpy.zeros((self.dimensions, 1))

        # Loop over events
        for event in events:
            gradient_path = (
                self.lasif_root
                / "GRADIENTS"
                / f"ITERATION_{iteration}"
                / event
                / "gradient.h5"
            )

            # Open the gradient for the event using a context manager
            with _h5py.File(gradient_path, "r") as gradient_file_handle:
                gradient_file_handle

                # Get the names of the columns in the mesh data array
                gradient_parametrization_fields = [
                    string.replace("b'[ ", "").replace(" ]'", "")
                    for string in str(
                        gradient_file_handle["MODEL"]["data"].attrs.get(
                            "DIMENSION_LABELS"
                        )[1]
                    ).split(" | ")
                ]

                # Create empty map from gradient to model vector
                gradient_to_model_map = [None] * len(self.parametrization)

                mass_matrix_index = gradient_parametrization_fields.index(
                    "FemMassMatrix"
                )

                # Loop over all parameters in the parametrization
                for i_parameter, parameter in enumerate(self.parametrization):

                    # Assert that the parameter is in the gradient
                    assert parameter in gradient_parametrization_fields

                    # Construct a map from gradient to mesh
                    gradient_to_model_map[
                        i_parameter
                    ] = gradient_parametrization_fields.index(parameter)

                if multiply_mass:
                    gradient += (
                        gradient_file_handle["MODEL"]["data"][:, mass_matrix_index, :][
                            :, None, :
                        ]
                        * gradient_file_handle["MODEL"]["data"][()][
                            :, gradient_to_model_map, :
                        ]
                    ).reshape(self.dimensions, 1)
                else:
                    gradient += gradient_file_handle["MODEL"]["data"][()][
                        :, gradient_to_model_map, :
                    ].reshape(self.dimensions, 1)

        return gradient
