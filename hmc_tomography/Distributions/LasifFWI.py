# Common imports
from ctypes import ArgumentError
import pathlib as _pathlib
from shutil import copyfile as _copyfile
import os as _os
from time import time as _time
from typing import Dict as _Dict, Union as _Union, List as _List
import numpy as _numpy
import matplotlib.pyplot as _plt

import h5py as _h5py
from numpy.lib.polynomial import poly
from scipy import interpolate
import toml as _toml

import lasif as _lasif
import lasif.salvus_utils as _salvus_utils
from hmc_tomography.Distributions import _AbstractDistribution
from salvus.flow.sites.salvus_job import SalvusJob
from salvus.flow.sites.salvus_job_array import SalvusJobArray

from hmc_tomography.Helpers.CaptureStdout import stdout_redirector as _stdout_redirector

from contextlib import redirect_stderr as _redirect_stderr
from contextlib import redirect_stdout as _redirect_stdout

import bspline as _bspline
import bspline.splinelab as _splinelab
import scipy as _scipy


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

    salvus_verbosity: int = 0
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

    use_splines: bool = False

    # current_iteration = None

    def __init__(self, path_to_project: str, verbosity: int = 0, spline=None):

        # Set verbosity
        self.verbosity = verbosity

        # Set path
        self.lasif_root = _pathlib.Path(path_to_project)

        # Set communicator
        self.f1 = open("lasif-salvus.stdout.log", "a", encoding="utf-8")
        self.f2 = open("lasif-salvus.stderr.log", "a", encoding="utf-8")
        with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
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

        if _os.path.exists(self.working_mesh_file):
            _os.remove(self.working_mesh_file)

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
        self.mesh_dimensions = self.dimensions

        self.current_iteration = f"{_time()}"

        with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
            _lasif.api.set_up_iteration(
                self.lasif_root, iteration=self.current_iteration
            )

        self.forward_job = None
        self.adjoint_job = None

        if spline is not None:

            self.dimensions = 0
            self.use_splines = True

            self.spline_basis: _Dict[_SplineBasis] = {}
            for parameter in self.parametrization:
                self.spline_basis[parameter] = _SplineBasis(
                    self.mesh,
                    interfaces=spline["interfaces"],
                    dof=spline["dof_per_parameter"],
                    knot_locations=spline["knot_locations"],
                    polynomial_order=spline["polynomial_order"],
                    background_model=spline["background_interpolators"][parameter],
                    collocation_interfaces=spline["collocation_interfaces"],
                )
                self.dimensions += self.spline_basis[parameter].dimensions

    @property
    def current_model(self):
        """The attribute setter for the current model.

        Gets the model directly from the mesh file."""

        if self.use_splines:

            model_per_parameter = [None] * len(self.parametrization)
            for i, parameter in enumerate(self.parametrization):
                model_per_parameter[i] = (
                    self.spline_basis[parameter].matrix_premultiplier
                    @ self.spline_basis[parameter].collocation_matrix.T
                    @ (
                        self.mesh["MODEL"]["data"][
                            :, self.mesh_to_model_map[i], :
                        ].flatten()[:, None]
                        - self.spline_basis[parameter].collocation_vector
                    )
                )
            return _numpy.vstack(model_per_parameter)
        else:
            return self.mesh["MODEL"]["data"][:, self.mesh_to_model_map, :].reshape(
                self.dimensions, 1
            )

    @current_model.setter
    def current_model(self, current_model):
        """The attribute setter for the current model.

        It also invalidates the current misfit and gradient, such that it requires new
        simulations."""

        assert current_model.shape == (self.dimensions, 1)

        if self.use_splines:
            # Split the model vector per parameter into a dictionary
            model_per_group = _numpy.vsplit(current_model, len(self.parametrization))
            model_per_group = {
                self.parametrization[i]: model_per_group[i]
                for i in range(len(self.parametrization))
            }
            print("Updating model")
            for i, parameter in enumerate(self.parametrization):
                self.mesh["MODEL"]["data"][:, self.mesh_to_model_map[i], :] = (
                    self.spline_basis[parameter].collocation_matrix
                    @ model_per_group[parameter]
                    + self.spline_basis[parameter].collocation_vector
                ).reshape(
                    self.mesh["MODEL"]["data"].shape[0],
                    self.mesh["MODEL"]["data"].shape[2],
                    order="C",
                )
        else:
            if _numpy.all(
                current_model
                == self.mesh["MODEL"]["data"][:, self.mesh_to_model_map, :].flatten(
                    order="C"
                )[:, None]
            ):
                # If the model we're trying to set is already the same as the one in the
                # mesh, we don't need to do anything.
                pass
            else:
                # If the model we're trying to set is NOT the same as the one in the
                # mesh, we update the mesh and invalidate simulations.
                self.mesh["MODEL"]["data"][
                    :, self.mesh_to_model_map, :
                ] = current_model.reshape(
                    self.mesh["MODEL"]["data"].shape[0],
                    len(self.parametrization),
                    # One column less for z-index, hence the minus 1. the other values
                    # of the second dimensions (shape[1]) are VP, VS, RHO (typically).
                    # We still need some checks on that.
                    self.mesh["MODEL"]["data"].shape[2],
                )

        # Set up new iteration
        if self.current_iteration is not None:
            with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
                _lasif.api.set_up_iteration(
                    self.lasif_root, iteration=self.current_iteration, remove_dirs=True,
                )

        self.current_iteration = f"{_time()}"

        with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
            _lasif.api.set_up_iteration(
                self.lasif_root, iteration=self.current_iteration
            )

        self._last_computed_misfit = None
        self._last_computed_gradient = None

        return

    def misfit(self, model):

        self.current_model = model

        # The misfit is still stored if we didn't change our model from the last
        # computation, so we can return it without re-running any simulation
        if self._last_computed_misfit is not None:
            return self._last_computed_misfit

        # Remove previous jobs' files
        if self.forward_job is not None:

            with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
                self.forward_job.delete()
            self.forward_job = None

        print("running forward")

        # Get all events to simulate

        with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
            events = _lasif.api.list_events(
                self.lasif_root,
                just_list=True,
                iteration=self.current_iteration,
                output=True,
            )

        forward_simulations = []

        # Create simulation objects
        for event in events:

            with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
                simulation = _salvus_utils.create_salvus_forward_simulation(
                    comm=self.communicator,
                    event=event,
                    iteration=self.current_iteration,
                    side_set="r1",
                )
            forward_simulations.append(simulation)

        # Submit the simulations

        with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
            self.running_job = _salvus_utils.submit_salvus_simulation(
                comm=self.communicator,
                simulations=forward_simulations,
                events=events,
                iteration=self.current_iteration,
                sim_type="forward",
                verbosity=self.salvus_verbosity,
            )
        self.forward_job = self.running_job

        # Retrieve outputs
        try:
            with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
                _salvus_utils.retrieve_salvus_simulations_blocking(
                    comm=self.communicator,
                    events=events,
                    iteration=self.current_iteration,
                    sim_type="forward",
                    verbosity=self.verbosity,
                )
        except Exception as e:
            if self.running_job is not None:
                with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
                    self.running_job.cancel()

            with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
                if self.forward_job is not None:
                    self.forward_job.delete()
                    self.forward_job = None
                if self.adjoint_job is not None:
                    self.adjoint_job.delete()
                    self.adjoint_job = None

            if self.verbosity > 0:
                print("--- Closing mesh ---")
            self.mesh.close()

            self.f1.close()
            self.f2.close()

            if self.verbosity > 0:
                print("Keyboard interrupt during forward run, closing mesh file.")
            raise e

        self.running_job = None

        with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
            _lasif.api.calculate_adjoint_sources(
                self.lasif_root, iteration=self.current_iteration, window_set="B",
            )
            self._last_computed_misfit = _lasif.api.write_misfit(
                lasif_root=self.lasif_root, iteration=self.current_iteration
            )

        return self._last_computed_misfit

    def gradient(
        self, model, iteration=None, override_misfit=False, multiply_mass=True
    ):

        self.current_model = model

        if self._last_computed_misfit is not None and not override_misfit:
            pass
        else:
            self.misfit(model)

        if self._last_computed_gradient is not None:
            return self._last_computed_gradient

        if iteration is None:
            iteration = self.current_iteration

        print("running gradient")

        # Remove previous jobs' files
        if self.adjoint_job is not None:

            with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
                self.adjoint_job.delete()
            self.adjoint_job = None

        # Create adjoint simulations ===================================================

        with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
            events = _lasif.api.list_events(
                self.lasif_root, just_list=True, iteration=iteration, output=True
            )

        adjoint_simulations = []

        for event in events:

            with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
                simulation = _salvus_utils.create_salvus_adjoint_simulation(
                    comm=self.communicator, event=event, iteration=iteration,
                )
            adjoint_simulations.append(simulation)

        # Submit adjoint simulation ====================================================

        with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
            self.running_job = _lasif.salvus_utils.submit_salvus_simulation(
                comm=self.communicator,
                simulations=adjoint_simulations,
                events=events,
                iteration=iteration,
                sim_type="adjoint",
            )
        self.adjoint_job = self.running_job

        # Retrieve =====================================================================

        with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
            events = _lasif.api.list_events(
                self.lasif_root, just_list=True, iteration=iteration, output=True
            )

        try:

            with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
                _lasif.salvus_utils.retrieve_salvus_simulations_blocking(
                    comm=self.communicator,
                    events=events,
                    iteration=iteration,
                    sim_type="adjoint",
                    verbosity=self.verbosity,
                )
        except Exception as e:
            if self.running_job is not None:
                with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
                    self.running_job.cancel()

            with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
                if self.forward_job is not None:
                    self.forward_job.delete()
                    self.forward_job = None
                if self.adjoint_job is not None:
                    self.adjoint_job.delete()
                    self.adjoint_job = None

            if self.verbosity > 0:
                print("--- Closing mesh ---")
            self.mesh.close()

            self.f1.close()
            self.f2.close()

            if self.verbosity > 0:
                print("Keyboard interrupt during adjoint run, closing mesh file.")
            raise e

        self.running_job = None

        return self.construct_gradient(multiply_mass=multiply_mass)

    def generate(self):
        pass

    def __del__(self):

        if self.running_job is not None:
            with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
                self.running_job.cancel()

        with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
            if self.forward_job is not None:
                self.forward_job.delete()
                self.forward_job = None
            if self.adjoint_job is not None:
                self.adjoint_job.delete()
                self.adjoint_job = None

        if self.verbosity > 0:
            print("--- Deleting LasifFWI object, closing mesh ---")
        self.mesh.close()

        self.f1.flush()
        self.f2.flush()
        self.f1.close()
        self.f2.close()

    def write_current_mesh(self, mesh_output_filename: str):

        mesh_filename = self.mesh.filename
        mesh_h5_mode = self.mesh.mode

        # Close the mesh
        self.mesh.close()

        # Copy the file
        _copyfile(mesh_filename, mesh_output_filename)

        # Re-open the mesh
        self.mesh = _h5py.File(mesh_filename, mesh_h5_mode)

    @staticmethod
    def create_default() -> "LasifFWI":
        raise NotImplementedError()

    def construct_gradient(self, iteration=None, multiply_mass=True) -> _numpy.ndarray:
        if iteration is None:
            iteration = self.current_iteration

        with _redirect_stdout(self.f1), _redirect_stderr(self.f2):
            events = _lasif.api.list_events(
                self.lasif_root, just_list=True, iteration=iteration, output=True
            )

        gradient = _numpy.zeros((self.mesh_dimensions, 1))

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
                    ).reshape(self.mesh_dimensions, 1,)
                else:
                    gradient += gradient_file_handle["MODEL"]["data"][()][
                        :, gradient_to_model_map, :
                    ].reshape(self.mesh_dimensions, 1)

        if self.use_splines:
            gradient = _numpy.vsplit(gradient, len(self.parametrization))

            gradient_per_parameter = [None] * len(self.parametrization)

            for i, parameter in enumerate(self.parametrization):
                gradient_per_parameter[i] = (
                    self.spline_basis[parameter].matrix_premultiplier
                    @ self.spline_basis[parameter].collocation_matrix.T
                    @ gradient[i]
                )
            return _numpy.vstack(gradient_per_parameter)

        return gradient

    def plot_bspline_models(
        self, model: _numpy.ndarray, internal_points=1000, axis=None
    ):

        assert model.shape[0] == self.dimensions
        assert self.use_splines

        if axis is None:
            _, axis = _plt.subplots(1, 1)
        models = _numpy.vsplit(model, len(self.parametrization))

        for i, parameter in enumerate(self.parametrization):

            tau = _numpy.linspace(
                self.spline_basis[parameter].radius_min,
                self.spline_basis[parameter].radius_max,
                internal_points,
            )

            profile = self.spline_basis[parameter].generate_profile(models[i], tau)

            axis.plot(
                profile, tau, "k", label=f"current model {parameter}", alpha=0.1,
            )
            axis.plot(
                self.spline_basis[parameter].background_model(tau)[:, None],
                tau,
                "r",
                label=f"background model {parameter}",
            )

        return axis

    def plot_bspline_marginals(
        self, model: _numpy.ndarray, internal_points=1000, axes=None, bins=200
    ):

        assert model.shape[0] == self.dimensions
        assert self.use_splines

        if axes is None:
            _, axes = _plt.subplots(1, len(self.parametrization))
        models = _numpy.vsplit(model, len(self.parametrization))

        for i, parameter in enumerate(self.parametrization):

            tau = _numpy.linspace(
                self.spline_basis[parameter].radius_min + 1,
                self.spline_basis[parameter].radius_max - 1,
                internal_points,
            )

            profile = self.spline_basis[parameter].generate_profile(models[i], tau)

            # profile = profile[
            #     :,
            #     _numpy.logical_and(
            #         profile[400, :] > profile[400, :].mean() - 50,
            #         profile[400, :] < profile[400, :].mean() + 50,
            #     ),
            # ]
            min = 0
            max = profile.max()

            frequency_image = _numpy.empty((internal_points, bins - 1))
            for it, t in enumerate(tau):
                hist, bin_edges = _numpy.histogram(
                    profile[it, :], bins=_numpy.linspace(min, max, bins), density=False
                )
                frequency_image[it, :] = hist

            axes[i].imshow(
                frequency_image,
                extent=[
                    min,
                    max,
                    self.spline_basis[parameter].radius_max - 1,
                    self.spline_basis[parameter].radius_min + 1,
                ],
                cmap=_plt.get_cmap("binary"),
                aspect="auto",
            )
            axes[i].set_ylabel("radius [km]")
            axes[i].set_xlabel(parameter)
            axes[i].invert_yaxis()

        return axes


class _SplineBasis:

    collocation_matrix: _scipy.sparse.csr_matrix
    matrix_premultiplier: _numpy.ndarray
    polynomial_order: int
    radius_max: float
    radius_min: float
    knots: _numpy.ndarray
    spline: _bspline.Bspline
    dimensions: int = None

    def __init__(
        self,
        mesh,
        background_model: _scipy.interpolate.interp1d,
        dof: int = 30,
        interfaces: _List = None,
        knot_locations: _List = None,
        polynomial_order=3,
        collocation_interfaces=None,
    ):

        self.polynomial_order = polynomial_order
        self.background_model = background_model

        # Assert that at least dof or knot_locations are given
        if (dof is None and knot_locations is None) or (
            dof is not None and knot_locations is not None
        ):
            raise ArgumentError()

        # Compute the radii for all points in the GLL model, such that we can do
        # transformations onto this mesh later using functions of radius (f(r)).
        radii_gll_model = (
            mesh["MODEL"]["coordinates"][:, :, 0] ** 2
            + mesh["MODEL"]["coordinates"][:, :, 1] ** 2
            + mesh["MODEL"]["coordinates"][:, :, 2] ** 2
        ) ** 0.5 / 1000

        # This block takes care to correct for discontinuities and how these interact
        # with the collaction matrices. It adds miniscule perturbations to the perceived
        # locations of the model nodes such that they lie on the correct side of
        # interfaces, and not exactly on them. ASSUMPTION: interfaces are meshed, i.e.
        # interfaces align perfectly with cells.

        # Calculate average coordinate of a cell
        element_centers = radii_gll_model.mean(axis=1)

        # Repeat this for every node
        element_centers_per_node = _numpy.repeat(
            element_centers[:, None], radii_gll_model.shape[1], axis=1
        )

        # Loop through the interfaces
        for collocation_interface in collocation_interfaces:

            # Find cells below the interface with a node on it
            nodes_just_below = _numpy.logical_and(
                _numpy.isclose(radii_gll_model, collocation_interface),
                element_centers_per_node < collocation_interface,
            )

            # Find cells above the interface with a node on it
            nodes_just_above = _numpy.logical_and(
                _numpy.isclose(radii_gll_model, collocation_interface),
                element_centers_per_node > collocation_interface,
            )

            # Add perturbations
            radii_gll_model[nodes_just_below] -= 1e-2
            radii_gll_model[nodes_just_above] += 1e-2

        # Don't penalize this numerical radius calculation too much
        assert _numpy.any(
            radii_gll_model - 6371.0 < 1e-3
        ), "Radius is much bigger than actual radius of the Earth"
        radii_gll_model[radii_gll_model > 6371.0] = 6371.0

        # Compute the extent of the model
        self.radius_max = _numpy.max(radii_gll_model)
        self.radius_min = _numpy.min(radii_gll_model)

        # Check if all the knots are between the radii
        if interfaces is not None:
            for interface in interfaces:
                assert self.radius_min <= interface <= self.radius_max, (
                    f"One of the knots of the parametrization (at {interface} km) "
                    f"lies outside of the mesh (limits {self.radius_min:.2f} and "
                    f"{self.radius_max:.2f} km)."
                )

            # Building the parametrization -------------------------------------------------

        # Place the knots within the mesh if the are not given
        if knot_locations is None:
            self.knots = _numpy.linspace(self.radius_min - 1, self.radius_max + 1, dof)
            # add endpoint repeats as appropriate for spline order p
            self.knots = _splinelab.augknt(self.knots, self.polynomial_order)
        else:
            knot_locations.sort()
            self.knots = _numpy.array(knot_locations)
            required_repeats = self.polynomial_order
            count_repeats_lower = (self.knots == self.knots[0]).sum()
            count_repeats_upper = (self.knots == self.knots[-1]).sum()
            assert required_repeats - count_repeats_lower >= 0, (
                f"Lower endpoint ({self.knots[0]}) has too many knots for the chosen "
                "polynomial order."
            )
            assert required_repeats - count_repeats_upper >= 0, (
                f"Upper endpoint ({self.knots[-1]}) has too many knots for the chosen "
                "polynomial order."
            )
            if required_repeats - count_repeats_lower:
                self.knots = _numpy.concatenate(
                    (
                        [self.knots[0]] * required_repeats - count_repeats_lower,
                        self.knots[:],
                    )
                )
            if required_repeats - count_repeats_upper:
                self.knots = _numpy.concatenate(
                    (
                        self.knots[:],
                        [self.knots[-1]] * required_repeats - count_repeats_upper,
                    )
                )

        # Insert knot repeats for discontinuities
        if interfaces is not None:
            for interface in interfaces:
                # Calculate how many knots we need based on the polynomial order
                knots_to_insert = self.polynomial_order + 1

                # If the interfaces already is a knot, we should insert less knots
                if interface in self.knots:
                    # Count how many times the knot is in there
                    count = (interface == self.knots).sum()

                    # Check that we don't have too many knots already, as that will lead to
                    # unmappable transformations, i.e. LinAlgError for the matrix
                    # premultiplier.)
                    assert knots_to_insert - count >= 0, (
                        f"One interface ({interface}) has too many knots for the chosen "
                        "polynomial order."
                    )

                    # Compute the new amount of knots to insert
                    knots_to_insert -= count

                # If there is too few knots already in place, insert them
                if knots_to_insert > 0:
                    idx = self.knots.searchsorted(interface)
                    self.knots = _numpy.concatenate(
                        (
                            self.knots[:idx],
                            [interface] * knots_to_insert,
                            self.knots[idx:],
                        )
                    )

        # Compute dimensions of the parametrization
        # self.dimensions = len(self.knots) + polynomial_order - 1 - len(interfaces)

        # Create spline object
        self.spline = _bspline.Bspline(self.knots, self.polynomial_order)

        self.collocation_matrix: _scipy.sparse.csr_matrix = _scipy.sparse.csr_matrix(
            self.spline.collmat(radii_gll_model.flatten(order="C")[:])
        )

        self.dimensions = self.collocation_matrix.shape[1]

        self.collocation_vector = self.background_model(
            radii_gll_model.flatten(order="C")[:]
        )[:, None]

        self.matrix_premultiplier = _numpy.linalg.inv(
            (self.collocation_matrix.T @ self.collocation_matrix).todense()
        )

        assert self.collocation_matrix.shape == (
            self.collocation_vector.size,
            self.dimensions,
        ), "Collocation objects and degrees of freedom are inconsistent."

    def generate_radial_collocation_matrix(
        self, internal_points=1000, derivative_order=0
    ):
        tau = _numpy.linspace(self.radius_min + 1, self.radius_max - 1, internal_points)
        A_mat = self.spline.collmat(tau, deriv_order=derivative_order)
        return tau, A_mat

    def plot_basis_functions(self, axis: _plt.axis = None):

        if axis is None:
            axis = _plt.gca()

        axisright = axis.twiny()

        tau, A0 = self.generate_radial_collocation_matrix()

        for i in range(A0.shape[1]):
            axis.plot(A0[:, i], tau, "-.b", alpha=0.1)

        axisright.plot(self.background_model(tau), tau, "k", alpha=0.5)

        axis.set_xlabel("Basis function magnitude")
        axisright.set_xlabel("Background function")
        axisright.set_xlim([0, axisright.get_xlim()[1]])
        axis.set_ylabel("Radius [km]")

    def generate_profile(self, model, tau):
        return self.spline.collmat(tau) @ model + self.background_model(tau)[:, None]

    def plot_confusion_matrix(self, axis: _plt.axis = None):

        if axis is None:
            axis = _plt.gca()

        _plt.imshow((self.collocation_matrix.T @ self.collocation_matrix).todense())
