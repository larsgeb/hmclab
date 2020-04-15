# Common imports
import pathlib as _pathlib
from shutil import copyfile as _copyfile
from time import sleep as _sleep
from time import time as _time
from typing import Dict as _Dict

import h5py as _h5py
import toml as _toml

import lasif as _lasif
import lasif.salvus_utils as _salvus_utils
from hmc_tomography.Distributions import _AbstractDistribution


class LasifFWI(_AbstractDistribution):

    lasif_root: _pathlib.Path = None
    """Path to the LASIF project root folder."""

    communicator: _lasif.components.communicator.Communicator = None
    """LASIF communicator object from LASIF project."""

    config_dict: _Dict = None
    """LASIF configuration dictionary."""

    def __init__(self, path_to_project):

        # Set path
        self.lasif_root = _pathlib.Path(path_to_project)

        # Set communicator
        self.communicator = _lasif.api.find_project_comm(path_to_project)

        # Load configuration
        config_file_path = self.lasif_root / "lasif_config.toml"
        self.config_dict = _toml.load(config_file_path)

        # Create a backup of the mesh
        _copyfile(
            self.config_dict["lasif_project"]["domain_settings"]["domain_file"],
            self.lasif_root / "MODELS" / f"backup_domain_{_time()}.h5",
        )

        # Open the mesh
        self.mesh = _h5py.File(
            self.config_dict["lasif_project"]["domain_settings"]["domain_file"], "r+"
        )

        self.mesh["MODEL"]["data"].attrs.get("DIMENSION_LABELS")

        # Get model parametrization as a list of strings
        self.parametrization_fields = str(
            self.mesh["MODEL"]["data"].attrs.get("DIMENSION_LABELS")[1]
        ).split(" | ")[:-1]
        self.parametrization_fields[0] = self.parametrization_fields[0][4:]

        # Calculate amount of free parameters
        self.dimensions = (
            self.mesh["MODEL"]["data"].shape[0]
            * (self.mesh["MODEL"]["data"].shape[1] - 1)
            * self.mesh["MODEL"]["data"].shape[2]
        )

    @property
    def current_model(self):
        """The attribute setter for the current model.

        Gets the model directly from the mesh file"""
        return self.mesh["MODEL"]["data"][:, :-1, :].flatten(order="C")[:, None]

    @current_model.setter
    def current_model(self, current_model):
        """The attribute setter for the current model."""
        assert current_model.shape == (self.dimensions, 1)
        self.mesh["MODEL"]["data"][:, :-1, :] = current_model[:, 0].reshape(
            self.mesh["MODEL"]["data"].shape[0],
            (self.mesh["MODEL"]["data"].shape[1] - 1),
            self.mesh["MODEL"]["data"].shape[2],
        )

    def misfit(self, coordinates):

        # Set up new iteration
        current_iteration = f"it_{_time()}"
        _lasif.api.set_up_iteration(self.lasif_root, iteration=current_iteration)

        # Get all events to simulate
        events = _lasif.api.list_events(
            self.lasif_root, just_list=True, iteration=current_iteration, output=True
        )

        # Create simulation objects
        simulations = []
        for event in events:
            simulation = _salvus_utils.create_salvus_forward_simulation(
                comm=self.communicator,
                event=event,
                iteration=current_iteration,
                side_set="r1",
            )
            simulations.append(simulation)

        # Submit the simulations
        _salvus_utils.submit_salvus_simulation(
            comm=self.communicator,
            simulations=simulations,
            events=events,
            iteration=current_iteration,
            sim_type="forward",
        )

        _sleep(5)

        # Retrieve outputs
        _salvus_utils.retrieve_salvus_simulations_blocking(
            comm=self.communicator,
            events=events,
            iteration=current_iteration,
            sim_type="forward",
        )

    def gradient(self, coordinates):
        pass

    def generate(self):
        pass
