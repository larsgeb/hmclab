"""Elastic 2D FWI class
"""
import warnings as _warnings
from typing import Union as _Union

import numpy as _numpy

from hmc_tomography.Distributions import _AbstractDistribution

import pyWave_cpp


class pyWave(_AbstractDistribution):
    forward_up_to_date: bool = False
    temperature: float = 1.0

    def __init__(self, ini_file, ux_obs, uz_obs, temperature: float = 1.0):

        self.fdModel = pyWave_cpp.fdModel(ini_file)

        self.fdModel.set_observed_data(ux_obs, uz_obs)

        self.dimensions = self.fdModel.free_parameters

        self.temperature = temperature

    @staticmethod
    def create_default() -> "pyWave":
        ini_file = "hmc_tomography/Tests/configurations/forward_configuration.ini"

        # Create temporary simulation object to fake observed waveforms
        model = pyWave_cpp.fdModel(ini_file)

        # Create target model
        # Get the coordinates of every grid point
        IX, IZ = model.get_coordinates(True)
        extent = model.get_extent(True)
        # Get the associated parameter fields
        vp, vs, rho = model.get_parameter_fields()

        vp_starting = vp
        vs_starting = vs
        rho_starting = rho

        x_middle = (IX.max() + IX.min()) / 2
        z_middle = (IZ.max() + IZ.min()) / 2

        circle = ((IX - x_middle) ** 2 + (IZ - z_middle) ** 2) ** 0.5 < 15
        vs = vs * (1 - 0.1 * circle)
        vp = vp * (1 - 0.1 * circle)
        rho = rho * (1 + 0.1 * circle)

        vp_target = vp
        vs_target = vs
        rho_target = rho

        model.set_parameter_fields(vp_target, vs_target, rho_target)

        # Create true data
        # print("Faking observed data")
        for i_shot in range(model.n_shots):
            model.forward_simulate(i_shot)

        # Cheating of course, as this is synthetically generated data.
        ux_obs, uz_obs = model.get_synthetic_data()

        return pyWave(ini_file, ux_obs, uz_obs)

    def generate(self) -> _numpy.ndarray:
        pass

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:

        if not _numpy.all(coordinates == self.get_model_vector()):
            # print("Updating model")
            self.set_model_vector(coordinates)
        else:
            # print("Not updating model")
            pass

        if not self.forward_up_to_date:
            self.misfit(coordinates)

        self.fdModel.reset_kernels()
        for i_shot in range(self.fdModel.n_shots):
            self.fdModel.adjoint_simulate(i_shot)
        self.fdModel.map_kernels_to_velocity()

        return self.fdModel.get_gradient_vector()[:, None] / self.temperature

    def misfit(self, coordinates: _numpy.ndarray) -> float:

        if not _numpy.all(coordinates == self.get_model_vector()):
            # print("Updating model")
            self.set_model_vector(coordinates)
        else:
            # print("Not updating model")
            pass

        if self.forward_up_to_date:
            return self.fdModel.misfit

        for i_shot in range(self.fdModel.n_shots):
            self.fdModel.forward_simulate(i_shot)

        self.fdModel.calculate_l2_misfit()
        self.fdModel.calculate_l2_adjoint_sources()
        self.forward_up_to_date = True

        return self.fdModel.misfit / self.temperature

    def get_model_vector(self) -> _numpy.ndarray:
        return self.fdModel.get_model_vector()[:, None]

    def set_model_vector(self, m: _numpy.ndarray):
        self.forward_up_to_date = False
        assert m.shape == (self.dimensions, 1)
        self.fdModel.set_model_vector(m[:, 0])
