"""Elastic 2D FWI class
"""
from typing import Union as _Union

import numpy as _numpy
import psvWave as _psvWave

from hmc_tomography.Distributions import _AbstractDistribution
from hmc_tomography.Helpers.CustomExceptions import (
    InvalidCaseError as _InvalidCaseError,
)


class ElasticFullWaveform2D(_AbstractDistribution):
    forward_up_to_date: bool = False
    temperature: float = 1.0
    omp_threads_override: int = 0
    fdModel: _psvWave.fdModel

    def __init__(
        self,
        _input: _Union[str, "ElasticFullWaveform2D"],
        ux_obs: _numpy.ndarray = None,
        uz_obs: _numpy.ndarray = None,
        temperature: float = None,
        omp_threads_override: int = None,
    ):

        if type(_input) == str:
            make_copy = False
        elif self.__class__ == type(_input):
            make_copy = True
        else:
            raise ValueError(
                f"Incorrect initialization. Passed argument `{_input}` can not create "
                "an FWI object."
            )

        # Check if we are creating a new object or copying an existing one
        if not make_copy:
            # If passed, create object from the passed .ini file.
            self.fdModel = _psvWave.fdModel(_input)
        else:
            # If an object was passed, copy it.
            self.fdModel = _input.fdModel.copy()

        # Check if we need to set new data.
        if ux_obs is not None and uz_obs is not None:
            self.fdModel.set_observed_data(ux_obs, uz_obs)
        elif not make_copy:
            raise ValueError(
                "No passed observed data. Could not construct an inverse problem."
            )

        self.dimensions = self.fdModel.free_parameters

        if temperature is not None:
            # Use passed temperature
            self.temperature = temperature
        elif make_copy:
            # Copy from object
            self.temperature = _input.temperature
        else:
            # Default temperature
            self.temperature = 1.0

        if omp_threads_override is not None:
            # Use passed override
            self.omp_threads_override = omp_threads_override
        elif make_copy:
            # Copy from object
            self.omp_threads_override = _input.omp_threads_override
        else:
            # Default omp_threads_override
            self.omp_threads_override = 0

    @staticmethod
    def create_default(
        dimensions: int, ini_file: str = None, omp_threads_override: int = 0
    ) -> "ElasticFullWaveform2D":
        if ini_file is None:
            ini_file = "tests/configurations/default_testing_configuration.ini"

        # Create temporary simulation object to fake observed waveforms
        model = _psvWave.fdModel(ini_file)

        if model.free_parameters != dimensions:
            raise _InvalidCaseError()

        # Create target model
        # Get the coordinates of every grid point
        IX, IZ = model.get_coordinates(True)
        # Get the associated parameter fields
        vp, vs, rho = model.get_parameter_fields()

        x_middle = (IX.max() + IX.min()) / 2
        z_middle = (IZ.max() + IZ.min()) / 2

        # Add a circular negative anomaly to the 'true' model
        circle = ((IX - x_middle) ** 2 + (IZ - z_middle) ** 2) ** 0.5 < 15
        vs = vs * (1 - 0.1 * circle)
        vp = vp * (1 - 0.1 * circle)
        rho = rho * (1 + 0.1 * circle)

        vp_target = vp
        vs_target = vs
        rho_target = rho

        model.set_parameter_fields(vp_target, vs_target, rho_target)

        # Create 'true' data
        # print("Faking observed data")
        for i_shot in range(model.n_shots):
            model.forward_simulate(i_shot, omp_threads_override=omp_threads_override)

        # Cheating of course, as this is synthetically generated data.
        ux_obs, uz_obs = model.get_synthetic_data()
        # Noise free data, to change this, add noise below

        # Return the create wave simulation object
        return ElasticFullWaveform2D(
            ini_file,
            ux_obs=ux_obs,
            uz_obs=uz_obs,
            omp_threads_override=omp_threads_override,
        )

    def generate(self) -> _numpy.ndarray:
        raise _InvalidCaseError(
            "Can't generate samples from non-analytic distributions."
        )

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
            self.fdModel.adjoint_simulate(
                i_shot, omp_threads_override=self.omp_threads_override
            )
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
            self.fdModel.forward_simulate(
                i_shot, omp_threads_override=self.omp_threads_override
            )

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
