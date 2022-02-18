"""Elastic 2D FWI class
"""
from typing import Union as _Union, List as _List

import numpy as _numpy
import psvWave as _psvWave

from hmclab.Distributions import _AbstractDistribution
from hmclab.Helpers.CustomExceptions import InvalidCaseError as _InvalidCaseError


class ElasticFullWaveform2D(_AbstractDistribution):
    forward_up_to_date: bool = False
    temperature: float = 1.0
    omp_threads_override: int = 0
    fdModel: _psvWave.fdModel
    use_blob_par: bool = False

    def __init__(
        self,
        _input: _Union[str, "ElasticFullWaveform2D", dict],
        ux_obs: _numpy.ndarray = None,
        uz_obs: _numpy.ndarray = None,
        temperature: float = None,
        omp_threads_override: int = None,
        free_parameter_grid: _List = None,
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

        if free_parameter_grid is not None:
            self.make_blob(*free_parameter_grid)

    @staticmethod
    def create_default(
        dimensions: int, ini_file: str = None, omp_threads_override: int = 0
    ) -> "ElasticFullWaveform2D":
        if ini_file is None:
            ini_file = "tests/configurations/default_testing_configuration.ini"

        # Create temporary simulation object to fake observed waveforms
        model = _psvWave.fdModel(ini_file)

        if model.free_parameters != dimensions:
            raise _InvalidCaseError(
                f"{model.free_parameters} is not equal to {dimensions}"
            )

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

    def generate(self, repeat=1, rng=_numpy.random.default_rng()) -> _numpy.ndarray:
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

        gradient = self.fdModel.get_gradient_vector()[:, None] / self.temperature
        if self.use_blob_par:
            return self.blob.inverse_transform(gradient)
        else:
            return gradient

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
        if self.use_blob_par:
            return self.blob.get_model_vector()[:, None]
        else:
            return self.fdModel.get_model_vector()[:, None]

    def set_model_vector(self, m: _numpy.ndarray):
        self.forward_up_to_date = False
        assert m.shape == (
            self.dimensions,
            1,
        ), f"shape {m.shape} and problem dimensions {self.dimensions} don't match"

        if self.use_blob_par:
            self.blob.set_model_vector(m)
        else:
            self.fdModel.set_model_vector(m[:, 0])

    def plot_model_vector(self, m, *args, **kwargs):
        if self.use_blob_par:
            return self.fdModel.plot_model_vector(
                self.blob.forward_transform_background(m), *args, **kwargs
            )
        else:
            return self.fdModel.plot_model_vector(m, *args, **kwargs)

    def make_blob(self, nx, nz):
        if not self.use_blob_par:
            self.use_blob_par = True
            self.blob = BlobParametrization(self.fdModel, nx, nz)
            self.dimensions = self.blob.free_parameters
        elif self.blob.nx != nx or self.blob.nz != nz:
            print(
                f"Parametrization changed from {self.blob.nx} by "
                "{self.blob.nz} to {nx} by {nz}."
            )
            self.use_blob_par = True
            self.blob = BlobParametrization(self.fdModel, nx, nz)
            self.dimensions = self.blob.free_parameters
        else:
            print("Parametrization unchanged.")


class BlobParametrization:
    def __init__(self, model: _psvWave.fdModel, nx, nz):

        self.nx = nx
        self.nz = nz

        self.fdModel = model

        # Save background fields
        self.background_models = self.fdModel.get_model_vector()

        # Compute free parameters in new basis
        self.free_parameters = nx * nz * 3

        # Get coordinate fields
        self.coordinates_x, self.coordinates_z = self.fdModel.get_coordinates(False)

        # Extract only coordinates for free parameters
        self.coordinates_x = self.coordinates_x[
            self.fdModel.np_boundary
            + self.fdModel.nx_inner_boundary : -(
                self.fdModel.np_boundary + self.fdModel.nx_inner_boundary
            ),
            self.fdModel.nz_inner_boundary
            + self.fdModel.np_boundary : -(
                self.fdModel.nz_inner_boundary + self.fdModel.np_boundary
            ),
        ]
        self.coordinates_z = self.coordinates_z[
            self.fdModel.np_boundary
            + self.fdModel.nx_inner_boundary : -(
                self.fdModel.np_boundary + self.fdModel.nx_inner_boundary
            ),
            self.fdModel.nz_inner_boundary
            + self.fdModel.np_boundary : -(
                self.fdModel.nz_inner_boundary + self.fdModel.np_boundary
            ),
        ]

        # Define Gaussian center points as indicate in graph
        #       0                     max_x
        # max_z -----------------------  |  |
        #       |                     |  |  |  Par3
        #       |   *3     *4     *5  |  | /
        #       |                     |  |/
        #       |                     |  |  Par2
        #       |   *0     *1     *2  | /
        #       |                     |/
        #     0 ----------------------- Par1
        length_x = self.coordinates_x.max() - self.coordinates_x.min()
        length_z = self.coordinates_z.max() - self.coordinates_z.min()
        dx = length_x / nx
        dz = length_z / nz
        self.points_x = dx * 0.5 + _numpy.arange(nx) * dx + self.coordinates_x.min()
        self.points_z = dz * 0.5 + _numpy.arange(nz) * dz + self.coordinates_z.min()

        self.collocation_matrix = _numpy.zeros(
            (self.fdModel.free_parameters, self.free_parameters)
        )

        fr_div3 = int(self.fdModel.free_parameters / 3)

        # Construct collocation matrix
        for i_x, x_cor in enumerate(self.points_x):
            for i_z, z_cor in enumerate(self.points_z):
                index = i_x * nz + i_z
                distance = _numpy.exp(
                    -(
                        ((self.coordinates_x - x_cor) / (0.5**0.5 * dx)) ** 2
                        + ((self.coordinates_z - z_cor) / (0.5**0.5 * dz)) ** 2
                    )
                )

                # Write out basis function for VP
                self.collocation_matrix[:fr_div3, index] = distance.flatten(order="F")

                # Write out basis function for VS
                self.collocation_matrix[
                    fr_div3 : fr_div3 * 2, index + nx * nz
                ] = distance.flatten(order="F")

                # Write out basis function for RHO
                self.collocation_matrix[
                    fr_div3 * 2 : fr_div3 * 3, index + nx * nz * 2
                ] = distance.flatten(order="F")

        self.confusion_matrix = _numpy.linalg.inv(
            self.collocation_matrix.T @ self.collocation_matrix
        )

    def get_model_vector(self):
        return self.inverse_transform_background(self.fdModel.get_model_vector())

    def inverse_transform_background(self, vector):
        return (
            self.confusion_matrix
            @ self.collocation_matrix.T
            @ (vector - self.background_models)
        )

    def inverse_transform(self, vector):
        return self.confusion_matrix @ self.collocation_matrix.T @ vector

    def forward_transform_background(self, vector):
        return self.collocation_matrix @ vector[:, 0] + self.background_models

    def set_model_vector(self, vector):
        # if vector.shape == (self.free_parameters, 1):
        self.fdModel.set_model_vector(self.forward_transform_background(vector))
        # else:
        #     self.fdModel.set_model_vector(
        #         self.forward_transform_background(vector[:, None])
        #     )
