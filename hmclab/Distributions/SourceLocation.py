from typing import Union as _Union, Tuple as _Tuple

import numpy as _numpy
import matplotlib.pyplot as _plt

from hmclab.Distributions import _AbstractDistribution
from hmclab.Helpers.CustomExceptions import InvalidCaseError as _InvalidCaseError
import math as _math


class SourceLocation2D(_AbstractDistribution):
    """Earthquake source location in 2D using a single velocity for the subsurface."""

    name: str = "Earthquake source location in 2D"

    infer_velocity: bool = True
    """Boolean that determines whether or not the model velocity is also a free
    parameter."""

    receiver_array_x: _numpy.ndarray = None
    receiver_array_z: _numpy.ndarray = None
    number_of_events: int = None
    medium_velocity: float = None

    def __init__(
        self,
        receiver_array_x: _numpy.ndarray,
        receiver_array_z: _numpy.ndarray,
        observed_data: _numpy.ndarray,
        data_std: _Union[_numpy.ndarray, float],
        infer_velocity: bool = True,
        medium_velocity=None,
    ):

        receiver_array_x = receiver_array_x.copy()
        receiver_array_z = receiver_array_z.copy()
        observed_data = observed_data.copy()
        if type(data_std) == _numpy.ndarray:
            data_std = data_std.copy()

        # Stations ---------------------------------------------------------------------

        # Assert that the arrays are row vectors.
        try:
            assert receiver_array_x.shape == (
                1,
                receiver_array_x.size,
            ), f"Receivers should be row vectors, not {receiver_array_x.shape}"
            assert receiver_array_z.shape == (
                1,
                receiver_array_z.size,
            ), f"Receivers should be row vectors, not {receiver_array_z.shape}"
        except AssertionError:
            receiver_array_x.shape = (1, receiver_array_x.size)
            receiver_array_z.shape = (1, receiver_array_z.size)

        self.receiver_array_x = receiver_array_x
        self.receiver_array_z = receiver_array_z

        # Free parameters --------------------------------------------------------------

        # Determine if velocity is a free parameter
        self.infer_velocity = infer_velocity
        if not self.infer_velocity:
            assert medium_velocity is not None
            self.medium_velocity = medium_velocity

        # Observed data and error statistics -------------------------------------------

        # Get number of events from data and array size
        assert (
            observed_data.size % receiver_array_x.size == 0
        ), "Data does not match receiver array"
        self.number_of_events: int = int(observed_data.size / receiver_array_x.size)
        self.number_of_stations: int = self.receiver_array_z.size
        self.number_of_datums: int = self.number_of_events * self.number_of_stations

        try:
            assert observed_data.shape == (
                self.number_of_events,
                self.number_of_stations,
            ), "Wrong shape for the observed data, trying to transpose..."
        except AssertionError:
            observed_data = observed_data.T
            assert observed_data.shape == (
                self.number_of_events,
                self.number_of_stations,
            ), "Wrong shape for the observed data, not sure what to do."

        self.observed_data = observed_data

        if type(data_std) is float:
            data_std = _numpy.ones_like(observed_data) * data_std

        try:
            assert data_std.shape == (
                self.number_of_events,
                self.number_of_stations,
            ), "Wrong shape for the data uncertainty, trying to transpose..."
        except AssertionError:
            data_std = data_std.T
            assert data_std.shape == (
                self.number_of_events,
                self.number_of_stations,
            ), "Wrong shape for the data uncertainty, not sure what to do."

        self.data_std = data_std

        # Assert that the data and data error dispersion are the same shape
        assert observed_data.shape == data_std.shape

        # Set amount of free parameters.
        self.dimensions: int = self.number_of_events * 3 + int(self.infer_velocity)

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        x, z, T, v = self.split_vector(coordinates, velocity=self.medium_velocity)

        distances = (
            (x - self.receiver_array_x) ** 2.0 + (z - self.receiver_array_z) ** 2.0
        ) ** 0.5

        return self.misfit_bounds(coordinates) + 0.5 * _numpy.nansum(
            ((self.observed_data - (T + distances / v)) / self.data_std) ** 2
        )

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:

        x, z, T, v = self.split_vector(coordinates, velocity=self.medium_velocity)

        dx = x - self.receiver_array_x
        dz = z - self.receiver_array_z

        d = (dx**2.0 + dz**2.0) ** 0.5

        # Data
        t_calc = T + d / v

        # Data gradients
        data_grad_x = dx / (v * d)
        data_grad_z = dz / (v * d)
        data_grad_v = -d / (v * v)
        data_grad_T = _numpy.ones_like(data_grad_x)

        # Misfit gradient
        misfit_grad = (t_calc - self.observed_data) / (self.data_std**2)

        # Applying chain rule
        gx = _numpy.sum(misfit_grad * data_grad_x, axis=1)
        gz = _numpy.sum(misfit_grad * data_grad_z, axis=1)
        gT = _numpy.sum(misfit_grad * data_grad_T, axis=1)
        gv = _numpy.sum(misfit_grad * data_grad_v)

        # Compiling into total gradient
        total_grad = _numpy.zeros_like(coordinates)

        if self.infer_velocity:
            total_grad[0:-1:3, 0] = gx
            total_grad[1:-1:3, 0] = gz
            total_grad[2:-1:3, 0] = gT
            total_grad[-1, 0] = gv
            return total_grad
        else:
            total_grad[0::3, 0] = gx
            total_grad[1::3, 0] = gz
            total_grad[2::3, 0] = gT
            return total_grad

    def generate(self, repeat=1, rng=_numpy.random.default_rng()) -> _numpy.ndarray:
        raise NotImplementedError(
            "Generating samples from this distribution is not implemented or supported."
        )

    @staticmethod
    def forward(x, z, T, v, receiver_array_x, receiver_array_z) -> _numpy.ndarray:

        # # Assert that the vectors are column-shaped
        # assert x.shape == (x.size, 1)
        # assert z.shape == (z.size, 1)
        # assert T.shape == (T.size, 1)
        # Assert that the vectors are the same length
        # assert x.size == z.size and x.size == T.size

        # assert receiver_array_x.shape == receiver_array_z.shape
        # assert receiver_array_x.shape == (1, receiver_array_x.size)

        # Forward calculate the travel time by dividing distance (Pythagoras) by speed,
        # and adding origin time.
        #
        # This function relies heavily on matrix broadcasting in NumPy to handle
        # multiple stations. The arrays T, x and z are column vectors. By subtracting
        # the row vectors of the station coordinates, we create matrices for the
        # traveltimes, with the following structure (example data points are filled in):
        #
        #  Station:   1      2
        # Event 1:  [5.3s,  9.0s]
        # Event 2:  [2.7s,  7,4s]
        # Event 3:  [7.1s,  8.2s]

        traveltimes: _numpy.ndarray = (
            T
            + ((x - receiver_array_x) ** 2.0 + (z - receiver_array_z) ** 2.0) ** 0.5 / v
        )

        return traveltimes

    def forward_vector(self, m):
        x, z, T, v = self.split_vector(m, velocity=self.medium_velocity)
        traveltimes: _numpy.ndarray = (
            T
            + ((x - self.receiver_array_x) ** 2.0 + (z - self.receiver_array_z) ** 2.0)
            ** 0.5
            / v
        )
        return traveltimes

    @staticmethod
    def forward_gradient(
        x, z, T, v, receiver_array_x, receiver_array_z
    ) -> _numpy.ndarray:

        # Assert that the vectors are column-shaped
        # assert x.shape == (x.size, 1)
        # assert z.shape == (z.size, 1)
        # assert T.shape == (T.size, 1)
        # Assert that the vectors are the same length
        # assert x.size == z.size and x.size == T.size

        # assert receiver_array_x.shape == receiver_array_z.shape
        # assert receiver_array_x.shape == (1, receiver_array_x.size)

        # Forward calculate the travel time by dividing distance (Pythagoras) by speed,
        # and adding origin time.
        #
        # This function relies heavily on matrix broadcasting in NumPy to handle
        # multiple stations. The arrays T, x and z are column vectors. By subtracting
        # the row vectors of the station coordinates, we create matrices for the
        # gradients, with the following structure (example data points are filled in):
        #
        #  Station:   1      2
        # Event 1:  [5.3s,  9.0s]
        # Event 2:  [2.7s,  7,4s]
        # Event 3:  [7.1s,  8.2s]

        distances = (
            (x - receiver_array_x) ** 2.0 + (z - receiver_array_z) ** 2.0
        ) ** 0.5

        gradient_x = (x - receiver_array_x) / (v * distances)
        gradient_z = (z - receiver_array_z) / (v * distances)
        gradient_T = 0.0 * gradient_x + 1.0
        gradient_v = -distances / (v * v)

        return gradient_x, gradient_z, gradient_T, gradient_v

    def describe(self):
        if self.infer_velocity:
            print("We are inferring the location of events AS WELL AS medium velocity.")
            print(
                "The parameters are ordered like: x1, z1, t1, x2, z2, t2, ... xn, "
                "zn, tn, v."
            )
        else:
            print("We are inferring the location of events GIVEN a medium velocity.")
            print(
                "The parameters are ordered like: x1, z1, t1, x2, z2, t2, ... xn, "
                "zn, tn."
            )

        print(
            f"We are placing {self.number_of_events} events using "
            f"{self.number_of_stations} stations, and a total of "
            f"{self.number_of_datums} observations"
        )

    @staticmethod
    def create_default(dimensions, seed=127):
        # Possible parameter numbers:
        # 3, 4, 6, 7, 9, 10, etcc

        events = _math.floor(dimensions / 3)

        medium_velocity = None

        if dimensions < 3:
            raise _InvalidCaseError()
        elif dimensions % 3 == 1:
            infer_velocity = True
        elif dimensions % 3 == 0:
            infer_velocity = False
        else:
            raise _InvalidCaseError()

        _numpy.random.seed(seed)

        # Create surface stations ------------------------------------------------------

        stations_x = _numpy.random.rand(1, 3) * 40 - 10
        stations_z = _numpy.zeros_like(stations_x)

        stations_z[0, -1] = 4.0

        # Create the true model and observations ---------------------------------------

        x = _numpy.random.rand(events, 1) * 20
        z = _numpy.random.rand(events, 1) * 10
        T = _numpy.random.rand(events, 1) * 10
        v = _numpy.random.rand(1, 1) * 3 + 1

        if not infer_velocity:
            medium_velocity = v

        fake_observed_data = SourceLocation2D.forward(
            x, z, T, v, stations_x, stations_z
        )

        # Create the likelihood --------------------------------------------------------

        data_std = 1.0 * _numpy.random.randn(*fake_observed_data.shape)

        fake_observed_data += data_std * _numpy.random.randn(*data_std.shape)

        return SourceLocation2D(
            stations_x,
            stations_z,
            fake_observed_data,
            data_std,
            infer_velocity=infer_velocity,
            medium_velocity=medium_velocity,
        )

    def plot_data(self, figure=None):

        if figure is None:
            figure = _plt.figure(figsize=(6, 6))

        _plt.grid()

        for event in range(self.number_of_events):
            _plt.errorbar(
                self.receiver_array_x.T,
                self.observed_data[event, :],
                yerr=self.data_std[event, :],
                fmt="x",
                label=f"Event {event+1}",
            )

        _plt.ylim([0, _plt.gca().get_ylim()[1]])

        _plt.xlabel("Horizontal distance [km]")
        _plt.ylabel("Time [s]")

    @staticmethod
    def split_vector(model_vector, velocity=None) -> _Tuple:
        if velocity is None:
            assert model_vector.size % 3 == 1
            x = model_vector[0:-1:3]
            z = model_vector[1:-1:3]
            T = model_vector[2:-1:3]
            v = model_vector[-1]
            return x, z, T, v
        else:
            assert model_vector.size % 3 == 0
            x = model_vector[0::3]
            z = model_vector[1::3]
            T = model_vector[2::3]
            return x, z, T, velocity


class SourceLocation3D(_AbstractDistribution):
    """Earthquake source location in 3D using a single velocity for the subsurface.
    Pretty close to a one-to-one copy of the 2D problem, but it was clearer to write the
    codes out separately."""

    name: str = "Earthquake source location in 3D"

    infer_velocity: bool = True
    """Boolean that determines whether or not the model velocity is also a free
    parameter."""

    receiver_array_x: _numpy.ndarray = None
    receiver_array_y: _numpy.ndarray = None
    receiver_array_z: _numpy.ndarray = None
    number_of_events: int = None
    medium_velocity: float = None

    def __init__(
        self,
        receiver_array_x: _numpy.ndarray,
        receiver_array_y: _numpy.ndarray,
        receiver_array_z: _numpy.ndarray,
        observed_data: _numpy.ndarray,
        data_std: _Union[_numpy.ndarray, float],
        infer_velocity: bool = True,
        medium_velocity=None,
    ):

        receiver_array_x = receiver_array_x.copy()
        receiver_array_y = receiver_array_y.copy()
        receiver_array_z = receiver_array_z.copy()
        observed_data = observed_data.copy()
        if type(data_std) == _numpy.ndarray:
            data_std = data_std.copy()

        # Stations ---------------------------------------------------------------------

        # Assert that the arrays are row vectors.
        try:
            assert receiver_array_x.shape == (
                1,
                receiver_array_x.size,
            ), f"Receivers should be row vectors, not {receiver_array_x.shape}"
            assert receiver_array_y.shape == (
                1,
                receiver_array_y.size,
            ), f"Receivers should be row vectors, not {receiver_array_y.shape}"
            assert receiver_array_z.shape == (
                1,
                receiver_array_z.size,
            ), f"Receivers should be row vectors, not {receiver_array_z.shape}"
        except AssertionError:
            receiver_array_x.shape = (1, receiver_array_x.size)
            receiver_array_y.shape = (1, receiver_array_y.size)
            receiver_array_z.shape = (1, receiver_array_z.size)

        self.receiver_array_x = receiver_array_x
        self.receiver_array_y = receiver_array_y
        self.receiver_array_z = receiver_array_z

        # Free parameters --------------------------------------------------------------

        # Determine if velocity is a free parameter
        self.infer_velocity = infer_velocity
        if not self.infer_velocity:
            assert medium_velocity is not None
            self.medium_velocity = medium_velocity

        # Observed data and error statistics -------------------------------------------

        # Get number of events from data and array size
        assert observed_data.size % receiver_array_x.size == 0
        self.number_of_events: int = int(observed_data.size / receiver_array_x.size)
        self.number_of_stations: int = self.receiver_array_z.size
        self.number_of_datums: int = self.number_of_events * self.number_of_stations

        try:
            assert observed_data.shape == (
                self.number_of_events,
                self.number_of_stations,
            ), "Wrong shape for the observed data, trying to transpose..."
        except AssertionError:
            observed_data = observed_data.T
            assert observed_data.shape == (
                self.number_of_events,
                self.number_of_stations,
            ), "Wrong shape for the observed data, not sure what to do."

        self.observed_data = observed_data

        if type(data_std) is float:
            data_std = _numpy.ones_like(observed_data) * data_std

        try:
            assert data_std.shape == (
                self.number_of_events,
                self.number_of_stations,
            ), "Wrong shape for the data uncertainty, trying to transpose..."
        except AssertionError:
            data_std = data_std.T
            assert data_std.shape == (
                self.number_of_events,
                self.number_of_stations,
            ), "Wrong shape for the data uncertainty, not sure what to do."

        self.data_std = data_std

        # Assert that the data and data error dispersion are the same shape
        assert observed_data.shape == data_std.shape

        # Set amount of free parameters.
        self.dimensions: int = self.number_of_events * 4 + int(self.infer_velocity)

    def misfit(self, coordinates: _numpy.ndarray) -> float:
        x, y, z, T, v = self.split_vector(coordinates, velocity=self.medium_velocity)

        distances = (
            (x - self.receiver_array_x) ** 2.0
            + (y - self.receiver_array_y) ** 2.0
            + (z - self.receiver_array_z) ** 2.0
        ) ** 0.5

        return self.misfit_bounds(coordinates) + 0.5 * _numpy.nansum(
            ((self.observed_data - (T + distances / v)) / self.data_std) ** 2
        )

    def gradient(self, coordinates: _numpy.ndarray) -> _numpy.ndarray:

        x, y, z, T, v = self.split_vector(coordinates, velocity=self.medium_velocity)

        dx = x - self.receiver_array_x
        dy = y - self.receiver_array_y
        dz = z - self.receiver_array_z

        d = (dx**2.0 + dy**2.0 + dz**2.0) ** 0.5

        # Data
        t_calc = T + d / v

        # Data gradients
        data_grad_x = dx / (v * d)
        data_grad_y = dy / (v * d)
        data_grad_z = dz / (v * d)
        data_grad_v = -d / (v * v)
        data_grad_T = _numpy.ones_like(data_grad_x)

        # Misfit gradient
        misfit_grad = (t_calc - self.observed_data) / (self.data_std**2)

        # Applying chain rule
        gx = _numpy.nansum(misfit_grad * data_grad_x, axis=1)
        gy = _numpy.nansum(misfit_grad * data_grad_y, axis=1)
        gz = _numpy.nansum(misfit_grad * data_grad_z, axis=1)
        gT = _numpy.nansum(misfit_grad * data_grad_T, axis=1)
        gv = _numpy.nansum(misfit_grad * data_grad_v)

        # Compiling into total gradient
        total_grad = _numpy.zeros_like(coordinates)

        if self.infer_velocity:
            total_grad[0:-1:4, 0] = gx
            total_grad[1:-1:4, 0] = gy
            total_grad[2:-1:4, 0] = gz
            total_grad[3:-1:4, 0] = gT
            total_grad[-1, 0] = gv
            return total_grad
        else:
            total_grad[0::4, 0] = gx
            total_grad[1::4, 0] = gy
            total_grad[2::4, 0] = gz
            total_grad[3::4, 0] = gT
            return total_grad

    def generate(self, repeat=1, rng=_numpy.random.default_rng()) -> _numpy.ndarray:
        raise NotImplementedError(
            "Generating samples from this distribution is not implemented or supported."
        )

    @staticmethod
    def forward(
        x, y, z, T, v, receiver_array_x, receiver_array_y, receiver_array_z
    ) -> _numpy.ndarray:

        traveltimes: _numpy.ndarray = (
            T
            + (
                (x - receiver_array_x) ** 2.0
                + (y - receiver_array_y) ** 2.0
                + (z - receiver_array_z) ** 2.0
            )
            ** 0.5
            / v
        )

        return traveltimes

    def forward_vector(self, m):
        x, y, z, T, v = self.split_vector(m, velocity=self.medium_velocity)
        traveltimes: _numpy.ndarray = (
            T
            + (
                (x - self.receiver_array_x) ** 2.0
                + (y - self.receiver_array_y) ** 2.0
                + (z - self.receiver_array_z) ** 2.0
            )
            ** 0.5
            / v
        )
        return traveltimes

    @staticmethod
    def forward_gradient(
        x, y, z, T, v, receiver_array_x, receiver_array_y, receiver_array_z
    ) -> _numpy.ndarray:

        distances = (
            (x - receiver_array_x) ** 2.0
            + (y - receiver_array_y) ** 2.0
            + (z - receiver_array_z) ** 2.0
        ) ** 0.5

        gradient_x = (x - receiver_array_x) / (v * distances)
        gradient_y = (y - receiver_array_y) / (v * distances)
        gradient_z = (z - receiver_array_z) / (v * distances)
        gradient_T = 0.0 * gradient_x + 1.0
        gradient_v = -distances / (v * v)

        return gradient_x, gradient_y, gradient_z, gradient_T, gradient_v

    def describe(self):
        if self.infer_velocity:
            print("We are inferring the location of events AS WELL AS medium velocity.")
            print(
                "The parameters are ordered like: x1, y1, z1, t1, x2, y2, z2, t2, ..."
                " xn, yn, zn, tn, v."
            )
        else:
            print("We are inferring the location of events GIVEN a medium velocity.")
            print(
                "The parameters are ordered like: x1, y1, z1, t1, x2, y2, z2, t2, ..."
                " xn, yn, zn, tn."
            )

        print(
            f"We are placing {self.number_of_events} events using "
            f"{self.number_of_stations} stations, and a total of "
            f"{self.number_of_datums} observations"
        )

    @staticmethod
    def create_default(dimensions, seed=127):

        # Possible parameter numbers 4 * n_events + (opt velocity) 1:
        # 4, 5, 8, 9, 12, 13, etc

        events = _math.floor(dimensions / 4)

        medium_velocity = None

        if dimensions < 4:
            raise _InvalidCaseError()
        elif dimensions % 4 == 1:
            infer_velocity = True
        elif dimensions % 4 == 0:
            infer_velocity = False
        else:
            raise _InvalidCaseError()

        _numpy.random.seed(seed)

        # Create surface stations ------------------------------------------------------
        n_stations = 3
        stations_x = _numpy.random.rand(1, n_stations) * 40 - 10
        stations_y = _numpy.random.rand(1, n_stations) * 40 - 10
        stations_z = _numpy.zeros_like(stations_x)

        # Create the true model and observations ---------------------------------------

        x = _numpy.random.rand(events, 1) * 20
        y = _numpy.random.rand(events, 1) * 20
        z = _numpy.random.rand(events, 1) * 10
        T = _numpy.random.rand(events, 1) * 10
        v = _numpy.random.rand(1, 1) * 3 + 1

        if not infer_velocity:
            medium_velocity = v

        fake_observed_data = SourceLocation3D.forward(
            x, y, z, T, v, stations_x, stations_y, stations_z
        )

        # Create the likelihood --------------------------------------------------------

        data_std = 1.0 * _numpy.random.randn(*fake_observed_data.shape)

        fake_observed_data += data_std * _numpy.random.randn(*data_std.shape)

        return SourceLocation3D(
            stations_x,
            stations_y,
            stations_z,
            fake_observed_data,
            data_std,
            infer_velocity=infer_velocity,
            medium_velocity=medium_velocity,
        )

    def plot_data(self, figure=None):

        if figure is None:
            figure = _plt.figure(figsize=(6, 6))

        _plt.grid()

        for event in range(self.number_of_events):
            _plt.errorbar(
                self.receiver_array_x.T,
                self.observed_data[event, :],
                yerr=self.data_std[event, :],
                fmt="x",
                label=f"Event {event+1}",
            )

        _plt.ylim([0, _plt.gca().get_ylim()[1]])

        _plt.xlabel("Horizontal distance [km]")
        _plt.ylabel("Time [s]")

    @staticmethod
    def split_vector(model_vector, velocity=None) -> _Tuple:
        if velocity is None:
            assert model_vector.size % 4 == 1
            x = model_vector[0:-1:4]
            y = model_vector[1:-1:4]
            z = model_vector[2:-1:4]
            T = model_vector[3:-1:4]
            v = model_vector[-1]
            return x, y, z, T, v
        else:
            assert model_vector.size % 4 == 0
            x = model_vector[0::4]
            y = model_vector[1::4]
            z = model_vector[2::4]
            T = model_vector[3::4]
            return x, y, z, T, velocity
