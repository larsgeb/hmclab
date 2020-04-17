import numpy as _numpy


class AbstractMethodError(Exception):
    def __init__(self, message="", errors=""):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors


class InvalidCaseError(Exception):
    def __init__(self, message="", errors=""):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

        # Now for your custom code...
        self.errors = errors


class Assertions:
    v_shape: str = "The passed vector is not of the correct shape."

    @staticmethod
    def assert_shape(v, shape):

        # Assert the passed parameters
        assert type(v) == _numpy.ndarray, "The passed object is not a numpy.ndarray."
        assert type(shape) == tuple, "The passed shape is not a tuple."

        type_ndarray = "vector" if shape[1] == 1 else "matrix"

        assert v.shape == shape, (
            f"The passed {type_ndarray} with shape: {v.shape} is not of the correct "
            f"shape, which would be {shape}."
        )
