"""
HMCLab base module.

"""
import warnings as _warnings

import numpy as _numpy


def _is_vector(arr, size=None):
    """
    Check if the NumPy array is shaped like a vector (column, row, or other).

    Parameters:
    arr (numpy.ndarray): Input NumPy array.

    Returns:
    bool: True if the array is a vector, False otherwise.
    """
    if size is None:
        return arr.ndim == 1 and arr.size > 0
    else:
        return arr.ndim == 1 and arr.size == size


def _parse_vector_input(arr, size=None, warn_if_changed=True):
    """
    Create a new NumPy array with shape (size,) while handling floats/ints if
    size is 1.

    Parameters:
    arr (numpy.ndarray, float, int): Input NumPy array or scalar.
    size (int or None): Specific size for the output array (optional).
    warn_if_changed (bool): Whether to issue a warning if the input is not
    exactly as generated (optional).

    Returns:
    numpy.ndarray: New NumPy array with shape (size,).
    """

    def issue_warning():
        if warn_if_changed:
            _warnings.warn(
                "Input array has been altered to match "
                "the desired shape and type; (size,)."
            )

    if isinstance(arr, list):
        arr = _numpy.array(arr)

    if isinstance(arr, (float, int)):
        if size is None or size == 1:
            return _numpy.array([arr])
        else:
            raise ValueError(f"Input must be size {size}, not scalar.")
    elif isinstance(arr, _numpy.ndarray):
        if size is None:
            new_arr = arr.ravel()
            if new_arr.shape != arr.shape:
                issue_warning()
            return new_arr.copy()
        elif size == 1:
            if arr.size == 1:
                return arr.ravel()
            else:
                raise ValueError("Array size must be 1 when size is 1.")
        else:
            new_arr = arr.reshape((size,))
            if new_arr.shape != arr.shape:
                issue_warning()
            return new_arr.copy()
    else:
        raise ValueError("Input must be a NumPy array, float, or int.")


class StandardNormalDistribution:
    dimensions = 1
    name = "Standard Normal Distribution"
    descriptions = """A 1-dimensional normal distribution with a variance of 1
and a mean of 0."""

    def f(self, m):
        m = _parse_vector_input(m, size=self.dimensions)
        return 0.5 * float(_numpy.sum(m)) ** 2

    def g(self, m):
        m = _parse_vector_input(m, size=self.dimensions)
        return m
