"""A collection of tests for likelihood functions.
"""
import matplotlib.pyplot as _plt
import numpy as _numpy
import pytest as _pytest
import dill as _dill

from hmclab import Distributions as _Distributions
from hmclab.Helpers.CustomExceptions import InvalidCaseError as _InvalidCaseError

dimensions = [1, 2, 5, 50]
subclasses = _Distributions._AbstractDistribution.__subclasses__()
deltas = [1e-10, 1e-2, -1e-10, -1e-2]


@_pytest.mark.parametrize("pclass", subclasses)
@_pytest.mark.parametrize("dimensions", dimensions)
def test_creation(pclass: _Distributions._AbstractDistribution, dimensions: int):
    # Create the object
    try:
        distribution: _Distributions._AbstractDistribution = pclass.create_default(
            dimensions
        )
    except _InvalidCaseError:
        return 0

    # Check if a subtype of mass matrices
    assert issubclass(type(distribution), _Distributions._AbstractDistribution)

    # Check if the right amount of dimensions
    assert distribution.dimensions == dimensions

    try:
        assert _dill.pickles(distribution)
    except AssertionError as e:
        print(_dill.detect.badtypes(distribution, depth=1))
        raise e
