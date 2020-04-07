"""A collection of integrated tests.
"""
import hmc_tomography as _hmc_tomography
from hmc_tomography.Helpers.CustomExceptions import (
    AbstractMethodError as _AbstractMethodError,
)
from hmc_tomography.Helpers.CustomExceptions import (
    InvalidCaseError as _InvalidCaseError,
)

import pytest as _pytest
from pytest_harvest import saved_fixture as _saved_fixture
import numpy as _numpy
import matplotlib.pyplot as _plt
import os as _os


dimensions = [1, 2, 10, 100]
distribution_classes = (
    _hmc_tomography.Distributions._AbstractDistribution.__subclasses__()
)
proposals = [100, 321, 731, 1500]


@_pytest.mark.parametrize("pclass", distribution_classes)
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("proposals", proposals)
def test_basic_sampling(
    pclass: _hmc_tomography.Distributions._AbstractDistribution,
    dimensions: int,
    proposals: int,
):

    try:
        distribution: _hmc_tomography.Distributions._AbstractDistribution = pclass.create_default(
            dimensions
        )
    except _InvalidCaseError:
        return 0

    sampler = _hmc_tomography.Samplers.RWMH()

    filename = "temporary_file.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)

    sampler.sample(filename, distribution, proposals=proposals)

    # Check if the file was created. If it wasn't, fail
    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")

    # Remove the file
    _os.remove(filename)


@_pytest.mark.parametrize("pclass", distribution_classes)
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("proposals", proposals)
def test_samples_file(
    pclass: _hmc_tomography.Distributions._AbstractDistribution,
    dimensions: int,
    proposals: int,
):

    try:
        distribution: _hmc_tomography.Distributions._AbstractDistribution = pclass.create_default(
            dimensions
        )
    except _InvalidCaseError:
        return 0

    sampler = _hmc_tomography.Samplers.RWMH()

    filename = "temporary_file.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)

    sampler.sample(filename, distribution, proposals=proposals)

    # Check if the file was created. If it wasn't, fail
    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")

    with _hmc_tomography.Post.Samples(filename) as samples:
        assert samples.raw_samples_hdf.shape == (distribution.dimensions + 1, proposals)

    # Remove the file
    _os.remove(filename)
