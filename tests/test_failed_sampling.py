"""A collection of integrated tests.
"""
import os as _os

import numpy as _numpy
import pytest as _pytest

import hmclab as _hmclab
from hmclab.Helpers.CustomExceptions import (
    InvalidCaseError as _InvalidCaseError,
)

_ad = _hmclab.Distributions._AbstractDistribution
_as = _hmclab.Samplers._AbstractSampler

dimensions = [1, 2, 10]
distribution_classes = _ad.__subclasses__()
sampler_classes = _as.__subclasses__()
proposals = [10, 1000]  # , 731, 1500]


@_pytest.mark.parametrize("sampler_class", sampler_classes)
@_pytest.mark.parametrize("distribution_class", distribution_classes)
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("proposals", proposals)
def test_basic_sampling(
    sampler_class: _as,
    distribution_class: _ad,
    dimensions: int,
    proposals: int,
):

    try:
        distribution: _ad = distribution_class.create_default(dimensions)
    except _InvalidCaseError:
        return 0

    sampler = sampler_class()

    assert isinstance(sampler, _as)

    filename = "temporary_file.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)

    try:
        sampler.sample()
    except Exception as e:
        print(e)

    try:
        sampler.sample(
            filename,
            distribution,
            proposals=proposals,
            ram_buffer_size=int(proposals / _numpy.random.rand() * 10),
            max_time=1.0,
            mass_matrix=_hmclab.MassMatrices.Unit(434),
        )
    except Exception as e:
        print(e)

    sampler.sample(
        filename,
        distribution,
        proposals=proposals,
        online_thinning=10,
        ram_buffer_size=int(proposals / _numpy.random.rand() * 10),
        max_time=1.0,
        overwrite_existing_file=True,
    )

    # Check if the file was created. If it wasn't, fail
    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")

    # Remove the file
    _os.remove(filename)
