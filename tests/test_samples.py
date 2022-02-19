"""A collection of integrated tests.
"""
from hmclab.Samples import combine_samples as _combine_samples
import os as _os

import numpy as _numpy
import pytest as _pytest

import hmclab as _hmclab
from hmclab.Helpers.CustomExceptions import InvalidCaseError as _InvalidCaseError

_ad = _hmclab.Distributions._AbstractDistribution
_as = _hmclab.Samplers._AbstractSampler

dimensions = [1, 2, 50]
distribution_classes = _ad.__subclasses__()
sampler_classes = [_hmclab.Samplers.RWMH]  # Doesn't impact the test
proposals = [5, 100]


@_pytest.mark.parametrize("sampler_class", sampler_classes)
@_pytest.mark.parametrize("distribution_class", distribution_classes)
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("proposals", proposals)
def test_samples_detail(
    sampler_class: _as,
    distribution_class: _ad,
    dimensions: int,
    proposals: int,
):

    try:
        distribution: _ad = distribution_class.create_default(dimensions)
    except _InvalidCaseError:
        return 0

    sampler_instance = sampler_class()

    assert isinstance(sampler_instance, _as)

    filename = "temporary_file.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)  # pragma: no cover

    sampler_instance.sample(
        filename,
        distribution,
        proposals=proposals,
        ram_buffer_size=int(proposals / _numpy.random.rand() * 10),
        max_time=0.1,
    )

    # Check if the file was created. If it wasn't, fail
    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")

    with _hmclab.Samples(filename) as samples:
        samples.print_details()

    # Remove the file
    _os.remove(filename)


@_pytest.mark.parametrize("sampler_class", sampler_classes)
@_pytest.mark.parametrize("distribution_class", distribution_classes)
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("proposals", proposals)
def test_samples_concat(
    sampler_class: _as,
    distribution_class: _ad,
    dimensions: int,
    proposals: int,
):

    try:
        distribution: _ad = distribution_class.create_default(dimensions)
    except _InvalidCaseError:
        return 0

    sampler_instance = sampler_class()

    assert isinstance(sampler_instance, _as)

    filename_1 = "temporary_file_1.h5"
    filename_2 = "temporary_file_2.h5"
    filenames = [filename_1, filename_2]

    # Remove file before attempting to sample
    for filename in filenames:
        if _os.path.exists(filename):
            _os.remove(filename)

        sampler_instance.sample(
            filename,
            distribution,
            proposals=proposals,
            ram_buffer_size=int(proposals / _numpy.random.rand() * 10),
            max_time=0.1,
        )

        # Check if the file was created. If it wasn't, fail
        if not _os.path.exists(filename):
            _pytest.fail("Samples file wasn't created")

    combined_samples = _combine_samples(filenames)

    # The sample files also contain the misfit, so + 1
    assert combined_samples.shape[0] == distribution.dimensions + 1

    for filename in filenames:

        # Remove the file
        _os.remove(filename)
