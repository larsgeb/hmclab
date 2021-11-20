"""A collection of integrated tests.
"""
from hmclab import Distributions
import os as _os
import copy as _copy
import h5py as _h5py

import numpy as _numpy
import pytest as _pytest

import hmclab as _hmclab

from hmclab.Helpers.CustomExceptions import (
    InvalidCaseError as _InvalidCaseError,
)

_ad = _hmclab.Distributions._AbstractDistribution
_as = _hmclab.Samplers._AbstractSampler

dimensions = [1, 2, 100]
distribution_classes = [Distributions.Normal]
sampler_classes = _as.__subclasses__()
sampler_classes.remove(_hmclab.Samplers._AbstractVisualSampler)
proposals = [10, 1000]
autotuning = [True, False]


@_pytest.mark.parametrize("sampler_class", sampler_classes)
def test_basic_copying(
    sampler_class: _as,
):

    sampler_instance = sampler_class()

    assert isinstance(sampler_instance, _as)

    sampler_instance_copy = _copy.deepcopy(sampler_instance)

    assert isinstance(sampler_instance_copy, _as)

    assert sampler_instance is not sampler_instance_copy

    assert type(sampler_instance) == type(sampler_instance_copy)


@_pytest.mark.parametrize("sampler_class", sampler_classes)
@_pytest.mark.parametrize("distribution_class", distribution_classes)
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("proposals", proposals)
@_pytest.mark.parametrize("autotuning", autotuning)
def test_samples_file(
    sampler_class: _as,
    distribution_class: _ad,
    dimensions: int,
    proposals: int,
    autotuning: bool,
):

    try:
        distribution: _ad = distribution_class.create_default(dimensions)
    except _InvalidCaseError:
        return 0

    sampler_instance = sampler_class()

    filename = "temporary_file.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)

    sampler_instance.sample(
        filename, distribution, proposals=proposals, max_time=0.5, autotuning=autotuning
    )

    # Check if the file was created. If it wasn't, fail
    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")

    samples_written_expected = int(
        _numpy.floor(
            sampler_instance.current_proposal / sampler_instance.online_thinning
        )
        + 1
    )

    with _hmclab.Samples(filename) as samples:
        # Assert that the HDF array has the right dimensions
        assert samples.numpy.shape == (
            distribution.dimensions + 1,
            samples_written_expected,
        )

        # Assert that the actual written samples have the right dimensions
        assert samples[:, :].shape == (
            distribution.dimensions + 1,
            samples_written_expected,
        )

        assert type(samples.numpy) == _numpy.ndarray

        assert type(samples.h5) == _h5py._hl.dataset.Dataset

    # Remove the file
    _os.remove(filename)

    sampler_instance_copy = _copy.deepcopy(sampler_instance)

    assert isinstance(sampler_instance_copy, _as)

    assert sampler_instance is not sampler_instance_copy

    assert type(sampler_instance) == type(sampler_instance_copy)
