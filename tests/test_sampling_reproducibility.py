"""A collection of integrated tests.
"""
from hmclab import Distributions as _Distributions
import os as _os

import numpy as _numpy
import pytest as _pytest

import hmclab as _hmclab
from hmclab.Helpers.CustomExceptions import InvalidCaseError as _InvalidCaseError

_ad = _hmclab.Distributions._AbstractDistribution
_as = _hmclab.Samplers._AbstractSampler

dimensions = [1, 2, 3]
distribution_classes = _Distributions._AbstractDistribution.__subclasses__()
sampler_classes = _as.__subclasses__()
sampler_classes.remove(_hmclab.Samplers._AbstractVisualSampler)

proposals = [3, 10]
autotuning = [True, False]


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
        return _pytest.skip("Invalid case")

    sampler_instance_1 = sampler_class(seed=1)

    filename_1 = "temporary_file_1.h5"

    sampler_instance_2 = sampler_class(seed=1)

    filename_2 = "temporary_file_2.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename_1):
        _os.remove(filename_1)
    if _os.path.exists(filename_2):
        _os.remove(filename_2)

    try:
        initial_model = distribution.generate()
    except:
        initial_model = _numpy.ones((distribution.dimensions, 1))

    sampler_instance_1.sample(
        filename_1,
        distribution,
        proposals=proposals,
        initial_model=initial_model,
        max_time=0.1,
        autotuning=autotuning,
        disable_progressbar=True,
    )
    sampler_instance_2.sample(
        filename_2,
        distribution,
        proposals=proposals,
        initial_model=initial_model,
        max_time=0.1,
        autotuning=autotuning,
        disable_progressbar=True,
    )

    # Check if the file was created. If it wasn't, fail
    if not _os.path.exists(filename_1) or not _os.path.exists(filename_2):
        _pytest.fail("Samples file wasn't created")

    samples_written_expected_1 = int(
        _numpy.floor(
            sampler_instance_1.current_proposal / sampler_instance_1.online_thinning
        )
        + 1
    )

    samples_written_expected_2 = int(
        _numpy.floor(
            sampler_instance_2.current_proposal / sampler_instance_2.online_thinning
        )
        + 1
    )

    with _hmclab.Samples(filename_1) as samples_1, _hmclab.Samples(
        filename_2
    ) as samples_2:
        # Assert that the HDF array has the right dimensions
        assert samples_1.numpy.shape == (
            distribution.dimensions + 1,
            samples_written_expected_1,
        )
        assert samples_2.numpy.shape == (
            distribution.dimensions + 1,
            samples_written_expected_2,
        )

        min_written_samples = min(
            samples_written_expected_1, samples_written_expected_2
        )

        var_a = samples_1[:, :min_written_samples]
        var_b = samples_2[:, :min_written_samples]

        assert _numpy.all(
            samples_1[:, :min_written_samples] == samples_2[:, :min_written_samples]
        )

    # Remove the file
    _os.remove(filename_1)
    _os.remove(filename_2)
