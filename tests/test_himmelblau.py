import os as _os

import numpy as _numpy
import pytest as _pytest

import hmclab as _hmclab

sampler_classes = _hmclab.Samplers._AbstractSampler.__subclasses__()
sampler_classes.remove(_hmclab.Samplers._AbstractVisualSampler)


@_pytest.mark.parametrize("sampler_class", sampler_classes)
def test_sampling_visualization_himmelblau(sampler_class):
    dist = _hmclab.Distributions.Himmelblau(temperature=100)

    filename = "temporary_file.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)  # pragma: no cover

    sampler = sampler_class()

    sampler.sample(filename, dist, proposals=10000)

    samples_written_expected = int(
        _numpy.floor(sampler.current_proposal / sampler.online_thinning) + 1
    )

    print(f"Samples written to disk: {samples_written_expected}")

    with _hmclab.Samples(filename) as samples:
        assert samples[:, :].shape == (3, samples_written_expected)

    _os.remove(filename)


@_pytest.mark.parametrize("sampler_class", sampler_classes)
def test_sampling_interrupt_himmelblau(sampler_class):
    dist = _hmclab.Distributions.Himmelblau(temperature=100)

    filename = "temporary_file.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)  # pragma: no cover

    sampler = sampler_class()

    sampler.sample(filename, dist, proposals=1000000, online_thinning=100, max_time=2.0)

    samples_written_expected = int(
        _numpy.floor(sampler.current_proposal / sampler.online_thinning) + 1
    )

    with _hmclab.Samples(filename) as samples:
        assert samples[:, :].shape == (3, samples_written_expected)

    _os.remove(filename)
