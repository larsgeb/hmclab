"""A collection of integrated tests.
"""
from hmclab.Samples import combine_samples as _combine_samples
import os as _os

import numpy as _numpy
import pytest as _pytest
import uuid as _uuid


import hmclab as _hmclab
from hmclab.Helpers.CustomExceptions import InvalidCaseError as _InvalidCaseError

_ad = _hmclab.Distributions._AbstractDistribution
_as = _hmclab.Samplers._AbstractSampler

dimensions = [1, 2, 50]
distribution_classes = _ad.__subclasses__()
sampler_classes = [_hmclab.Samplers.RWMH]  # Doesn't impact the test
proposals = [5, 100]
extensions = ["h5", "npy"]


@_pytest.mark.parametrize("sampler_class", sampler_classes)
@_pytest.mark.parametrize("distribution_class", distribution_classes)
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("proposals", proposals)
@_pytest.mark.parametrize("extension", extensions)
def test_samples_detail(
    sampler_class: _as,
    distribution_class: _ad,
    dimensions: int,
    proposals: int,
    extension: str,
):
    try:
        distribution: _ad = distribution_class.create_default(dimensions)
    except _InvalidCaseError:
        return _pytest.skip("Invalid case")

    sampler_instance = sampler_class()

    assert isinstance(sampler_instance, _as)

    unique_name = _uuid.uuid4().hex.upper()
    filename = f"temporary_file_{unique_name}.{extension}"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)  # pragma: no cover

    try:
        initial_model = distribution.generate()
    except:
        initial_model = _numpy.ones((distribution.dimensions, 1))

    sampler_instance.sample(
        filename,
        distribution,
        proposals=proposals,
        initial_model=initial_model,
        max_time=0.1,
        disable_progressbar=True,
    )

    # Check if the file was created. If it wasn't, fail
    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")  # pragma: no cover

    with _hmclab.Samples(filename) as samples:
        samples.print_details()

    # Remove the file
    _os.remove(filename)
    if extension == "npy":
        _os.remove(f"{filename}.pkl")


@_pytest.mark.parametrize("sampler_class", sampler_classes)
@_pytest.mark.parametrize("distribution_class", distribution_classes)
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("proposals", proposals)
@_pytest.mark.parametrize("extension", extensions)
def test_samples_concat(
    sampler_class: _as,
    distribution_class: _ad,
    dimensions: int,
    proposals: int,
    extension: str,
):
    try:
        distribution: _ad = distribution_class.create_default(dimensions)
    except _InvalidCaseError:
        return _pytest.skip("Invalid case")

    sampler_instance = sampler_class()

    assert isinstance(sampler_instance, _as)

    filename_1 = f"temporary_file_1.{extension}"
    filename_2 = f"temporary_file_2.{extension}"
    filenames = [filename_1, filename_2]

    # Remove file before attempting to sample
    for filename in filenames:
        if _os.path.exists(filename):
            _os.remove(filename)  # pragma: no cover

        try:
            initial_model = distribution.generate()
        except:
            initial_model = _numpy.ones((distribution.dimensions, 1))

        sampler_instance.sample(
            filename,
            distribution,
            initial_model=initial_model,
            proposals=proposals,
            max_time=0.1,
            disable_progressbar=True,
        )

        # Check if the file was created. If it wasn't, fail
        if not _os.path.exists(filename):
            _pytest.fail("Samples file wasn't created")  # pragma: no cover

    combined_samples = _combine_samples(filenames)

    # The sample files also contain the misfit, so + 1
    assert combined_samples.shape[0] == distribution.dimensions + 1

    for filename in filenames:
        # Remove the file
        _os.remove(filename)
        if extension == "npy":
            _os.remove(f"{filename}.pkl")


def test_samples_exception_cases():
    filename = "non_existent_file.h5"

    try:
        with _hmclab.Samples(filename) as _:
            pass  # pragma: no cover
    except FileNotFoundError:
        pass
