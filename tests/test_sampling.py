"""A collection of integrated tests.
"""
from enum import auto
from fileinput import filename
import h5py as _h5py
import os as _os

import numpy as _numpy
import pytest as _pytest
import uuid as _uuid

import hmclab as _hmclab
from hmclab.Helpers.CustomExceptions import InvalidCaseError as _InvalidCaseError

_ad = _hmclab.Distributions._AbstractDistribution
_as = _hmclab.Samplers._AbstractSampler

dimensions = [1, 5, 50]
distribution_classes = _ad.__subclasses__()
sampler_classes = _as.__subclasses__()
sampler_classes.remove(_hmclab.Samplers._AbstractVisualSampler)
proposals = [10, 1000]
autotuning = [True, False]
extensions = ["h5", "npy"]


@_pytest.mark.parametrize("sampler_class", sampler_classes)
@_pytest.mark.parametrize("distribution_class", distribution_classes)
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("proposals", proposals)
@_pytest.mark.parametrize("autotuning", autotuning)
@_pytest.mark.parametrize("extension", extensions)
def test_basic_sampling(
    sampler_class: _as,
    distribution_class: _ad,
    dimensions: int,
    proposals: int,
    autotuning: bool,
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
        online_thinning=10,
        max_time=0.1,
        autotuning=autotuning,
        disable_progressbar=True,
    )
    if sampler_instance.amount_of_writes > 0:
        # 10 percent burn_in
        burn_in = int(0.1 * sampler_instance.amount_of_writes)
        sampler_instance.load_results(burn_in=burn_in)

    # Check if the file was created. If it wasn't, fail
    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")  # pragma: no cover

    # Remove the file
    _os.remove(filename)
    if extension == "npy":
        _os.remove(f"{filename}.pkl")


@_pytest.mark.parametrize("sampler_class", sampler_classes)
@_pytest.mark.parametrize("distribution_class", distribution_classes)
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("proposals", proposals)
@_pytest.mark.parametrize("autotuning", autotuning)
@_pytest.mark.parametrize("extension", extensions)
def test_samples_file(
    sampler_class: _as,
    distribution_class: _ad,
    dimensions: int,
    proposals: int,
    autotuning: bool,
    extension: str,
):
    try:
        distribution: _ad = distribution_class.create_default(dimensions)
    except _InvalidCaseError:
        return _pytest.skip("Invalid case")

    sampler_instance = sampler_class()

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
        initial_model=initial_model,
        proposals=proposals,
        max_time=0.1,
        autotuning=autotuning,
        disable_progressbar=True,
    )

    # Check if the file was created. If it wasn't, fail
    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")  # pragma: no cover

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

    # Remove the file
    _os.remove(filename)
    if extension == "npy":
        _os.remove(f"{filename}.pkl")


def test_improper_name():
    distribution: _ad = _hmclab.Distributions.Normal.create_default(10)
    sampler_instance = _hmclab.Samplers.RWMH()
    assert isinstance(sampler_instance, _as)

    unique_name = _uuid.uuid4().hex.upper()
    filename_sampler = f"temporary_file_{unique_name}"
    filename = f"temporary_file_{unique_name}.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)  # pragma: no cover

    try:
        initial_model = distribution.generate()
    except:
        initial_model = _numpy.ones((distribution.dimensions, 1))

    sampler_instance.sample(
        filename_sampler,
        distribution,
        initial_model=initial_model,
        proposals=100,
        max_time=0.1,
        autotuning=True,
        disable_progressbar=True,
    )
    if sampler_instance.amount_of_writes > 0:
        # 10 percent burn_in
        burn_in = int(0.1 * sampler_instance.amount_of_writes)
        sampler_instance.load_results(burn_in=burn_in)

    # Check if the file was created. If it wasn't, fail
    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")  # pragma: no cover
    # Remove the file
    _os.remove(filename)


@_pytest.mark.parametrize("sampler_class", sampler_classes)
def test_widget_functions(sampler_class: _as):
    distribution: _ad = _hmclab.Distributions.Normal.create_default(10)
    sampler_instance = sampler_class()
    assert isinstance(sampler_instance, _as)

    unique_name = _uuid.uuid4().hex.upper()
    filename = f"temporary_file_{unique_name}.h5"

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
        initial_model=initial_model,
        proposals=100,
        max_time=0.1,
        autotuning=True,
        disable_progressbar=True,
    )
    if sampler_instance.amount_of_writes > 0:
        # 10 percent burn_in
        burn_in = int(0.1 * sampler_instance.amount_of_writes)
        sampler_instance.load_results(burn_in=burn_in)
    # Check if the file was created. If it wasn't, fail
    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")  # pragma: no cover
    # Remove the file
    _os.remove(filename)

    # Test widget / printing
    # sampler_instance.print_results()

    # Test __str__()
    # print(str(sampler_instance))


@_pytest.mark.parametrize("diagnostic_mode", [True, False])
@_pytest.mark.parametrize("sampler_class", sampler_classes)
def test_diagnostic_mode(sampler_class: _as, diagnostic_mode: bool):
    distribution: _ad = _hmclab.Distributions.Normal.create_default(10)
    sampler_instance = sampler_class()
    assert isinstance(sampler_instance, _as)

    unique_name = _uuid.uuid4().hex.upper()
    filename = f"temporary_file_{unique_name}.h5"
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
        max_time=0.1,
        initial_model=initial_model,
        diagnostic_mode=diagnostic_mode,
        disable_progressbar=True,
    )

    if sampler_instance.amount_of_writes > 0:
        # 10 percent burn_in
        burn_in = int(0.1 * sampler_instance.amount_of_writes)
        sampler_instance.load_results(burn_in=burn_in)

    # Check if the file was created. If it wasn't, fail
    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")  # pragma: no cover

    # Remove the file
    _os.remove(filename)


@_pytest.mark.parametrize("seed", [None, 42])
@_pytest.mark.parametrize("sampler_class", sampler_classes)
def test_seed(sampler_class: _as, seed: float):
    distribution: _ad = _hmclab.Distributions.Normal.create_default(10)
    sampler_instance = sampler_class(seed=seed)
    assert isinstance(sampler_instance, _as)

    unique_name = _uuid.uuid4().hex.upper()
    filename = f"temporary_file_{unique_name}.h5"
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
        initial_model=initial_model,
        max_time=0.1,
        disable_progressbar=True,
    )

    if sampler_instance.amount_of_writes > 0:
        # 10 percent burn_in
        burn_in = int(0.1 * sampler_instance.amount_of_writes)
        sampler_instance.load_results(burn_in=burn_in)

    # Check if the file was created. If it wasn't, fail
    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")  # pragma: no cover
    # Remove the file
    _os.remove(filename)


@_pytest.mark.parametrize("sampler_class", sampler_classes)
def test_initial_model(sampler_class: _as):
    distribution: _ad = _hmclab.Distributions.Normal.create_default(10)
    sampler_instance = sampler_class()
    assert isinstance(sampler_instance, _as)

    unique_name = _uuid.uuid4().hex.upper()
    filename = f"temporary_file_{unique_name}.h5"
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
        initial_model=initial_model,
        max_time=0.1,
        disable_progressbar=True,
    )

    if sampler_instance.amount_of_writes > 0:
        # 10 percent burn_in
        burn_in = int(0.1 * sampler_instance.amount_of_writes)
        sampler_instance.load_results(burn_in=burn_in)

    # Check if the file was created. If it wasn't, fail
    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")  # pragma: no cover
    # Remove the file
    _os.remove(filename)


@_pytest.mark.parametrize("sampler_class", sampler_classes)
def test_preexisting_file(sampler_class: _as):
    distribution: _ad = _hmclab.Distributions.Normal.create_default(10)
    sampler_instance = sampler_class()
    assert isinstance(sampler_instance, _as)

    unique_name = _uuid.uuid4().hex.upper()
    filename = f"temporary_file_{unique_name}.h5"
    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)  # pragma: no cover

    open(filename, "a").close()

    try:
        initial_model = distribution.generate()
    except:
        initial_model = _numpy.ones((distribution.dimensions, 1))

    with _pytest.raises(FileExistsError):
        sampler_instance.sample(
            filename,
            distribution,
            initial_model=initial_model,
            max_time=0.1,
            disable_progressbar=True,
        )

    # Remove the file
    _os.remove(filename)


@_pytest.mark.parametrize("sampler_class", sampler_classes)
def test_plot(sampler_class: _as):
    distribution: _ad = _hmclab.Distributions.Normal.create_default(10)
    sampler_instance = sampler_class()
    assert isinstance(sampler_instance, _as)

    unique_name = _uuid.uuid4().hex.upper()
    filename = f"temporary_file_{unique_name}.h5"

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
        initial_model=initial_model,
        max_time=0.1,
        autotuning=True,
        disable_progressbar=True,
    )

    # Check if the file was created. If it wasn't, fail
    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")  # pragma: no cover

    # Remove the file
    _os.remove(filename)

    # sampler_instance.plot_stepsizes()
    # sampler_instance.plot_acceptance_rate()


@_pytest.mark.parametrize("sampler_class", sampler_classes)
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("parallel_chains", [2, 3])
@_pytest.mark.parametrize("exchange_interval", [1, 5])
@_pytest.mark.parametrize("exchange", [True, False])
def test_parallel_sampling(
    sampler_class: _as,
    dimensions: int,
    parallel_chains: int,
    exchange_interval: int,
    exchange: bool,
):
    distributions = [
        _hmclab.Distributions.Normal.create_default(dimensions)
        for _ in range(parallel_chains)
    ]
    sampler_instances = [sampler_class() for _ in range(parallel_chains)]
    unique_name = _uuid.uuid4().hex.upper()
    filenames = [f"temporary_file_{i}_{unique_name}.h5" for i in range(parallel_chains)]

    # Remove file before attempting to sample
    for filename in filenames:
        if _os.path.exists(filename):
            _os.remove(filename)  # pragma: no cover

    controller_instance = _hmclab.Samplers.ParallelSampleSMP()

    try:
        initial_model = distributions[0].generate()
    except:
        initial_model = _numpy.ones((distributions[0].dimensions, 1))

    controller_instance.sample(
        sampler_instances,
        filenames,
        distributions,
        initial_model=initial_model,
        exchange=exchange,
        exchange_interval=exchange_interval,
        overwrite_existing_files=True,
        kwargs={"disable_progressbar": True},
    )

    for filename in filenames:
        with _hmclab.Samples(filename) as samples:
            assert samples.numpy.shape == (dimensions + 1, 100)

    # Check if the file was created. If it wasn't, fail
    for filename in filenames:
        if not _os.path.exists(filename):
            _pytest.fail("Samples file wasn't created")  # pragma: no cover

        # Remove the file
        _os.remove(filename)

    # controller_instance.print_results()
