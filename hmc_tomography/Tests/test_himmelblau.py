import hmc_tomography as _hmc_tomography
import numpy as _numpy
import os as _os


def test_sampling_visualization_himmelblau():
    dist = _hmc_tomography.Distributions.Himmelblau(temperature=100)

    filename = "temporary_file.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)

    RWMH = _hmc_tomography.Samplers.RWMH()

    RWMH.sample(
        filename,
        dist,
        proposals=10000,
        samples_ram_buffer_size=100,
        online_thinning=10,
    )

    samples_written_expected = int(
        _numpy.floor(RWMH.current_proposal / RWMH.online_thinning) + 1
    )

    print(f"Samples written to disk: {samples_written_expected}")

    with _hmc_tomography.Post.Samples(filename) as samples:
        assert samples[:, :].shape == (3, samples_written_expected)

    _os.remove(filename)


def test_sampling_interrupt_himmelblau():
    dist = _hmc_tomography.Distributions.Himmelblau(temperature=100)

    filename = "temporary_file.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)

    RWMH = _hmc_tomography.Samplers.RWMH()

    RWMH.sample(
        filename,
        dist,
        proposals=100000,
        samples_ram_buffer_size=1000,
        online_thinning=100,
        max_time=2.0,
    )

    samples_written_expected = int(
        _numpy.floor(RWMH.current_proposal / RWMH.online_thinning) + 1
    )

    print(f"Samples written to disk: {samples_written_expected}")

    with _hmc_tomography.Post.Samples(filename) as samples:
        assert samples[:, :].shape == (3, samples_written_expected)

    _os.remove(filename)
