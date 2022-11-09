import os as _os

import numpy as _numpy
import uuid as _uuid


import hmclab as _hmclab


def test_leapfrog():
    dist = _hmclab.Distributions.Himmelblau(temperature=100)

    unique_name = _uuid.uuid4().hex.upper()
    filename = f"temporary_file_{unique_name}.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)  # pragma: no cover

    sampler = _hmclab.Samplers.HMC()

    sampler.sample(
        filename,
        dist,
        proposals=1000,
        stepsize=1.0,
        integrator="lf",
        max_time=0.1,
        disable_progressbar=True,
    )

    samples_written_expected = int(
        _numpy.floor(sampler.current_proposal / sampler.online_thinning) + 1
    )

    print(f"Samples written to disk: {samples_written_expected}")

    with _hmclab.Samples(filename) as samples:
        assert samples[:, :].shape == (3, samples_written_expected)

    _os.remove(filename)


def test_four_stage():
    dist = _hmclab.Distributions.Himmelblau(temperature=100)

    unique_name = _uuid.uuid4().hex.upper()
    filename = f"temporary_file_{unique_name}.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)  # pragma: no cover

    sampler = _hmclab.Samplers.HMC()

    sampler.sample(
        filename,
        dist,
        proposals=1000,
        stepsize=3.0,
        integrator="4s",
        max_time=0.1,
        disable_progressbar=True,
    )

    samples_written_expected = int(
        _numpy.floor(sampler.current_proposal / sampler.online_thinning) + 1
    )

    print(f"Samples written to disk: {samples_written_expected}")

    with _hmclab.Samples(filename) as samples:
        assert samples[:, :].shape == (3, samples_written_expected)

    _os.remove(filename)


def test_three_stage():
    dist = _hmclab.Distributions.Himmelblau(temperature=100)

    unique_name = _uuid.uuid4().hex.upper()
    filename = f"temporary_file_{unique_name}.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)  # pragma: no cover

    sampler = _hmclab.Samplers.HMC()

    sampler.sample(
        filename,
        dist,
        proposals=1000,
        stepsize=3.0,
        integrator="3s",
        max_time=0.1,
        disable_progressbar=True,
    )

    samples_written_expected = int(
        _numpy.floor(sampler.current_proposal / sampler.online_thinning) + 1
    )

    print(f"Samples written to disk: {samples_written_expected}")

    with _hmclab.Samples(filename) as samples:
        assert samples[:, :].shape == (3, samples_written_expected)

    _os.remove(filename)
