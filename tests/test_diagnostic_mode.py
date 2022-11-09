"""A collection of integrated tests.
"""
from hmclab.Distributions import Normal
import os as _os
import hmclab as _hmclab
import uuid as _uuid


_ad = _hmclab.Distributions._AbstractDistribution
_as = _hmclab.Samplers._AbstractSampler


def test_basic_sampling():

    distribution = Normal.create_default(dimensions=10)

    sampler_instance = _hmclab.Samplers.HMC()

    unique_name = _uuid.uuid4().hex.upper()
    filename = f"temporary_file_{unique_name}.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)  # pragma: no cover

    sampler_instance.sample(
        filename,
        distribution,
        diagnostic_mode=True,
        proposals=10000,
        online_thinning=1,
        max_time=0.1,
        autotuning=True,
        disable_progressbar=True,
    )

    print()

    timers = sampler_instance.get_diagnostics()
    for timer in timers:
        print(timer)

    # Remove the file
    _os.remove(filename)
