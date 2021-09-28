"""A collection of integrated tests.
"""
from hmclab.Distributions import Normal
import os as _os
import hmclab as _hmclab

_ad = _hmclab.Distributions._AbstractDistribution
_as = _hmclab.Samplers._AbstractSampler


def test_basic_sampling():

    distribution = Normal.create_default(dimensions=10)

    sampler_instance = _hmclab.Samplers.HMC()

    filename = "temporary_file.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)

    sampler_instance.sample(
        filename,
        distribution,
        diagnostic_mode=True,
        proposals=10000,
        online_thinning=1,
        max_time=1.0,
        autotuning=True,
    )

    print()

    timers = sampler_instance.get_diagnostics()
    for timer in timers:
        print(timer)

    # Remove the file
    _os.remove(filename)
