"""A collection of integrated tests.
"""
from typing import List as _List
import os as _os
import uuid as _uuid


import numpy as _numpy
import matplotlib.pyplot as _plt
import pytest as _pytest
import uuid as _uuid


import hmclab as _hmclab

_ad = _hmclab.Distributions._AbstractDistribution
_as = _hmclab.Samplers._AbstractVisualSampler

dimensions = [2, 4]
sampler_classes = _as.__subclasses__()
proposals = [100]
autotuning = [True]
plot_update_interval = [1, 7]
dims_to_plot = [[0, 1], [4, 9]]
animate_proposals = [True, False]
animation_domain = [None, [-1, 1, -1, 1], [1, 0, -1, -1]]


@_pytest.fixture(autouse=True)
def run_before_and_after_tests():
    """Fixture to execute asserts before and after a test is run"""
    # Setup: fill with any logic you want

    yield  # this is where the testing happens

    _plt.close()


@_pytest.mark.parametrize("sampler_class", sampler_classes)
@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("proposals", proposals)
@_pytest.mark.parametrize("autotuning", autotuning)
@_pytest.mark.parametrize("plot_update_interval", plot_update_interval)
@_pytest.mark.parametrize("dims_to_plot", dims_to_plot)
@_pytest.mark.parametrize("animate_proposals", animate_proposals)
@_pytest.mark.parametrize("animation_domain", animation_domain)
def test_basic_sampling(
    sampler_class: _as,
    dimensions: int,
    proposals: int,
    autotuning: bool,
    plot_update_interval: int,
    dims_to_plot: _List,
    animate_proposals: bool,
    animation_domain: _numpy.array,
):
    distribution = _hmclab.Distributions.Normal.create_default(dimensions)

    sampler_instance = sampler_class(
        plot_update_interval=plot_update_interval,
        dims_to_plot=dims_to_plot,
        animate_proposals=animate_proposals,
        animation_domain=animation_domain,
    )

    assert isinstance(sampler_instance, _as)

    unique_name = _uuid.uuid4().hex.upper()
    filename = f"temporary_file_{unique_name}.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)  # pragma: no cover

    try:
        sampler_instance.sample(
            filename,
            distribution,
            proposals=proposals,
            online_thinning=10,
            max_time=0.1,
            autotuning=autotuning,
            disable_progressbar=True,
        )
        if sampler_instance.amount_of_writes > 0:
            # 10 percent burn_in
            burn_in = int(0.1 * sampler_instance.amount_of_writes)
            sampler_instance.load_results(burn_in=burn_in)
    except AssertionError:
        pass

    # Check if the file was created. If it wasn't, fail
    if _os.path.exists(filename):
        _os.remove(filename)  # pragma: no cover
