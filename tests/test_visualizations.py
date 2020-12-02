"""A collection of integrated tests.
"""
from hmc_tomography.Distributions import Normal
import os as _os
import pytest as _pytest
import hmc_tomography as _hmc_tomography
import matplotlib.pyplot as _plt

_ad = _hmc_tomography.Distributions._AbstractDistribution
_as = _hmc_tomography.Samplers._AbstractSampler

dimensions = [1, 2, 10]
proposals = [10000]


@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("proposals", proposals)
def test_basic_sampling(
    dimensions: int, proposals: int,
):

    distribution = Normal.create_default(dimensions=dimensions)

    sampler_instance = _hmc_tomography.Samplers.HMC()

    filename = "temporary_file.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)

    sampler_instance.sample(
        filename,
        distribution,
        proposals=proposals,
        online_thinning=1,
        max_time=1.0,
        autotuning=True,
    )

    # Check if the file was created. If it wasn't, fail
    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")

    with _hmc_tomography.Samples(filename) as samples:
        _hmc_tomography.Visualization.marginal(samples, 0, 10, False, "r")
        _plt.close()

        try:
            _hmc_tomography.Visualization.marginal_grid(
                samples, [0, 1], 25, False, _plt.get_cmap("seismic"),
            )
            _plt.close()

            # The previous is only allowed to succeed for dimensions > 1, so otherwise
            # fail.
            if dimensions == 1:
                _pytest.fail(
                    "This test should've failed. Was able to create a 2d plot with 1d "
                    "data."
                )
        except AssertionError:
            if dimensions != 1:
                _pytest.fail(
                    "This test should not have raise an AssertionError failed. Was not "
                    "able to create a 2d plot with at least 2d data."
                )

        _hmc_tomography.Visualization.visualize_2_dimensions(samples, 0, 1, 25, False)
        _plt.close()

    # Remove the file
    _os.remove(filename)
