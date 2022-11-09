"""A collection of integrated tests.
"""
from hmclab.Distributions import Normal
import os as _os
import pytest as _pytest
import hmclab as _hmclab
import matplotlib.pyplot as _plt
import uuid as _uuid


_ad = _hmclab.Distributions._AbstractDistribution
_as = _hmclab.Samplers._AbstractSampler

dimensions = [1, 2, 10]
proposals = [1000]


@_pytest.mark.parametrize("dimensions", dimensions)
@_pytest.mark.parametrize("proposals", proposals)
def test_basic_sampling(
    dimensions: int,
    proposals: int,
):

    distribution = Normal.create_default(dimensions=dimensions)

    sampler_instance = _hmclab.Samplers.HMC()

    unique_name = _uuid.uuid4().hex.upper()
    filename = f"temporary_file_{unique_name}.h5"

    # Remove file before attempting to sample
    if _os.path.exists(filename):
        _os.remove(filename)  # pragma: no cover

    sampler_instance.sample(
        filename,
        distribution,
        proposals=proposals,
        online_thinning=1,
        max_time=0.1,
        autotuning=True,
        disable_progressbar=True,
    )

    # Check if the file was created. If it wasn't, fail
    if not _os.path.exists(filename):
        _pytest.fail("Samples file wasn't created")

    with _hmclab.Samples(filename) as samples:
        _hmclab.Visualization.marginal(samples, 0, 10, False, "r")
        _plt.close()

        try:
            _hmclab.Visualization.marginal_grid(
                samples,
                [0, 1],
                25,
                False,
                _plt.get_cmap("seismic"),
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

        _hmclab.Visualization.visualize_2_dimensions(samples, 0, 1, 25, False)
        _plt.close()

    # Remove the file
    _os.remove(filename)
